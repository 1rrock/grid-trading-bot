import os
import time
import json
import logging
import requests
import pyupbit
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import openai
import ast
import math

load_dotenv()

# 네트워크 장애 감지 및 자동 일시중지 설정
NETWORK_FAILURES = 0
NETWORK_FAILURE_THRESHOLD = int(os.getenv('NETWORK_FAILURE_THRESHOLD', '6'))
NETWORK_PAUSE_SLEEP_SECS = int(os.getenv('NETWORK_PAUSE_SLEEP_SECS', '60'))
PAUSE_TRADING_ON_NETWORK = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('grid_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def format_order_short(result: Dict) -> str:
    try:
        price = int(result.get('price')) if result.get('price') else None
        volume = float(result.get('volume')) if result.get('volume') else None
        uuid = result.get('uuid')
        locked = result.get('locked') or result.get('reserved_fee') or 0
        return f"uuid={uuid} | price={price:,}원 | vol={volume:.6f} | locked={float(locked):,.2f}원"
    except Exception:
        return str(result)

UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

INITIAL_CAPITAL = None
MIN_ORDER_AMOUNT = 5000

COINS = ['XRP', 'BTC', 'ETH']
COIN_CONFIGS = {
    'XRP': {'range': 0.04, 'grids': 4, 'portion': 0.5, 'min_order': 5000},
    'BTC': {'range': 0.03, 'grids': 2, 'portion': 0.3, 'min_order': 5000},
    'ETH': {'range': 0.03, 'grids': 2, 'portion': 0.2, 'min_order': 5000}
}
API_CALL_INTERVAL = 0.2
ORDER_TTL_HOURS = int(os.getenv('ORDER_TTL_HOURS', '12'))
FILL_REMAINING_MARKET = os.getenv('FILL_REMAINING_MARKET', 'false').lower() in ('1', 'true', 'yes')
MARKET_FILL_IF_LESS_THAN_RATIO = float(os.getenv('MARKET_FILL_IF_LESS_THAN_RATIO', '0.5'))
PREPLACE_ALL_GRIDS = False

# 🆕 동적 그리드 설정
ENABLE_DYNAMIC_GRID = True  # 동적 조정 활성화
RSI_OVERSOLD = 30  # RSI 과매도 기준
RSI_OVERBOUGHT = 70  # RSI 과매수 기준
PAUSE_BUY_ON_DOWNTREND = True  # 하락 추세 시 매수 중단

def calculate_ema(prices: np.ndarray, period: int) -> float:
    """EMA 계산"""
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_macd(prices: np.ndarray) -> Tuple[float, float, float]:
    """MACD 계산 (MACD, Signal, Histogram)"""
    if len(prices) < 26:
        return 0, 0, 0
    
    ema12 = calculate_ema(prices[-12:], 12)
    ema26 = calculate_ema(prices[-26:], 26)
    macd_line = ema12 - ema26
    
    # Signal line (MACD의 9일 EMA)
    if len(prices) < 35:
        signal_line = macd_line
    else:
        macd_values = []
        for i in range(9, len(prices)):
            e12 = calculate_ema(prices[i-12:i], 12)
            e26 = calculate_ema(prices[i-26:i], 26)
            macd_values.append(e12 - e26)
        signal_line = calculate_ema(np.array(macd_values[-9:]), 9)
    
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """RSI 계산"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def send_discord_notification(message: str, color: int = 3447003):
    try:
        data = {"embeds": [{"title": "🤖 그리드 트레이딩 봇", "description": message, "color": color, "timestamp": datetime.utcnow().isoformat()}]}
        if DISCORD_WEBHOOK_URL:
            requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=5)
    except Exception as e:
        logger.error(f"디스코드 알림 실패: {e}")


def with_retry(fn, *args, retries=3, backoff_factor=0.5, **kwargs):
    """재시도 래퍼: 네트워크/TLS 오류를 감지해 백오프 후 재시도하고, 연속 오류가 많으면 전체 트레이딩을 일시중지합니다."""
    global NETWORK_FAILURES, PAUSE_TRADING_ON_NETWORK
    attempt = 0
    while True:
        if PAUSE_TRADING_ON_NETWORK:
            # 네트워크 이슈로 인해 트레이딩이 일시중지된 상태입니다.
            raise RuntimeError("Trading paused due to repeated network/SSL failures")
        try:
            res = fn(*args, **kwargs)
            # 성공하면 실패 카운터 리셋
            if NETWORK_FAILURES != 0:
                NETWORK_FAILURES = 0
                if PAUSE_TRADING_ON_NETWORK:
                    PAUSE_TRADING_ON_NETWORK = False
                    logger.info("네트워크 복구 감지: 거래 재개")
            return res
        except Exception as e:
            attempt += 1
            is_ssl = False
            try:
                import ssl as _ssl
                if isinstance(e, _ssl.SSLError):
                    is_ssl = True
            except Exception:
                pass
            # 문자열에 SSLError 표시가 있는 경우도 체크
            if not is_ssl and isinstance(e, Exception) and ('SSLEOFError' in str(e) or 'SSL' in str(e) or 'ssl' in str(e).lower()):
                is_ssl = True

            if is_ssl:
                NETWORK_FAILURES += 1
                logger.warning(f"with_retry: SSL/네트워크 실패 감지 (count={NETWORK_FAILURES}) - {e}")
                if NETWORK_FAILURES >= NETWORK_FAILURE_THRESHOLD:
                    PAUSE_TRADING_ON_NETWORK = True
                    logger.error(f"네트워크 오류가 {NETWORK_FAILURES}회 발생하여 트레이딩을 일시중지합니다. 수동 확인 필요." )

            if attempt > retries:
                logger.error(f"with_retry: 함수 {fn.__name__} 실패 after {attempt} attempts: {e}")
                raise

            sleep_time = backoff_factor * (2 ** (attempt - 1))
            logger.warning(f"with_retry: {fn.__name__} 실패(시도 {attempt}/{retries}), {e} — {sleep_time:.1f}s 후 재시도")
            time.sleep(sleep_time)


def safe_get_current_price(market: str) -> Optional[float]:
    try:
        price = with_retry(pyupbit.get_current_price, market, retries=4, backoff_factor=0.3)
        return float(price) if price else None
    except Exception as e:
        logger.error(f"safe_get_current_price 실패: {e}")
        return None


def safe_get_ohlcv(market: str, interval: str = "minute60", count: int = 100):
    try:
        return with_retry(pyupbit.get_ohlcv, market, interval=interval, count=count, retries=3, backoff_factor=0.5)
    except Exception as e:
        logger.error(f"safe_get_ohlcv 실패: {e}")
        return None


def safe_get_order(upbit, uuid_or_market, **kwargs):
    try:
        return with_retry(upbit.get_order, uuid_or_market, **kwargs, retries=3, backoff_factor=0.4)
    except Exception as e:
        logger.warning(f"safe_get_order 실패: {e}")
        return None


def execute_buy_limit_order(upbit, ticker, price, volume):
    try:
        min_order_value = MIN_ORDER_AMOUNT
        order_value = price * volume
        if order_value < min_order_value:
            logger.warning(f"BUY_LIMIT_REJECTED - {ticker}: 주문 금액 부족 ({order_value:.0f}원)")
            return None
        
        result = upbit.buy_limit_order(ticker, price, volume)
        if result and 'uuid' in result:
            locked = result.get('locked') or result.get('reserved_fee') or 0
            logger.info(f"[ORDER][BUY] {ticker} | price={int(result.get('price')):,}원 | volume={float(result.get('volume')):.6f} | uuid={result.get('uuid')}")
            return result
        return None
    except Exception as e:
        logger.error(f"BUY_LIMIT_EXCEPTION - {ticker}: {e}")
        return None

def execute_sell_limit_order(upbit, ticker, price, volume):
    try:
        result = upbit.sell_limit_order(ticker, price, volume)
        if result and 'uuid' in result:
            logger.info(f"[ORDER][SELL] {ticker} | price={int(result.get('price')):,}원 | volume={float(result.get('volume')):.6f} | uuid={result.get('uuid')}")
            return result
        return None
    except Exception as e:
        logger.error(f"SELL_LIMIT_ERROR - {ticker}: {e}")
        return None

def execute_sell_market_order(upbit, ticker, volume):
    try:
        result = upbit.sell_market_order(ticker, volume)
        if result and 'uuid' in result:
            logger.info(f"[ORDER][SELL_MARKET] {ticker} | volume={float(result.get('volume')):.6f}")
            return result
        return None
    except Exception as e:
        logger.error(f"SELL_MARKET_ERROR - {ticker}: {e}")
        return None

def cancel_order_safe(upbit, uuid):
    try:
        return upbit.cancel_order(uuid)
    except Exception as e:
        logger.warning(f"cancel_order_safe 실패: {e}")
        return None

def is_peak_time():
    hour = datetime.now().hour
    return (9 <= hour <= 11) or (21 <= hour <= 23)

def round_price(price: float, coin: str) -> float:
    p = float(price)
    if p < 10000:
        tick = 1
    elif p < 100000:
        tick = 5
    elif p < 500000:
        tick = 10
    elif p < 1000000:
        tick = 50
    elif p < 2000000:
        tick = 100
    elif p < 5000000:
        tick = 500
    else:
        tick = 1000
    return int(round(p / tick) * tick)

class MarketIndicators:
    """🆕 시장 지표 분석 클래스"""
    def __init__(self, market: str):
        self.market = market
        self.ema_short = 0
        self.ema_long = 0
        self.macd = 0
        self.macd_signal = 0
        self.macd_histogram = 0
        self.rsi = 50
        self.last_update = None
    
    def update(self):
        """지표 업데이트 (1시간마다)"""
        try:
            if self.last_update and datetime.now() - self.last_update < timedelta(hours=1):
                return
            
            df = pyupbit.get_ohlcv(self.market, interval="minute60", count=100)
            if df is None or len(df) < 50:
                logger.warning(f"[{self.market}] 데이터 부족으로 지표 계산 불가")
                return
            
            prices = df['close'].values
            
            # EMA 계산
            self.ema_short = calculate_ema(prices[-50:], 50)
            self.ema_long = calculate_ema(prices, 200) if len(prices) >= 200 else self.ema_short
            
            # MACD 계산
            self.macd, self.macd_signal, self.macd_histogram = calculate_macd(prices)
            
            # RSI 계산
            self.rsi = calculate_rsi(prices, 14)
            
            self.last_update = datetime.now()
            
            logger.info(f"[{self.market}] 📊 지표 업데이트 | RSI: {self.rsi:.1f} | EMA50: {self.ema_short:.0f} | EMA200: {self.ema_long:.0f} | MACD: {self.macd:.2f}")
            
        except Exception as e:
            logger.error(f"[{self.market}] 지표 업데이트 실패: {e}")
    
    def is_downtrend(self) -> bool:
        """하락 추세 판단"""
        # EMA 데드크로스 또는 MACD 음수
        ema_downtrend = self.ema_short < self.ema_long
        macd_downtrend = self.macd < 0
        return ema_downtrend or macd_downtrend
    
    def is_oversold(self) -> bool:
        """과매도 상태 판단"""
        return self.rsi < RSI_OVERSOLD
    
    def is_overbought(self) -> bool:
        """과매수 상태 판단"""
        return self.rsi > RSI_OVERBOUGHT
    
    def should_pause_buy(self) -> bool:
        """매수 중단 여부"""
        if not PAUSE_BUY_ON_DOWNTREND:
            return False
        return self.is_downtrend() and not self.is_oversold()

class GridTradingBot:
    def __init__(self, upbit, coin: str, capital_per_coin: float, manager=None, portion=0.0):
        self.upbit = upbit
        self.coin = coin
        self.market = f"KRW-{coin}"
        self.initial_capital = capital_per_coin
        self.capital = capital_per_coin
        self.manager = manager
        self.portion = portion
        self.stop_trading = False
        
        config = COIN_CONFIGS.get(coin, {})
        self.price_range_percent = config.get('range', 0.05)
        self.min_order_amount = config.get('min_order', MIN_ORDER_AMOUNT)
        self.coin_name = config.get('name', coin)
        
        max_possible_grids = int(self.capital // self.min_order_amount)
        default_grids = config.get('grids', 10)
        # 최소 그리드 개수 보장
        self.grid_count = max(1, min(default_grids, max_possible_grids))

        self.grids: List[Dict] = []
        self.upper_price = 0.0
        self.lower_price = 0.0
        self.total_profit = 0.0
        self.trade_count = 0
        self.start_time = datetime.now()
        self.pending_orders = {}
        # pending_orders 파일 경로
        self.pending_file = f"pending_{self.coin}.json"
        # 시작 시 복원 시도
        self._load_pending_orders()

        # 🆕 기술적 지표 추가
        self.indicators = MarketIndicators(self.market)

        logger.info(f"[{self.coin}] 봇 초기화 (자본: {capital_per_coin:,.0f}원)")
    
    def get_current_price(self) -> Optional[float]:
        return safe_get_current_price(self.market)

    def _load_pending_orders(self):
        try:
            if os.path.exists(self.pending_file):
                with open(self.pending_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 검증 및 초기화
                    for k, v in data.items():
                        v.setdefault('filled_volume', 0.0)
                        # convert datetime strings to datetime if needed
                        if isinstance(v.get('created_at'), str):
                            try:
                                v['created_at'] = datetime.fromisoformat(v['created_at'])
                            except Exception:
                                v['created_at'] = datetime.now()
                    self.pending_orders = data
                    logger.info(f"[{self.coin}] pending_orders 복원: {len(self.pending_orders)}개")
                # reconcile with exchange state
                try:
                    for uuid, info in list(self.pending_orders.items()):
                        try:
                            od = safe_get_order(self.upbit, uuid)
                        except Exception:
                            od = None
                        if not od:
                            # couldn't fetch -> keep it for now
                            continue
                        executed = float(od.get('executed_volume') or od.get('executed_units') or 0)
                        remaining = float(od.get('remaining_volume') or 0)
                        # if fully filled on exchange but our pending still present, apply quick reconcile
                        if remaining <= 0 and executed > 0:
                            # apply to grid if possible
                            gi = info.get('grid_index')
                            if gi is not None and gi < len(self.grids):
                                grid = self.grids[gi]
                                side = info.get('side')
                                if side == 'buy':
                                    grid['has_position'] = True
                                    grid['volume'] = grid.get('volume', 0.0) + executed
                                elif side == 'sell':
                                    # sold: reduce or clear
                                    grid['volume'] = max(0.0, grid.get('volume', 0.0) - executed)
                            # remove pending
                            self.pending_orders.pop(uuid, None)
                        else:
                            # update filled_volume to match exchange to avoid double-apply
                            info['filled_volume'] = executed
                            info['expected_volume'] = remaining
                            self.pending_orders[uuid] = info
                    # save reconciled state
                    self._save_pending_orders()
                except Exception as e:
                    logger.warning(f"[{self.coin}] pending_orders 동기화 중 오류: {e}")
        except Exception as e:
            logger.error(f"[{self.coin}] pending_orders 복원 실패: {e}")

    def _save_pending_orders(self):
        try:
            to_save = {}
            for k, v in self.pending_orders.items():
                copy_v = v.copy()
                # datetime serialization
                if isinstance(copy_v.get('created_at'), datetime):
                    copy_v['created_at'] = copy_v['created_at'].isoformat()
                to_save[k] = copy_v
            with open(self.pending_file, 'w', encoding='utf-8') as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[{self.coin}] pending_orders 저장 실패: {e}")
    
    def initialize_grids(self) -> bool:
        try:
            current_price = self.get_current_price()
            if not current_price:
                return False
            
            # 지표 업데이트
            self.indicators.update()
            
            # 🆕 동적 범위 조정
            if ENABLE_DYNAMIC_GRID:
                if self.indicators.is_downtrend():
                    # 하락 추세: 범위 확대
                    self.price_range_percent *= 1.5
                    logger.info(f"[{self.coin}] 📉 하락 추세 감지 → 그리드 범위 확대: {self.price_range_percent:.1%}")
                elif self.indicators.is_overbought():
                    # 과매수: 범위 축소
                    self.price_range_percent *= 0.8
                    logger.info(f"[{self.coin}] 📈 과매수 감지 → 그리드 범위 축소: {self.price_range_percent:.1%}")
            
            self.upper_price = round_price(current_price * (1 + self.price_range_percent), self.coin)
            self.lower_price = round_price(current_price * (1 - self.price_range_percent), self.coin)
            
            step = (self.upper_price - self.lower_price) / self.grid_count
            self.grids = []
            
            for i in range(self.grid_count + 1):
                grid_price = self.lower_price + (step * i)
                price = round_price(grid_price, self.coin)
                self.grids.append({
                    'price': price,
                    'has_position': False,
                    'buy_order_uuid': None,
                    'sell_order_uuid': None,
                    'volume': 0.0,
                    'buy_price': 0.0
                })
            
            logger.info(f"[{self.coin}] 그리드 설정: {self.lower_price:,.0f} ~ {self.upper_price:,.0f} ({self.grid_count}개)")
            send_discord_notification(f"🎯 **{self.coin} 그리드 설정**\n• 범위: {self.lower_price:,.0f}~{self.upper_price:,.0f}원\n• RSI: {self.indicators.rsi:.1f}", color=5763719)
            return True
        except Exception as e:
            logger.error(f"[{self.coin}] 그리드 초기화 실패: {e}")
            return False
    
    def place_grid_orders(self):
        """주문 배치 (동적 조정 적용)"""
        try:
            current_price = self.get_current_price()
            if not current_price:
                return
            
            # 🆕 지표 업데이트
            self.indicators.update()
            
            # 🆕 하락 추세 시 매수 중단 체크
            if self.indicators.should_pause_buy():
                logger.warning(f"[{self.coin}] ⏸️  하락 추세 감지 - 매수 일시 중단 (RSI: {self.indicators.rsi:.1f})")
                # 매도 주문만 처리
                self._place_sell_orders()
                return
            
            balances = self.upbit.get_balances()
            available_balance = 0
            for balance in balances:
                if balance['currency'] == 'KRW':
                    available_balance = float(balance['balance'])
                    break
            
            # 매수 주문 배치
            for i, grid in enumerate(self.grids[:-1]):
                buy_price = grid['price']
                
                if PREPLACE_ALL_GRIDS:
                    place_buy = (not grid['has_position'] and not grid['buy_order_uuid'])
                else:
                    place_buy = (current_price > buy_price and not grid['has_position'] and not grid['buy_order_uuid'])
                
                if not place_buy:
                    reason = None
                    if grid['has_position']:
                        reason = 'already_has_position'
                    elif grid.get('buy_order_uuid'):
                        reason = 'existing_buy_order'
                    else:
                        # 가격 조건 미달 또는 기타
                        reason = 'price_condition_or_other'
                    logger.debug(f"[{self.coin}] 스킵된 그리드 i={i} price={buy_price} 이유={reason}")
                    continue
                
                amount_per_grid = self.capital / self.grid_count
                if amount_per_grid < self.min_order_amount or available_balance < amount_per_grid:
                    continue
                
                volume = amount_per_grid / buy_price
                volume = float(Decimal(str(volume)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
                
                result = execute_buy_limit_order(self.upbit, self.market, buy_price, volume)
                if result and 'uuid' in result:
                    grid['buy_order_uuid'] = result['uuid']
                    self.pending_orders[result['uuid']] = {
                        'grid_index': i,
                        'side': 'buy',
                        'created_at': datetime.now(),
                        'expected_volume': volume,
                        'filled_volume': 0.0
                    }
                    # persist pending orders to disk
                    try:
                        self._save_pending_orders()
                    except Exception:
                        logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")
                    available_balance -= amount_per_grid
                time.sleep(API_CALL_INTERVAL)
            
            # 매도 주문 배치
            self._place_sell_orders()
            
        except Exception as e:
            logger.error(f"[{self.coin}] 주문 배치 실패: {e}")
    
    def _place_sell_orders(self):
        """매도 주문만 별도 처리"""
        for i, grid in enumerate(self.grids[:-1]):
            if grid['has_position'] and grid['volume'] > 0 and not grid['sell_order_uuid']:
                sell_price = self.grids[i + 1]['price']
                
                result = execute_sell_limit_order(self.upbit, self.market, sell_price, grid['volume'])
                if result and 'uuid' in result:
                    grid['sell_order_uuid'] = result['uuid']
                    self.pending_orders[result['uuid']] = {
                        'grid_index': i,
                        'side': 'sell',
                        'created_at': datetime.now(),
                        'expected_volume': grid['volume'],
                        'filled_volume': 0.0
                    }
                    try:
                        self._save_pending_orders()
                    except Exception:
                        logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")
                    logger.info(f"[{self.coin}] ✅ 매도 주문 배치: {sell_price:,.0f}원")
                time.sleep(API_CALL_INTERVAL)
    
    def check_filled_orders(self):
        """체결 확인 및 부분체결 정확 반영

        동작 원리:
        - pending_orders에 기록된 각 주문(uuid)을 개별 조회해 executed_volume/remaining_volume을 확인합니다.
        - 이미 반영된(filled_volume) 만큼은 재반영하지 않도록 pending_orders에 'filled_volume'을 저장합니다.
        - 부분체결이면 grid['volume']에 증가분만 반영하고 pending_orders의 expected_volume/filled_volume을 갱신합니다.
        - 완전 체결이면 pending_orders 항목을 제거하고(혹은 sell 주문 생성) 그리드 상태를 업데이트합니다.
        """
        try:
            for uuid in list(self.pending_orders.keys()):
                info = self.pending_orders.get(uuid, {})
                grid_index = info.get('grid_index')
                side = info.get('side')

                if grid_index is None or grid_index >= len(self.grids):
                    self.pending_orders.pop(uuid, None)
                    continue

                grid = self.grids[grid_index]

                # 주문 상세 조회 시도 (pyupbit의 버전에 따라 동작이 다를 수 있음)
                order_detail = None
                try:
                    order_detail = self.upbit.get_order(uuid)
                except Exception:
                    # fallback: 현재 오픈 주문 목록에서 찾기
                    open_orders = self.upbit.get_order(self.market, state='wait') or []
                    order_detail = next((o for o in open_orders if o.get('uuid') == uuid), None)

                # 조회 실패 시 건너뜀
                if order_detail is None:
                    continue

                executed = float(order_detail.get('executed_volume') or order_detail.get('executed_units') or 0)
                remaining = float(order_detail.get('remaining_volume') or 0)

                # 이미 반영된 체결량(filled_volume)을 확인
                filled_before = float(info.get('filled_volume') or 0)
                new_filled = max(0.0, executed - filled_before)

                if new_filled > 0:
                    # 매수 체결이면 포지션 증가
                    if side == 'buy':
                        grid['has_position'] = True
                        grid['volume'] += new_filled
                        grid['buy_price'] = grid.get('buy_price') or grid['price']
                        logger.info(f"[{self.coin}] ✅ 매수 체결 반영: {grid['price']:,.0f}원 | 신규 체결량: {new_filled:.6f} | 총 보유: {grid['volume']:.6f}")

                        # 매수의 경우 잔여가 없으면(완전 체결) 즉시 매도 지정 주문 생성
                        if remaining <= 0:
                            # buy order fully filled -> remove pending buy
                            try:
                                self.pending_orders.pop(uuid, None)
                                try:
                                    self._save_pending_orders()
                                except Exception:
                                    logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")
                            except KeyError:
                                pass
                            grid['buy_order_uuid'] = None

                            # 즉시 매도 주문 생성
                            if grid_index + 1 < len(self.grids) and grid['volume'] > 0:
                                sell_price = self.grids[grid_index + 1]['price']
                                try:
                                    sell_res = execute_sell_limit_order(self.upbit, self.market, sell_price, grid['volume'])
                                    if sell_res and 'uuid' in sell_res:
                                        grid['sell_order_uuid'] = sell_res['uuid']
                                        self.pending_orders[sell_res['uuid']] = {
                                            'grid_index': grid_index,
                                            'side': 'sell',
                                            'created_at': datetime.now(),
                                            'expected_volume': grid['volume'],
                                            'filled_volume': 0.0
                                        }
                                        try:
                                            self._save_pending_orders()
                                        except Exception:
                                            logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")
                                        logger.info(f"[{self.coin}] 🚀 즉시 매도 주문 배치: {sell_price:,.0f}원 | 수량: {grid['volume']:.6f}")
                                except Exception as e:
                                    logger.error(f"[{self.coin}] 즉시 매도 주문 실패: {e}")
                        else:
                            # 부분 체결 -> pending_orders의 filled_volume, expected_volume 갱신
                            info['filled_volume'] = filled_before + new_filled
                            info['expected_volume'] = remaining
                            self.pending_orders[uuid] = info
                            try:
                                self._save_pending_orders()
                            except Exception:
                                logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")

                    elif side == 'sell':
                        # 매도 체결 반영
                        prev_vol = grid.get('volume', 0.0)
                        grid['volume'] = max(0.0, prev_vol - new_filled)
                        logger.info(f"[{self.coin}] ✅ 매도 체결 반영: {order_detail.get('price') or ''} | 체결량: {new_filled:.6f} | 남은 보유: {grid['volume']:.6f}")

                        if remaining <= 0:
                            # 완전 체결 -> 정리
                            try:
                                self.pending_orders.pop(uuid, None)
                                try:
                                    self._save_pending_orders()
                                except Exception:
                                    logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")
                            except KeyError:
                                pass
                            grid['sell_order_uuid'] = None
                            grid['has_position'] = False
                            grid['volume'] = 0.0
                            grid['buy_price'] = 0.0
                            # 수익 계산
                            try:
                                sell_price = float(order_detail.get('price') or self.grids[grid_index + 1]['price'])
                                profit = (sell_price - grid.get('buy_price', 0)) * new_filled
                                net_profit = profit * 0.999
                                self.total_profit += net_profit
                                self.trade_count += 1
                                logger.info(f"[{self.coin}] ✅ 매도 체결: {sell_price:,.0f}원 | 순수익: +{net_profit:,.0f}원")
                                send_discord_notification(f"**{self.coin} 매도 체결! 💰**\n• 수익: +{net_profit:,.0f}원\n• 누적: {self.total_profit:,.0f}원", color=3066993)
                            except Exception:
                                pass
                        else:
                            # 부분 체결 -> 갱신
                            info['filled_volume'] = filled_before + new_filled
                            info['expected_volume'] = remaining
                            self.pending_orders[uuid] = info
                            try:
                                self._save_pending_orders()
                            except Exception:
                                logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")

                else:
                    # executed == filled_before -> 주문 상태만 체크 (취소/완료 여부)
                    state = order_detail.get('state')
                    if state and state != 'wait' and float(order_detail.get('executed_volume') or 0) == 0:
                        # 무체결 종료(취소 등) -> pending 제거
                        logger.info(f"[{self.coin}] 주문 종료(무체결) 처리: uuid={uuid} state={state}")
                        try:
                            self.pending_orders.pop(uuid, None)
                            try:
                                self._save_pending_orders()
                            except Exception:
                                logger.warning(f"[{self.coin}] pending_orders 저장 실패(무시)")
                        except KeyError:
                            pass

        except Exception as e:
            logger.error(f"[{self.coin}] 체결 확인 실패: {e}")
    
    def run_cycle(self):
        self.place_grid_orders()
        time.sleep(API_CALL_INTERVAL)
        self.check_filled_orders()
    
    def get_stats(self) -> Dict:
        runtime = datetime.now() - self.start_time
        return {
            'coin': self.coin,
            'total_profit': self.total_profit,
            'trade_count': self.trade_count,
            'runtime': str(runtime).split('.')[0],
            'profit_rate': (self.total_profit / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            'rsi': self.indicators.rsi,
            'trend': '📉 하락' if self.indicators.is_downtrend() else '📈 상승'
        }

class GridBotManager:
    def __init__(self):
        self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.bots: List[GridTradingBot] = []
    
    def initialize(self) -> bool:
        try:
            balances = self.upbit.get_balances()
            krw_balance = 0
            for balance in balances:
                if balance['currency'] == 'KRW':
                    krw_balance = float(balance['balance'])
                    break
            
            global INITIAL_CAPITAL
            INITIAL_CAPITAL = krw_balance
            logger.info(f"현재 원화 잔고: {krw_balance:,.0f}원")
            
            for coin in COINS:
                config = COIN_CONFIGS.get(coin, {})
                portion = config.get('portion', 1.0 / len(COINS))
                capital_for_coin = INITIAL_CAPITAL * portion
                bot = GridTradingBot(self.upbit, coin, capital_for_coin, self, portion)
                if bot.initialize_grids():
                    self.bots.append(bot)
                    time.sleep(API_CALL_INTERVAL * 2)
            
            logger.info(f"총 {len(self.bots)}개 봇 초기화 완료")
            send_discord_notification(f"🚀 **그리드 봇 시작!**\n• 코인: {', '.join([bot.coin for bot in self.bots])}\n• 투자금: {INITIAL_CAPITAL:,}원", color=3447003)
            return len(self.bots) > 0
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            return False
    
    def run(self):
        logger.info("그리드 트레이딩 봇 시작!")
        last_report_time = datetime.now()
        
        try:
            while True:
                # 네트워크 일시중지 플래그가 켜져 있으면 주기적으로 네트워크 상태를 검사
                if PAUSE_TRADING_ON_NETWORK:
                    logger.warning("네트워크 이슈로 인해 트레이딩 일시중지 상태입니다. 네트워크 복구를 대기합니다...")
                    # 간단한 복구 체크: markets 엔드포인트 확인
                    try:
                        import requests
                        r = requests.get('https://api.upbit.com/v1/markets', timeout=5)
                        if r.status_code == 200:
                            # 복구로 간주
                            global NETWORK_FAILURES
                            NETWORK_FAILURES = 0
                            logger.info("네트워크 복구 감지: 트레이딩 재개")
                            # 재개
                            # fallthrough to normal loop
                        else:
                            logger.warning(f"복구 체크 실패 상태코드: {r.status_code}")
                    except Exception as e:
                        logger.warning(f"복구 체크 중 오류: {e}")
                    time.sleep(NETWORK_PAUSE_SLEEP_SECS)
                    continue

                for bot in self.bots:
                    try:
                        bot.run_cycle()
                        time.sleep(API_CALL_INTERVAL)
                    except Exception as e:
                        logger.error(f"[{bot.coin}] 사이클 오류: {e}")

                if datetime.now() - last_report_time > timedelta(hours=1):
                    self.send_report()
                    last_report_time = datetime.now()

                time.sleep(10)
        
        except KeyboardInterrupt:
            logger.info("봇 종료 중...")
            self.send_final_report()
    
    def send_report(self):
        try:
            total_profit = sum(bot.total_profit for bot in self.bots)
            total_trades = sum(bot.trade_count for bot in self.bots)
            
            report = f"📊 **1시간 리포트**\n\n"
            for bot in self.bots:
                stats = bot.get_stats()
                report += f"**{stats['coin']}**\n• 수익: {stats['total_profit']:,.0f}원 ({stats['profit_rate']:.2f}%)\n• 거래: {stats['trade_count']}회\n\n"
            
            report += f"**💰 전체**\n• 총 수익: {total_profit:,.0f}원\n• 총 거래: {total_trades}회"
            send_discord_notification(report, color=3447003)
        except Exception as e:
            logger.error(f"리포트 전송 실패: {e}")
    
    def send_final_report(self):
        try:
            total_profit = sum(bot.total_profit for bot in self.bots)
            total_trades = sum(bot.trade_count for bot in self.bots)
            
            report = f"🏁 **최종 리포트**\n\n"
            for bot in self.bots:
                stats = bot.get_stats()
                report += f"**{stats['coin']}**\n• 수익: {stats['total_profit']:,.0f}원\n• 거래: {stats['trade_count']}회\n\n"
            
            report += f"**💰 최종 성과**\n• 순수익: {total_profit:,.0f}원\n• 총 거래: {total_trades}회"
            send_discord_notification(report, color=15844367)
        except Exception as e:
            logger.error(f"최종 리포트 실패: {e}")

if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("그리드 트레이딩 봇 시작")
        logger.info("="*60)
        
        manager = GridBotManager()
        if manager.initialize():
            manager.run()
        else:
            logger.error("봇 초기화 실패!")
    except Exception as e:
        logger.error(f"메인 오류: {e}")