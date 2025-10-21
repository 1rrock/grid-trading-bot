import os
import time
import json
import logging
import requests
import pyupbit
import numpy as np
import openai
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# 네트워크 장애 설정
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

UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

INITIAL_CAPITAL = None
MIN_ORDER_AMOUNT = 5000
UPBIT_FEE_RATE = 0.0005  # 업비트 수수료 0.05%

# AI가 자동으로 설정하므로 portion만 설정
COINS = ['XRP', 'BTC', 'ETH']
COIN_CONFIGS = {
    'XRP': {'portion': 0.4, 'min_order': 5000},
    'BTC': {'portion': 0.35, 'min_order': 5000},
    'ETH': {'portion': 0.25, 'min_order': 5000}
}
API_CALL_INTERVAL = 0.2
ORDER_TTL_HOURS = int(os.getenv('ORDER_TTL_HOURS', '12'))

# 동적 그리드 설정
ENABLE_DYNAMIC_GRID = True
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
PAUSE_BUY_ON_DOWNTREND = True

def execute_sell_market_order(upbit, ticker, volume):
    """시장가 매도"""
    try:
        result = upbit.sell_market_order(ticker, volume)
        if result and 'uuid' in result:
            logger.info(f"[시장가 매도] {ticker} | volume={float(result.get('volume')):.6f}")
            return result
        return None
    except Exception as e:
        logger.error(f"시장가 매도 실패: {e}")
        return None

def calculate_ema(prices: np.ndarray, period: int) -> float:
    if len(prices) < period:
        return prices[-1] if len(prices) > 0 else 0
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """✅ 정확한 RSI 계산 (Wilder's Smoothing)"""
    if len(prices) < period + 1:
        return 50
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # 🔧 첫 번째 평균은 단순 평균
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # 🔧 이후는 Wilder's Smoothing (EMA와 유사)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
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

def is_network_error(e: Exception) -> bool:
    error_str = str(e).lower()
    error_types = ['ssl', 'timeout', 'connection', 'network', 'unreachable']
    return any(err in error_str for err in error_types) or \
            any(err in str(type(e).__name__).lower() for err in error_types)

def with_retry(fn, *args, retries=3, backoff_factor=0.5, **kwargs):
    global NETWORK_FAILURES, PAUSE_TRADING_ON_NETWORK
    attempt = 0
    while True:
        if PAUSE_TRADING_ON_NETWORK:
            raise RuntimeError("Trading paused due to network failures")
        try:
            res = fn(*args, **kwargs)
            if NETWORK_FAILURES != 0:
                NETWORK_FAILURES = 0
                logger.info("네트워크 복구")
            return res
        except Exception as e:
            attempt += 1
            if is_network_error(e):
                NETWORK_FAILURES += 1
                if NETWORK_FAILURES >= NETWORK_FAILURE_THRESHOLD:
                    PAUSE_TRADING_ON_NETWORK = True
                    logger.error(f"네트워크 오류 {NETWORK_FAILURES}회 - 트레이딩 중지")
            if attempt > retries:
                logger.error(f"{fn.__name__} 실패: {e}")
                raise
            sleep_time = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_time)

def safe_get_current_price(market: str) -> Optional[float]:
    try:
        price = with_retry(pyupbit.get_current_price, market, retries=4, backoff_factor=0.3)
        return float(price) if price else None
    except Exception as e:
        logger.error(f"가격 조회 실패: {e}")
        return None

def safe_get_order(upbit, uuid_or_market, **kwargs):
    try:
        return with_retry(upbit.get_order, uuid_or_market, **kwargs, retries=3, backoff_factor=0.4)
    except Exception as e:
        return None

def execute_buy_limit_order(upbit, ticker, price, volume):
    try:
        order_value = price * volume
        if order_value < MIN_ORDER_AMOUNT:
            logger.warning(f"매수 거부 - 최소금액 미달: {order_value:.0f}원")
            return None
        result = upbit.buy_limit_order(ticker, price, volume)
        if result and 'uuid' in result:
            logger.info(f"[매수주문] {ticker} | {int(result.get('price')):,}원 | {float(result.get('volume')):.6f}")
            return result
        return None
    except Exception as e:
        logger.error(f"매수 실패: {e}")
        return None

def execute_sell_limit_order(upbit, ticker, price, volume):
    try:
        result = upbit.sell_limit_order(ticker, price, volume)
        if result and 'uuid' in result:
            logger.info(f"[매도주문] {ticker} | {int(result.get('price')):,}원 | {float(result.get('volume')):.6f}")
            return result
        return None
    except Exception as e:
        logger.error(f"매도 실패: {e}")
        return None

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
    def __init__(self, market: str):
        self.market = market
        self.rsi = 50
        self.ema_short = 0
        self.ema_long = 0
        self.last_update = None
    
    def update(self):
        """✅ 업데이트 주기 단축 (1시간 → 10분)"""
        try:
            # 🔧 10분마다 업데이트
            if self.last_update and datetime.now() - self.last_update < timedelta(minutes=10):
                return
            
            df = pyupbit.get_ohlcv(self.market, interval="minute60", count=100)
            if df is None or len(df) < 50:
                return
            
            prices = df['close'].values
            self.ema_short = calculate_ema(prices[-50:], 50)
            self.ema_long = calculate_ema(prices, 200) if len(prices) >= 200 else self.ema_short
            self.rsi = calculate_rsi(prices, 14)
            self.last_update = datetime.now()
            logger.info(f"[{self.market}] RSI: {self.rsi:.1f} | EMA50: {self.ema_short:.0f}")
        except Exception as e:
            logger.error(f"지표 업데이트 실패: {e}")
    
    def is_downtrend(self) -> bool:
        return self.ema_short < self.ema_long
    
    def is_oversold(self) -> bool:
        return self.rsi < RSI_OVERSOLD
    
    def should_pause_buy(self) -> bool:
        if not PAUSE_BUY_ON_DOWNTREND:
            return False
        return self.is_downtrend() and not self.is_oversold()

class GridTradingBot:
    def __init__(self, upbit, coin: str, capital_per_coin: float, manager=None):
        self.upbit = upbit
        self.coin = coin
        self.market = f"KRW-{coin}"
        self.initial_capital = capital_per_coin
        self.capital = capital_per_coin
        self.stop_trading = False  # 🔧 추가
        
        config = COIN_CONFIGS.get(coin, {})
        
        # 🔧 AI 파라미터 추천 받기
        logger.info(f"[{coin}] 🤖 AI에게 최적 파라미터 요청 중...")
        ai_params = self.get_ai_optimized_params(coin)
        
        if ai_params:
            # AI 추천 사용
            self.price_range_percent = ai_params['range']
            self.grid_count = ai_params['grids']
            self.stop_loss_threshold = ai_params['stop_loss']
            self.take_profit_threshold = ai_params['take_profit']
            self.rebalance_threshold = ai_params['rebalance_threshold']
            logger.info(f"[{coin}] ✅ AI 추천 파라미터 적용")
        else:
            # 설정 파일 or 기본값 사용
            default = self.get_default_params(coin)
            self.price_range_percent = config.get('range', default['range'])
            self.grid_count = config.get('grids', default['grids'])
            self.stop_loss_threshold = config.get('stop_loss', default['stop_loss'])
            self.take_profit_threshold = config.get('take_profit', default['take_profit'])
            self.rebalance_threshold = config.get('rebalance_threshold', default['rebalance_threshold'])
            logger.warning(f"[{coin}] ⚠️ 기본 파라미터 사용")
        
        self.min_order_amount = config.get('min_order', MIN_ORDER_AMOUNT)
        
        self.grids: List[Dict] = []
        self.upper_price = 0.0
        self.lower_price = 0.0
        self.total_profit = 0.0
        self.trade_count = 0
        self.start_time = datetime.now()
        self.pending_orders = {}
        self.pending_file = f"pending_{self.coin}.json"
        self._load_pending_orders()
        
        self.indicators = MarketIndicators(self.market)
        logger.info(f"[{self.coin}] 봇 초기화 완료 (자본: {capital_per_coin:,.0f}원)")

    def get_ai_optimized_params(self, coin: str) -> Dict:
        """✅ AI 응답 검증 + 캐싱"""
        try:
            # 🔧 캐시 파일 체크 (1시간 이내 재사용)
            cache_file = f"ai_params_{coin}.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    cache_time = datetime.fromisoformat(cached['timestamp'])
                    if datetime.now() - cache_time < timedelta(hours=1):
                        logger.info(f"[{coin}] 📦 캐시된 AI 파라미터 사용")
                        return cached['params']
            
            # 시장 데이터 수집
            df = pyupbit.get_ohlcv(f"KRW-{coin}", interval="minute60", count=168)
            if df is None:
                logger.error(f"[{coin}] 데이터 수집 실패")
                return None
            
            current_price = self.get_current_price()
            prices = df['close'].values
            volumes = df['volume'].values
            
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * 100
            
            week_high = df['high'].max()
            week_low = df['low'].min()
            price_range = (week_high - week_low) / week_low * 100
            
            volume_change = (volumes[-24:].mean() / volumes[:-24].mean() - 1) * 100
            rsi = calculate_rsi(prices, 14)
            
            ma7 = np.mean(prices[-7:])
            ma25 = np.mean(prices[-25:])
            trend = "상승" if ma7 > ma25 else "하락"
            
            prompt = f"""
            당신은 암호화폐 그리드 트레이딩 전문가입니다. 다음 시장 데이터를 분석하여 최적의 그리드 매매 파라미터를 추천해주세요.

            **코인**: {coin}
            **현재가**: {current_price:,.0f}원
            **주간 변동성**: {volatility:.2f}%
            **주간 가격 범위**: {price_range:.2f}% (최고 {week_high:,.0f}원, 최저 {week_low:,.0f}원)
            **RSI**: {rsi:.1f}
            **추세**: {trend} (MA7: {ma7:,.0f}, MA25: {ma25:,.0f})
            **거래량 변화**: {volume_change:+.1f}%

            다음 파라미터를 JSON 형식으로만 응답해주세요:
            {{
            "range": 0.05,              // 그리드 범위 (0.02~0.15 사이)
            "grids": 10,                // 그리드 개수 (5~20 사이)
            "stop_loss": -0.15,         // 손절 기준 (-0.30~-0.10 사이)
            "take_profit": 0.30,        // 익절 기준 (0.20~0.50 사이)
            "rebalance_threshold": 0.05, // 리밸런싱 (0.03~0.10 사이)
            "reason": "변동성 분석 결과..."
            }}

            **중요**: 반드시 JSON만 출력하세요.
            """
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a crypto trading expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # JSON 추출
            if "```json" in ai_response:
                ai_response = ai_response.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_response:
                ai_response = ai_response.split("```")[1].split("```")[0].strip()
            
            params = json.loads(ai_response)
            
            # 🔧 파라미터 검증
            if not (0.02 <= params['range'] <= 0.15):
                logger.warning(f"[{coin}] AI 범위 비정상: {params['range']} → 기본값 사용")
                return None
            
            if not (5 <= params['grids'] <= 20):
                logger.warning(f"[{coin}] AI 그리드 수 비정상: {params['grids']} → 기본값 사용")
                return None
            
            if not (-0.30 <= params['stop_loss'] <= -0.10):
                logger.warning(f"[{coin}] AI 손절 비정상: {params['stop_loss']} → 기본값 사용")
                return None
            
            if not (0.20 <= params['take_profit'] <= 0.50):
                logger.warning(f"[{coin}] AI 익절 비정상: {params['take_profit']} → 기본값 사용")
                return None
            
            if not (0.03 <= params['rebalance_threshold'] <= 0.10):
                logger.warning(f"[{coin}] AI 리밸런싱 비정상: {params['rebalance_threshold']} → 기본값 사용")
                return None
            
            # 🔧 캐시 저장
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'params': params
                }, f, indent=2)
            
            logger.info(f"[{coin}] 🤖 AI 추천 파라미터:")
            logger.info(f"  • 그리드 범위: ±{params['range']*100:.1f}%")
            logger.info(f"  • 그리드 수: {params['grids']}개")
            logger.info(f"  • 손절: {params['stop_loss']*100:.1f}%")
            logger.info(f"  • 익절: {params['take_profit']*100:.1f}%")
            logger.info(f"  • 리밸런싱: {params['rebalance_threshold']*100:.1f}%")
            logger.info(f"  • 이유: {params.get('reason', 'N/A')}")
            
            send_discord_notification(
                f"🤖 **{coin} AI 파라미터 추천**\n"
                f"• 범위: ±{params['range']*100:.1f}%\n"
                f"• 그리드: {params['grids']}개\n"
                f"• 손절/익절: {params['stop_loss']*100:.1f}% / {params['take_profit']*100:.1f}%\n"
                f"• 변동성: {volatility:.2f}% | RSI: {rsi:.1f}\n"
                f"• {params.get('reason', '')}",
                color=3447003
            )
            
            return params
            
        except json.JSONDecodeError as e:
            logger.error(f"[{coin}] AI 응답 JSON 파싱 실패: {e}")
            logger.error(f"AI 응답: {ai_response}")
            return None
        except Exception as e:
            logger.error(f"[{coin}] AI 파라미터 추천 실패: {e}")
            return None

    def get_default_params(self, coin: str) -> Dict:
        """AI 실패 시 기본 파라미터"""
        return {
            'range': 0.05,
            'grids': 10,
            'stop_loss': -0.15,
            'take_profit': 0.25,
            'rebalance_threshold': 0.05,
            'reason': 'AI 추천 실패로 기본값 사용'
        }

    def _reconcile_pending_with_grids(self):
        """✅ 재시작 시 pending_orders를 grids에 동기화"""
        try:
            if not self.pending_orders:
                logger.debug(f"[{self.coin}] pending_orders 없음 (신규 시작)")
                return
            
            logger.info(f"[{self.coin}] pending_orders 동기화 시작: {len(self.pending_orders)}개")
            
            restored = 0
            removed = 0
            
            for uuid, info in list(self.pending_orders.items()):
                grid_index = info.get('grid_index')
                side = info.get('side')
                
                # 🔧 유효성 검사
                if grid_index is None or grid_index >= len(self.grids):
                    logger.warning(f"[{self.coin}] 잘못된 grid_index: {grid_index}, pending 삭제")
                    self.pending_orders.pop(uuid, None)
                    removed += 1
                    continue
                
                grid = self.grids[grid_index]
                
                # 주문 상태 확인
                try:
                    order_detail = self.upbit.get_order(uuid)
                    state = order_detail.get('state')
                    executed = float(order_detail.get('executed_volume') or 0)
                    remaining = float(order_detail.get('remaining_volume') or 0)
                    
                    # 🔧 취소/만료 주문 제거
                    if state in ['cancel', 'done'] and executed == 0:
                        logger.info(f"[{self.coin}] 만료된 pending 제거: {side} Grid{grid_index}")
                        self.pending_orders.pop(uuid, None)
                        removed += 1
                        continue
                    
                    # 🔧 grids에 매핑
                    if side == 'buy':
                        grid['buy_order_uuid'] = uuid
                        if executed > 0:
                            grid['has_position'] = True
                            grid['volume'] = executed
                            grid['buy_price'] = grid['price']
                        info['filled_volume'] = executed
                        info['expected_volume'] = remaining
                        restored += 1
                        logger.debug(f"[{self.coin}] 매수 주문 복원: Grid{grid_index} (체결: {executed:.6f})")
                        
                    elif side == 'sell':
                        grid['sell_order_uuid'] = uuid
                        sell_price = float(info.get('price') or order_detail.get('price') or 0)
                        info['price'] = sell_price
                        info['filled_volume'] = executed
                        info['expected_volume'] = remaining
                        
                        # 부분 체결된 경우 volume 조정
                        if executed > 0:
                            grid['volume'] = max(0, grid.get('volume', 0) - executed)
                        restored += 1
                        logger.debug(f"[{self.coin}] 매도 주문 복원: Grid{grid_index} (체결: {executed:.6f})")
                    
                    self.pending_orders[uuid] = info
                    
                except Exception as e:
                    logger.warning(f"[{self.coin}] 주문 조회 실패: uuid={uuid[:8]}... - {e}")
                    # 조회 실패한 pending은 유지 (다음 사이클에서 재확인)
            
            self._save_pending_orders()
            logger.info(f"[{self.coin}] pending 동기화 완료: 복원 {restored}개 | 제거 {removed}개")
            
        except Exception as e:
            logger.error(f"[{self.coin}] pending 동기화 실패: {e}")

    def emergency_close_all(self):
        """✅ 긴급 전량 청산 + 안전 확인"""
        try:
            logger.warning(f"[{self.coin}] 🚨 긴급 청산 시작...")
            
            # 1. 주문 취소
            cancelled_count = 0
            for uuid in list(self.pending_orders.keys()):
                try:
                    self.upbit.cancel_order(uuid)
                    cancelled_count += 1
                    time.sleep(API_CALL_INTERVAL)
                except Exception as e:
                    logger.warning(f"주문 취소 실패 ({uuid[:8]}...): {e}")
            
            logger.info(f"[{self.coin}] 취소된 주문: {cancelled_count}개")
            self.pending_orders = {}
            self._save_pending_orders()
            
            # 2. 시장가 매도
            balances = self.upbit.get_balances()
            for balance in balances:
                if balance['currency'] == self.coin:
                    volume = float(balance['balance'])
                    if volume > 0:
                        logger.info(f"[{self.coin}] 🔴 긴급 매도: {volume:.6f}")
                        result = execute_sell_market_order(self.upbit, self.market, volume)
                        
                        # 🔧 매도 확인
                        if result:
                            time.sleep(1)
                            final_balances = self.upbit.get_balances()
                            remaining = 0
                            for b in final_balances:
                                if b['currency'] == self.coin:
                                    remaining = float(b['balance'])
                            
                            if remaining > 0:
                                logger.error(f"[{self.coin}] ⚠️ 청산 미완료: {remaining:.6f} 남음")
                            else:
                                logger.info(f"[{self.coin}] ✅ 청산 완료")
                        else:
                            logger.error(f"[{self.coin}] ❌ 긴급 매도 실패")
            
            # 3. 그리드 초기화
            for grid in self.grids:
                grid['has_position'] = False
                grid['volume'] = 0.0
                grid['buy_order_uuid'] = None
                grid['sell_order_uuid'] = None
            
        except Exception as e:
            logger.error(f"[{self.coin}] 긴급 청산 실패: {e}")
            send_discord_notification(
                f"❌ **{self.coin} 긴급 청산 실패**\n"
                f"• 오류: {str(e)}\n"
                f"• 수동 확인 필요!",
                color=15158332
            )

    def cancel_all_orders(self):
        """모든 미체결 주문 취소"""
        try:
            for uuid in list(self.pending_orders.keys()):
                try:
                    self.upbit.cancel_order(uuid)
                    logger.info(f"[{self.coin}] 주문 취소: {uuid[:8]}...")
                    time.sleep(API_CALL_INTERVAL)
                except Exception as e:
                    logger.warning(f"주문 취소 실패: {e}")
            
            self.pending_orders = {}
            self._save_pending_orders()
            
            # 그리드 상태 초기화
            for grid in self.grids:
                grid['buy_order_uuid'] = None
                grid['sell_order_uuid'] = None
                
        except Exception as e:
            logger.error(f"전체 취소 실패: {e}")

    def check_grid_rebalancing(self):
        """✅ 개선: 점진적 조정 + 초기자본 업데이트"""
        try:
            current_price = self.get_current_price()
            if not current_price:
                return
            
            # 🔧 상단/하단 기준 개선
            grid_range = self.upper_price - self.lower_price
            upper_trigger = self.upper_price - (grid_range * self.rebalance_threshold)
            lower_trigger = self.lower_price + (grid_range * self.rebalance_threshold)
            
            need_rebalance = False
            reason = ""
            rebalance_type = None
            
            if current_price > upper_trigger:
                need_rebalance = True
                rebalance_type = 'upper'
                reason = f"상단 근접 (현재가: {current_price:,.0f}원 > 기준: {upper_trigger:,.0f}원)"
            elif current_price < lower_trigger:
                need_rebalance = True
                rebalance_type = 'lower'
                reason = f"하단 근접 (현재가: {current_price:,.0f}원 < 기준: {lower_trigger:,.0f}원)"
            
            if not need_rebalance:
                return
            
            logger.warning(f"[{self.coin}] ⚠️ 그리드 리밸런싱 필요: {reason}")
            
            # 🔧 현재 자산 계산
            balances = self.upbit.get_balances()
            krw_balance = 0
            coin_balance = 0
            
            for balance in balances:
                if balance['currency'] == 'KRW':
                    krw_balance = float(balance['balance'])
                elif balance['currency'] == self.coin:
                    coin_balance = float(balance['balance'])
            
            total_value = krw_balance + (coin_balance * current_price)
            profit_rate = (total_value / self.initial_capital - 1) * 100
            
            # 🔧 하단 돌파 시 손절 우선 체크
            if rebalance_type == 'lower':
                config = COIN_CONFIGS.get(self.coin, {})
                stop_loss = config.get('stop_loss', -0.20)
                
                if profit_rate / 100 <= self.stop_loss_threshold:
                    logger.error(f"[{self.coin}] 🛑 손절 조건 충족 - 리밸런싱 대신 청산")
                    self.emergency_close_all()
                    self.stop_trading = True
                    send_discord_notification(
                        f"🛑 **{self.coin} 손절 실행**\n"
                        f"• 손실률: {profit_rate:.2f}%\n"
                        f"• 사유: 하단 돌파 + 손절 조건",
                        color=15158332
                    )
                    return
            
            # 기존 주문 모두 취소
            self.cancel_all_orders()
            
            # 🔧 초기 자본 업데이트 (리밸런싱 후 새 기준점)
            self.capital = total_value
            self.initial_capital = total_value  # 🔧 손익률 기준 재설정
            
            logger.info(f"[{self.coin}] 📊 현재 자산: {total_value:,.0f}원 (수익률: {profit_rate:+.2f}%)")
            
            # AI 파라미터 재요청
            logger.info(f"[{self.coin}] 🤖 리밸런싱을 위한 AI 파라미터 재요청...")
            ai_params = self.get_ai_optimized_params(self.coin)
            
            if ai_params:
                self.price_range_percent = ai_params['range']
                self.grid_count = ai_params['grids']
                self.stop_loss_threshold = ai_params['stop_loss']
                self.take_profit_threshold = ai_params['take_profit']
                self.rebalance_threshold = ai_params['rebalance_threshold']
            
            # 그리드 재초기화
            self.grids = []
            self.pending_orders = {}
            self.initialize_grids()
            
            send_discord_notification(
                f"🔄 **{self.coin} 그리드 리밸런싱**\n"
                f"• 사유: {reason}\n"
                f"• 현재가: {current_price:,.0f}원\n"
                f"• 총 자산: {total_value:,.0f}원\n"
                f"• 기존 수익률: {profit_rate:+.2f}%\n"
                f"• 새 범위: ±{self.price_range_percent*100:.1f}%\n"
                f"• 새 기준점으로 재시작",
                color=15844367
            )
            
        except Exception as e:
            logger.error(f"[{self.coin}] 리밸런싱 실패: {e}")
    
    def get_current_price(self) -> Optional[float]:
        return safe_get_current_price(self.market)
    
    def _load_pending_orders(self):
        try:
            if os.path.exists(self.pending_file):
                with open(self.pending_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        v.setdefault('filled_volume', 0.0)
                        if isinstance(v.get('created_at'), str):
                            try:
                                v['created_at'] = datetime.fromisoformat(v['created_at'])
                            except:
                                v['created_at'] = datetime.now()
                    self.pending_orders = data
                    logger.info(f"[{self.coin}] pending 복원: {len(self.pending_orders)}개")
        except Exception as e:
            logger.error(f"pending 복원 실패: {e}")
    
    def _save_pending_orders(self):
        try:
            to_save = {}
            for k, v in self.pending_orders.items():
                copy_v = v.copy()
                if isinstance(copy_v.get('created_at'), datetime):
                    copy_v['created_at'] = copy_v['created_at'].isoformat()
                to_save[k] = copy_v
            with open(self.pending_file, 'w', encoding='utf-8') as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"pending 저장 실패: {e}")
    
    def initialize_grids(self) -> bool:
        """✅ 개선: pending 복원 추가"""
        try:
            current_price = self.get_current_price()
            if not current_price:
                return False
            
            self.indicators.update()
            
            # 현재가 기준 상하 범위 설정
            self.upper_price = round_price(current_price * (1 + self.price_range_percent), self.coin)
            self.lower_price = round_price(current_price * (1 - self.price_range_percent), self.coin)
            
            # 균등 그리드 생성
            step = (self.upper_price - self.lower_price) / self.grid_count
            prices = []
            for i in range(self.grid_count + 1):
                price = round_price(self.lower_price + step * i, self.coin)
                if not prices or price != prices[-1]:
                    prices.append(price)
            
            # 그리드 초기화
            self.grids = []
            for i, price in enumerate(prices):
                self.grids.append({
                    'index': i,
                    'price': price,
                    'has_position': False,
                    'volume': 0.0,
                    'buy_price': 0.0,
                    'buy_order_uuid': None,
                    'sell_order_uuid': None
                })
            
            # 🔧 pending_orders 복원
            self._reconcile_pending_with_grids()
            
            logger.info(f"[{self.coin}] ✅ 그리드 생성 완료")
            logger.info(f"  • 현재가: {current_price:,.0f}원")
            logger.info(f"  • 범위: {self.lower_price:,.0f} ~ {self.upper_price:,.0f}원")
            logger.info(f"  • 그리드 수: {len(self.grids)}개")
            
            send_discord_notification(
                f"🎯 **{self.coin} 그리드 설정**\n"
                f"• 현재가: {current_price:,.0f}원\n"
                f"• 범위: {self.lower_price:,.0f}~{self.upper_price:,.0f}원\n"
                f"• 그리드: {len(self.grids)}개\n"
                f"• RSI: {self.indicators.rsi:.1f}", 
                color=5763719
            )
            return True
        except Exception as e:
            logger.error(f"그리드 초기화 실패: {e}")
            return False

    def place_grid_orders(self):
        """✅ 개선: 안전한 잔고 분배 + 최소금액 보장"""
        try:
            current_price = self.get_current_price()
            if not current_price:
                return

            # 거래 중단 체크 추가
            if self.stop_trading:
                return

            self.indicators.update()
            pause_buy = self.indicators.should_pause_buy()

            # 🔧 변동성 임시 중단 체크 추가
            if hasattr(self, 'pause_buy_until') and self.pause_buy_until:
                if datetime.now() < self.pause_buy_until:
                    logger.debug(f"[{self.coin}] 매수 중단 중 (고변동성)")
                    pause_buy = True
                else:
                    self.pause_buy_until = None

            # 잔고 확인
            balances = self.upbit.get_balances()
            available_krw = 0

            for balance in balances:
                if balance['currency'] == 'KRW':
                    available_krw = float(balance['balance'])
                    break

            # 🔧 안전 마진: 수수료 + 슬리피지 고려 (2%)
            safe_krw = available_krw * 0.98

            # 매수 필요한 그리드 찾기
            buy_needed_grids = []
            for i in range(len(self.grids) - 1):
                grid = self.grids[i]
                if (not pause_buy and
                        current_price > grid['price'] and
                        not grid['has_position'] and
                        not grid['buy_order_uuid']):
                    buy_needed_grids.append(i)

            if not buy_needed_grids:
                logger.debug(f"[{self.coin}] 매수 필요 그리드 없음")
            else:
                # 🔧 우선순위: 현재가에서 먼 그리드부터 (저가 매수 우선)
                buy_needed_grids.sort()

                # 🔧 필터링: 실제 주문 가능한 그리드만
                affordable_grids = []
                for i in buy_needed_grids:
                    grid = self.grids[i]
                    if safe_krw >= self.min_order_amount:
                        affordable_grids.append(i)

                if not affordable_grids:
                    logger.warning(f"[{self.coin}] 잔고 부족 (필요: {self.min_order_amount:,.0f}원, 보유: {safe_krw:,.0f}원)")
                    return

                # 🔧 균등 분배
                amount_per_grid = safe_krw / len(affordable_grids)

                # 🔧 최소금액 미달 시 개수 줄이기
                if amount_per_grid < self.min_order_amount:
                    max_grids = int(safe_krw / self.min_order_amount)
                    if max_grids == 0:
                        logger.warning(f"[{self.coin}] 잔고 부족 (최소 1개 그리드 주문 불가)")
                        return
                    affordable_grids = affordable_grids[:max_grids]
                    amount_per_grid = safe_krw / len(affordable_grids)

                logger.info(f"[{self.coin}] 매수 대상: {len(affordable_grids)}개 | 그리드당: {amount_per_grid:,.0f}원")

                for i in affordable_grids:
                    grid = self.grids[i]

                    # 🔧 실제 사용 가능 금액 재확인
                    if safe_krw < self.min_order_amount:
                        logger.debug(f"[{self.coin}] Grid{i} 잔고 소진")
                        break

                    # 실제 투입 금액 (남은 금액과 계획 금액 중 작은 값)
                    actual_amount = min(amount_per_grid, safe_krw)

                    volume = actual_amount / grid['price']
                    volume = float(Decimal(str(volume)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))

                    # 🔧 최종 금액 검증
                    order_value = volume * grid['price']
                    if order_value < self.min_order_amount:
                        logger.warning(f"[{self.coin}] Grid{i} 최소금액 미달: {order_value:.0f}원")
                        continue

                    result = execute_buy_limit_order(self.upbit, self.market, grid['price'], volume)
                    if result and 'uuid' in result:
                        grid['buy_order_uuid'] = result['uuid']
                        self.pending_orders[result['uuid']] = {
                            'grid_index': i,
                            'side': 'buy',
                            'created_at': datetime.now(),
                            'expected_volume': volume,
                            'filled_volume': 0.0
                        }
                        self._save_pending_orders()
                        safe_krw -= order_value
                        logger.info(f"[{self.coin}] 📥 매수: Grid{i} {grid['price']:,}원 | {volume:.6f} | {order_value:,.0f}원")
                    time.sleep(API_CALL_INTERVAL)

            # 매도 주문 배치
            for i in range(len(self.grids) - 1):
                grid = self.grids[i]
                next_grid = self.grids[i + 1]

                if grid['has_position'] and grid['volume'] > 0 and not grid['sell_order_uuid']:
                    sell_price = next_grid['price']
                    sell_volume = grid['volume']

                    if sell_volume * sell_price >= self.min_order_amount:
                        result = execute_sell_limit_order(self.upbit, self.market, sell_price, sell_volume)
                        if result and 'uuid' in result:
                            grid['sell_order_uuid'] = result['uuid']
                            self.pending_orders[result['uuid']] = {
                                'grid_index': i,
                                'side': 'sell',
                                'created_at': datetime.now(),
                                'expected_volume': sell_volume,
                                'filled_volume': 0.0,
                                'price': sell_price
                            }
                            self._save_pending_orders()
                            logger.info(f"[{self.coin}] 📤 매도: Grid{i}→{i+1} {sell_price:,}원")
                        time.sleep(API_CALL_INTERVAL)

            # 최상위 그리드 처리
            last_grid = self.grids[-1]
            if last_grid['has_position'] and last_grid['volume'] > 0 and not last_grid.get('sell_order_uuid'):
                sell_price = round_price(last_grid['price'] * 1.01, self.coin)
                sell_volume = last_grid['volume']

                if sell_volume * sell_price >= self.min_order_amount:
                    result = execute_sell_limit_order(self.upbit, self.market, sell_price, sell_volume)
                    if result and 'uuid' in result:
                        last_grid['sell_order_uuid'] = result['uuid']
                        self.pending_orders[result['uuid']] = {
                            'grid_index': len(self.grids) - 1,
                            'side': 'sell',
                            'created_at': datetime.now(),
                            'expected_volume': sell_volume,
                            'filled_volume': 0.0,
                            'price': sell_price
                        }
                        self._save_pending_orders()
                        logger.info(f"[{self.coin}] 📤 최상위 매도: {sell_price:,}원")

        except Exception as e:
            logger.error(f"[{self.coin}] 주문 배치 실패: {e}")

    def check_filled_orders(self):
        """✅ 간소화: 체결 확인만, 재주문은 place_grid_orders()에서"""
        try:
            for uuid in list(self.pending_orders.keys()):
                info = self.pending_orders.get(uuid, {})
                grid_index = info.get('grid_index')
                side = info.get('side')

                # 🔧 유효성 먼저 체크
                if grid_index is None or grid_index >= len(self.grids):
                    self.pending_orders.pop(uuid, None)
                    continue
                
                grid = self.grids[grid_index]  # ✅ 여기로 이동

                # 🔧 주문 TTL 체크
                created_at = info.get('created_at')
                if created_at and isinstance(created_at, datetime):
                    age_hours = (datetime.now() - created_at).total_seconds() / 3600
                    if age_hours > ORDER_TTL_HOURS:
                        logger.warning(f"[{self.coin}] 주문 TTL 초과: {uuid[:8]}... ({age_hours:.1f}시간)")
                        try:
                            self.upbit.cancel_order(uuid)
                            self.pending_orders.pop(uuid, None)
                            self._save_pending_orders()
                            
                            if side == 'buy':
                                grid['buy_order_uuid'] = None
                            elif side == 'sell':
                                grid['sell_order_uuid'] = None
                            continue
                        except Exception as e:
                            logger.warning(f"TTL 취소 실패: {e}")
                
                # 주문 조회
                order_detail = None
                try:
                    order_detail = self.upbit.get_order(uuid)
                except Exception as e:
                    logger.debug(f"[{self.coin}] 주문 조회 실패 ({uuid[:8]}...): {e}")
                    try:
                        open_orders = self.upbit.get_order(self.market, state='wait') or []
                        order_detail = next((o for o in open_orders if o.get('uuid') == uuid), None)
                    except Exception as e2:
                        logger.warning(f"[{self.coin}] 대체 조회 실패: {e2}")
                        continue
                
                if order_detail is None:
                    continue
                
                state = order_detail.get('state')
                executed = float(order_detail.get('executed_volume') or 0)
                remaining = float(order_detail.get('remaining_volume') or 0)
                filled_before = float(info.get('filled_volume') or 0)
                new_filled = max(0.0, executed - filled_before)
                
                # 취소/만료 체크
                if state in ['cancel', 'done'] and executed == 0:
                    logger.info(f"[{self.coin}] 주문 취소/만료: {side} Grid{grid_index}")
                    self.pending_orders.pop(uuid, None)
                    self._save_pending_orders()
                    
                    if side == 'buy':
                        grid['buy_order_uuid'] = None
                    elif side == 'sell':
                        grid['sell_order_uuid'] = None
                    continue
                
                # 체결 처리
                if new_filled > 0:
                    if side == 'buy':
                        # 매수 체결
                        grid['has_position'] = True
                        grid['volume'] += new_filled
                        grid['buy_price'] = grid['price']
                        logger.info(f"[{self.coin}] ✅ 매수 체결: Grid{grid_index} {grid['price']:,}원 | +{new_filled:.6f} | 총: {grid['volume']:.6f}")
                        
                        if remaining <= 0:
                            # 완전 체결
                            self.pending_orders.pop(uuid, None)
                            self._save_pending_orders()
                            grid['buy_order_uuid'] = None
                        else:
                            # 부분 체결
                            info['filled_volume'] = executed
                            info['expected_volume'] = remaining
                            self.pending_orders[uuid] = info
                            self._save_pending_orders()
                    
                    elif side == 'sell':
                        # 매도 체결
                        sell_price = float(info.get('price') or order_detail.get('price') or 0)
                        grid['volume'] = max(0.0, grid['volume'] - new_filled)
                        logger.info(f"[{self.coin}] ✅ 매도 체결: Grid{grid_index} {sell_price:,}원 | -{new_filled:.6f} | 남음: {grid['volume']:.6f}")
                        
                        if remaining <= 0:
                            # 완전 체결
                            self.pending_orders.pop(uuid, None)
                            self._save_pending_orders()
                            grid['sell_order_uuid'] = None
                            
                            # 수익 계산 (매수/매도 수수료 모두 반영)
                            buy_price = grid.get('buy_price', 0)
                            if buy_price > 0:
                                # 🔧 매수 원가 = 매수가 + 매수 수수료
                                actual_buy_cost = buy_price * (1 + UPBIT_FEE_RATE)
                                # 🔧 매도 수익 = 매도가 - 매도 수수료
                                actual_sell_revenue = sell_price * (1 - UPBIT_FEE_RATE)
                                # 🔧 순수익 = (매도 수익 - 매수 원가) * 수량
                                net_profit = (actual_sell_revenue - actual_buy_cost) * new_filled
                                
                                self.total_profit += net_profit
                                self.trade_count += 1
                                profit_rate = (net_profit / (actual_buy_cost * new_filled)) * 100
                                
                                logger.info(f"[{self.coin}] 💰 수익: +{net_profit:,.0f}원 ({profit_rate:.2f}%) | 누적: {self.total_profit:,.0f}원")
                                send_discord_notification(
                                    f"**{self.coin} 매도 완료! 💰**\n"
                                    f"• Grid {grid_index}→{grid_index+1}\n"
                                    f"• 매수: {buy_price:,.0f}원\n"
                                    f"• 매도: {sell_price:,.0f}원\n"
                                    f"• 수익: +{net_profit:,.0f}원 ({profit_rate:.2f}%)\n"
                                    f"• 누적: {self.total_profit:,.0f}원",
                                    color=3066993
                                )
                            
                            # 포지션 초기화
                            grid['has_position'] = False
                            grid['volume'] = 0.0
                            grid['buy_price'] = 0.0
                        else:
                            # 부분 체결
                            info['filled_volume'] = executed
                            info['expected_volume'] = remaining
                            self.pending_orders[uuid] = info
                            self._save_pending_orders()
        
        except Exception as e:
            logger.error(f"[{self.coin}] 체결 확인 실패: {e}")

    def check_volatility(self):
        """✅ 변동성 체크 - 급등락 시 그리드 범위 확대"""
        try:
            df = pyupbit.get_ohlcv(self.market, interval="minute5", count=12)  # 1시간
            if df is None or len(df) < 12:
                return
            
            high = df['high'].max()
            low = df['low'].min()
            current = df['close'].iloc[-1]
            volatility = (high - low) / low
            
            # 🔧 1시간 내 10% 이상 변동 시
            if volatility > 0.10:
                logger.warning(f"[{self.coin}] ⚠️ 고변동성 감지: {volatility*100:.1f}%")
                
                # 🔧 추가 안전장치: 급락 시 매수 중단
                price_change = (current - low) / low
                if price_change < 0.05:  # 저점 근처 (5% 이내)
                    logger.warning(f"[{self.coin}] ⚠️ 급락 후 저점 근처 - 매수 일시 중단")
                    # 임시 매수 중단 플래그 (다음 사이클에서 재평가)
                    self.pause_buy_until = datetime.now() + timedelta(minutes=30)
                
                send_discord_notification(
                    f"⚠️ **{self.coin} 고변동성 경고**\n"
                    f"• 1시간 변동폭: {volatility*100:.1f}%\n"
                    f"• 고점: {high:,.0f}원\n"
                    f"• 저점: {low:,.0f}원\n"
                    f"• 현재: {current:,.0f}원",
                    color=16776960  # 노란색
                )
                
        except Exception as e:
            logger.error(f"[{self.coin}] 변동성 체크 실패: {e}")

    def check_stop_loss_take_profit(self):
        """✅ 개선: 수수료 고려 + 안전 장치"""
        try:
            # 🔧 이미 중단 상태면 리턴
            if self.stop_trading:
                return
            
            current_price = self.get_current_price()
            if not current_price:
                return
            
            balances = self.upbit.get_balances()
            krw = 0
            coin = 0
            for b in balances:
                if b['currency'] == 'KRW':
                    krw = float(b['balance'])
                elif b['currency'] == self.coin:
                    coin = float(b['balance'])
            
            # 🔧 평가 금액에서 수수료 차감 (매도 시 0.05%)
            coin_value = coin * current_price * (1 - UPBIT_FEE_RATE)
            total_value = krw + coin_value
            profit_rate = (total_value / self.initial_capital - 1)
            
            # 🔧 AI 추천 파라미터 우선 사용
            stop_loss = self.stop_loss_threshold
            take_profit = self.take_profit_threshold
            
            if profit_rate <= stop_loss:
                logger.warning(f"[{self.coin}] 🛑 손절 발동: {profit_rate*100:.2f}%")
                self.emergency_close_all()
                self.stop_trading = True
                send_discord_notification(
                    f"🛑 **{self.coin} 손절 실행**\n"
                    f"• 손실률: {profit_rate*100:.2f}%\n"
                    f"• 초기 자본: {self.initial_capital:,.0f}원\n"
                    f"• 현재 자산: {total_value:,.0f}원\n"
                    f"• 손실액: {(total_value - self.initial_capital):,.0f}원",
                    color=15158332
                )
            
            elif profit_rate >= take_profit:
                logger.info(f"[{self.coin}] 🎉 익절 발동: {profit_rate*100:.2f}%")
                self.emergency_close_all()
                self.stop_trading = True
                send_discord_notification(
                    f"🎉 **{self.coin} 익절 완료!**\n"
                    f"• 수익률: {profit_rate*100:.2f}%\n"
                    f"• 초기 자본: {self.initial_capital:,.0f}원\n"
                    f"• 현재 자산: {total_value:,.0f}원\n"
                    f"• 수익액: {(total_value - self.initial_capital):,.0f}원",
                    color=5763719
                )
        except Exception as e:
            logger.error(f"[{self.coin}] 손절/익절 체크 실패: {e}")
    
    def run_cycle(self):
        """✅ 모든 체크 포함 + 순서 최적화"""
        try:
            # 🔧 거래 중단 상태 최우선 체크
            if self.stop_trading:
                logger.debug(f"[{self.coin}] 거래 중단 상태")
                return
            
            # 🔧 임시 매수 중단 체크
            if hasattr(self, 'pause_buy_until') and self.pause_buy_until:
                if datetime.now() < self.pause_buy_until:
                    logger.debug(f"[{self.coin}] 매수 일시 중단 중 (고변동성)")
                else:
                    self.pause_buy_until = None
            
            # 1. 손절/익절 체크 (최우선)
            self.check_stop_loss_take_profit()
            if self.stop_trading:
                return
            
            # 2. 변동성 체크
            self.check_volatility()
            
            # 3. 그리드 리밸런싱 체크
            self.check_grid_rebalancing()
            if self.stop_trading:
                return
            
            # 4. 체결 확인 (먼저)
            self.check_filled_orders()
            time.sleep(API_CALL_INTERVAL)
            
            # 5. 신규 주문 배치
            self.place_grid_orders()
            
        except Exception as e:
            logger.error(f"[{self.coin}] run_cycle 오류: {e}")

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
                bot = GridTradingBot(self.upbit, coin, capital_for_coin)
                if bot.initialize_grids():
                    self.bots.append(bot)
                    time.sleep(API_CALL_INTERVAL * 2)
            
            logger.info(f"총 {len(self.bots)}개 봇 초기화 완료")
            send_discord_notification(
                f"🚀 **그리드 봇 시작!**\n"
                f"• 코인: {', '.join([bot.coin for bot in self.bots])}\n"
                f"• 투자금: {INITIAL_CAPITAL:,}원",
                color=3447003
            )
            return len(self.bots) > 0
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            return False
    
    def run(self):
        logger.info("="*60)
        logger.info("✅ 순수 그리드 매매 봇 시작")
        logger.info("="*60)
        last_report_time = datetime.now()
        
        try:
            while True:
                # 🔧 전역 변수 접근 수정
                global PAUSE_TRADING_ON_NETWORK, NETWORK_FAILURES
                
                if PAUSE_TRADING_ON_NETWORK:
                    logger.warning("⏸️  네트워크 이슈로 트레이딩 일시중지...")
                    try:
                        r = requests.get('https://api.upbit.com/v1/markets', timeout=5)
                        if r.status_code == 200:
                            NETWORK_FAILURES = 0
                            PAUSE_TRADING_ON_NETWORK = False
                            logger.info("✅ 네트워크 복구: 트레이딩 재개")
                    except Exception as e:
                        logger.warning(f"복구 체크 오류: {e}")
                    time.sleep(NETWORK_PAUSE_SLEEP_SECS)
                    continue

                for bot in self.bots:
                    try:
                        # 🔧 봇별 중단 상태 체크
                        if not bot.stop_trading:
                            bot.run_cycle()
                        else:
                            logger.debug(f"[{bot.coin}] 거래 중단 상태 (손절/익절 발동)")
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
                report += (f"**{stats['coin']}** {stats['trend']}\n"
                          f"• 수익: {stats['total_profit']:,.0f}원 ({stats['profit_rate']:.2f}%)\n"
                          f"• 거래: {stats['trade_count']}회\n"
                          f"• RSI: {stats['rsi']:.1f}\n\n")
            
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
                report += (f"**{stats['coin']}**\n"
                          f"• 수익: {stats['total_profit']:,.0f}원 ({stats['profit_rate']:.2f}%)\n"
                          f"• 거래: {stats['trade_count']}회\n\n")
            
            report += f"**💰 최종 성과**\n• 순수익: {total_profit:,.0f}원\n• 총 거래: {total_trades}회"
            send_discord_notification(report, color=15844367)
        except Exception as e:
            logger.error(f"최종 리포트 실패: {e}")

if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("🎯 순수 그리드 트레이딩 봇")
        logger.info("="*60)
        
        manager = GridBotManager()
        if manager.initialize():
            manager.run()
        else:
            logger.error("❌ 봇 초기화 실패!")
    except Exception as e:
        logger.error(f"메인 오류: {e}")