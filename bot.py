import os
import time
import json
import logging
import requests
import pyupbit
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import openai
import ast

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('grid_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 로그 포맷 헬퍼 -------------------------------------------------
def format_order_short(result: Dict) -> str:
    try:
        price = int(result.get('price')) if result.get('price') else None
        volume = float(result.get('volume')) if result.get('volume') else None
        uuid = result.get('uuid')
        locked = result.get('locked') or result.get('reserved_fee') or 0
        return f"uuid={uuid} | price={price:,}원 | vol={volume:.6f} | locked={float(locked):,.2f}원"
    except Exception:
        return str(result)

# -------------------------------------------------------------------

# 설정
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

INITIAL_CAPITAL = None
MIN_ORDER_AMOUNT = 5000

COINS = ['XRP', 'BTC', 'ETH']  # 멀티코인 지원
COIN_CONFIGS = {
    'XRP': {
        'range': 0.04,  # 4% (좁게)
        'grids': 4,     # 4개 (27만원 테스트용으로 줄임)
        'portion': 0.5,  # 50%
        'min_order': 5000  # Upbit 최소 주문 5,000원으로 설정
    },
    'BTC': {
        'range': 0.03,  # 3%
        'grids': 2,     # 2개 (27만원 테스트용으로 줄임)
        'portion': 0.3,  # 30%
        'min_order': 5000
    },
    'ETH': {
        'range': 0.03,  # 3%
        'grids': 2,     # 2개 (27만원 테스트용으로 줄임)
        'portion': 0.2,  # 20%
        'min_order': 5000
    }
}
API_CALL_INTERVAL = 0.2
REINVEST_THRESHOLD = 30000
# 주문 TTL(시간) - 이 시간이 지나면 미체결 주문을 자동 취소하고 재배치할 수 있습니다.
ORDER_TTL_HOURS = int(os.getenv('ORDER_TTL_HOURS', '12'))  # 기본 12시간
# 미체결 주문의 잔여를 시장가로 채울지 여부 (리스크 있음)
FILL_REMAINING_MARKET = os.getenv('FILL_REMAINING_MARKET', 'false').lower() in ('1', 'true', 'yes')
# 시장가로 잔여를 채울 때 허용하는 최대 잔여 비율(원래 주문량 대비). 예: 0.5 = 잔여가 50% 이하일 때만 시장가로 채움
MARKET_FILL_IF_LESS_THAN_RATIO = float(os.getenv('MARKET_FILL_IF_LESS_THAN_RATIO', '0.5'))
# 옵션: 초기화시(또는 주기적으로) 모든 그리드에 지정매수 주문을 미리 걸 것인지 여부
# True면 그리드 개수만큼 모든 매수 주문을 걸어둡니다. False면 기존 동작(현재가 > 그리드 가격인 경우에만).
PREPLACE_ALL_GRIDS = False

def log_trade_event(event_type: str, details: Dict):
    try:
        log_entry = {'timestamp': datetime.utcnow().isoformat(), 'event_type': event_type, **details}
        with open('trade_log.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"거래 로그 기록 실패: {e}")

def send_discord_notification(message: str, color: int = 3447003):
    try:
        # 최근 로그를 파일에서 읽어 요약으로 포함
        def get_recent_logs(path: str, lines: int = 20, max_chars: int = 1000) -> str:
            try:
                if not os.path.exists(path):
                    return ''
                with open(path, 'r', encoding='utf-8') as f:
                    all_lines = f.read().splitlines()
                recent = all_lines[-lines:]
                # 전처리: 불필요한 길이 긴 dict 응답을 요약 형식으로 변환
                pretty_lines = []
                for ln in recent:
                    ln_strip = ln.strip()
                    # 이미 정리된 ORDER 로그는 그대로 사용
                    if ln_strip.startswith('[ORDER]'):
                        pretty_lines.append(ln_strip)
                        continue

                    # Upbit 응답이 로깅된 경우 (예: BUY_LIMIT_ORDER_RESPONSE - KRW-XRP: {...})
                    if 'BUY_LIMIT_ORDER_RESPONSE -' in ln_strip or 'BUY_LIMIT_ORDER_RESPONSE -' in ln_strip:
                        try:
                            # 딕셔너리 문자열 추출
                            parts = ln_strip.split(':', 1)
                            if len(parts) > 1:
                                dict_str = parts[1].strip()
                                # 안전하게 파싱
                                parsed = ast.literal_eval(dict_str)
                                pretty_lines.append(format_order_short(parsed))
                                continue
                        except Exception:
                            pass

                    # 기타 BUY/SELL raw response 패턴
                    if 'BUY_LIMIT_ORDER_RESPONSE' in ln_strip or 'SELL_LIMIT_ORDER_RESPONSE' in ln_strip:
                        try:
                            dict_start = ln_strip.find('{')
                            if dict_start != -1:
                                parsed = ast.literal_eval(ln_strip[dict_start:])
                                pretty_lines.append(format_order_short(parsed))
                                continue
                        except Exception:
                            pass

                    # 기본으로는 원문 유지(짧게)
                    pretty_lines.append(ln_strip)

                text = '\n'.join(pretty_lines)
                if len(text) > max_chars:
                    return text[-max_chars:]
                return text
            except Exception:
                return ''

        recent_logs = get_recent_logs('grid_bot.log', lines=20, max_chars=1500)
        if recent_logs:
            # embed description에 길이 제한이 있으므로 요약 형식으로 추가
            description = f"{message}\n\n최근 로그(마지막 {min(20, len(recent_logs.splitlines()))}줄):\n```text\n{recent_logs}\n```"
        else:
            description = message

        data = {"embeds": [{"title": "🤖 그리드 트레이딩 봇", "description": description, "color": color, "timestamp": datetime.utcnow().isoformat()}]}
        # 웹훅이 설정되어 있는지 확인
        if not DISCORD_WEBHOOK_URL:
            logger.warning("DISCORD_WEBHOOK_URL 미설정: 디스코드 알림을 건너뜁니다.")
            return
        requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=5)
    except Exception as e:
        logger.error(f"디스코드 알림 실패: {e}")

def execute_buy_limit_order(upbit, ticker, price, volume):
    try:
        logger.info(f"BUY_LIMIT_ORDER_ATTEMPT - {ticker}: 가격 {price:,.0f}원, 수량 {volume:.8f}")
        
        # 업비트 최소 주문 금액 적용 (기본 5,000원)
        min_order_value = MIN_ORDER_AMOUNT if MIN_ORDER_AMOUNT else 5000
        order_value = price * volume
        if order_value < min_order_value:
            logger.warning(f"BUY_LIMIT_REJECTED - {ticker}: 주문 금액이 너무 작음 ({order_value:.0f}원 < {min_order_value}원)")
            return None
            
        result = upbit.buy_limit_order(ticker, price, volume)
        logger.info(f"BUY_LIMIT_ORDER_RESPONSE - {ticker}: {result}")
        
        if result and 'uuid' in result:
            # 깔끔한 요약 로그 출력
            try:
                locked = result.get('locked') or result.get('reserved_fee') or 0
                logger.info(f"[ORDER][BUY] {ticker} | price={int(result.get('price')):,}원 | volume={float(result.get('volume')):.6f} | locked={float(locked):,.2f}원 | uuid={result.get('uuid')}")
            except Exception:
                logger.info(f"BUY_LIMIT_SUCCESS - {ticker}: {price:,.0f}원 x {volume:.6f}")
            return result
        elif result and 'error' in result:
            logger.error(f"BUY_LIMIT_API_ERROR - {ticker}: {result['error']}")
            return None
        else:
            logger.warning(f"BUY_LIMIT_INVALID_RESPONSE - {ticker}: {result}")
            return None
            
    except Exception as e:
        logger.error(f"BUY_LIMIT_EXCEPTION - {ticker}: {str(e)}")
        return None

def cancel_order_safe(upbit, uuid):
    try:
        return upbit.cancel_order(uuid)
    except Exception as e:
        logger.warning(f"cancel_order_safe 실패: {e}")
        return None

def cancel_stale_orders(self):
    """미체결 주문 TTL을 검사하고 오래된 주문을 취소합니다."""
    try:
        now = datetime.now()
        to_cancel = []
        for uuid, meta in list(self.pending_orders.items()):
            created = meta.get('created_at')
            if not created:
                continue
            if now - created > timedelta(hours=ORDER_TTL_HOURS):
                to_cancel.append((uuid, meta))

        for uuid, meta in to_cancel:
            logger.info(f"[{self.coin}] 오래된 주문 취소: uuid={uuid}, age={(now - meta.get('created_at')).total_seconds()/3600:.1f}h")
            cancel_order_safe(self.upbit, uuid)
            # 주문 취소 후 pending 제거 및 grid 상태 초기화
            grid_index = meta.get('grid_index')
            side = meta.get('side')
            if grid_index is not None and grid_index < len(self.grids):
                grid = self.grids[grid_index]
                if side == 'buy' and grid.get('buy_order_uuid') == uuid:
                    grid['buy_order_uuid'] = None
                if side == 'sell' and grid.get('sell_order_uuid') == uuid:
                    grid['sell_order_uuid'] = None

            try:
                del self.pending_orders[uuid]
            except KeyError:
                pass
            time.sleep(API_CALL_INTERVAL)
    except Exception as e:
        logger.error(f"[{self.coin}] cancel_stale_orders 실패: {e}")

def execute_sell_limit_order(upbit, ticker, price, volume):
    try:
        result = upbit.sell_limit_order(ticker, price, volume)
        if result and 'uuid' in result:
            try:
                locked = result.get('locked') or result.get('reserved_fee') or 0
                logger.info(f"[ORDER][SELL] {ticker} | price={int(result.get('price')):,}원 | volume={float(result.get('volume')):.6f} | locked={float(locked):,.2f}원 | uuid={result.get('uuid')}")
            except Exception:
                logger.info(f"SELL_LIMIT - {ticker}: {price:,.0f}원 x {volume:.6f}")
            return result
        return None
    except Exception as e:
        logger.error(f"SELL_LIMIT_ERROR - {ticker}: {e}")
        return None

def execute_sell_market_order(upbit, ticker, volume):
    try:
        result = upbit.sell_market_order(ticker, volume)
        if result and 'uuid' in result:
            logger.info(f"[ORDER][SELL_MARKET] {ticker} | volume={float(result.get('volume')):.6f} | uuid={result.get('uuid')}")
            return result
        return None
    except Exception as e:
        logger.error(f"SELL_MARKET_ERROR - {ticker}: {e}")
        return None

def is_peak_time():
    hour = datetime.now().hour
    return (9 <= hour <= 11) or (21 <= hour <= 23)

def round_price(price: float, coin: str) -> float:
    """업비트 가격 단위에 맞춰 가격을 반올림합니다.

    단위 기준(예상):
    - price < 10,000: 1
    - 10,000 <= price < 100,000: 5
    - 100,000 <= price < 500,000: 10
    - 500,000 <= price < 1,000,000: 50
    - 1,000,000 <= price < 2,000,000: 100
    - 2,000,000 <= price < 5,000,000: 500
    - price >= 5,000,000: 1000
    """
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

    # 반올림하여 tick 단위로 맞춤
    return int(round(p / tick) * tick)

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
        self.price_unit = config.get('price_unit', 1)
        
        max_possible_grids = int(self.capital // self.min_order_amount)
        default_grids = config.get('grids', 10)
        self.grid_count = min(default_grids, max_possible_grids)
        self.max_grids = max_possible_grids
        
        self.grids: List[Dict] = []
        self.upper_price = 0.0
        self.lower_price = 0.0
        self.total_profit = 0.0
        self.trade_count = 0
        self.start_time = datetime.now()
        self.last_reinvest_time = datetime.now()
        self.pending_orders = {}
        self.volatility = 0.0
        self.last_volatility_check = datetime.now()
        self.high_profit = 0.0
        self.trailing_sell_executed = False
        
        logger.info(f"[{self.coin}] 봇 초기화 (자본: {capital_per_coin:,.0f}원)")
    
    def get_current_price(self) -> Optional[float]:
        try:
            price = pyupbit.get_current_price(self.market)
            return float(price) if price else None
        except Exception as e:
            logger.error(f"[{self.coin}] 현재가 조회 실패: {e}")
            return None
    
    def check_volatility(self) -> float:
        try:
            if datetime.now() - self.last_volatility_check < timedelta(hours=1):
                return self.volatility
            
            df = pyupbit.get_ohlcv(self.market, interval="day", count=7)
            if df is not None and len(df) > 0:
                volatility = ((df['high'] - df['low']) / df['close'] * 100).mean()
                self.volatility = volatility
                self.last_volatility_check = datetime.now()
                logger.info(f"[{self.coin}] 변동성: {volatility:.2f}%")
                return volatility
        except Exception as e:
            logger.error(f"[{self.coin}] 변동성 체크 실패: {e}")
        return self.volatility
    
    def get_market_analysis(self) -> Dict:
        """AI를 사용한 시장 분석"""
        try:
            if not OPENAI_API_KEY:
                logger.warning(f"[{self.coin}] OpenAI API 키가 없어 기본 분석 사용")
                return {}
            
            # 최근 30일 데이터 수집
            df = pyupbit.get_ohlcv(self.market, interval="day", count=30)
            if df is None or len(df) < 7:
                return {}
            
            # 기술적 지표 계산
            current_price = df['close'].iloc[-1]
            price_change_7d = ((current_price - df['close'].iloc[-7]) / df['close'].iloc[-7]) * 100
            price_change_30d = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            volatility = ((df['high'] - df['low']) / df['close'] * 100).mean()
            volume_avg = df['volume'].mean()
            volume_current = df['volume'].iloc[-1]
            volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1
            
            # RSI 계산 (간단 버전)
            def calculate_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return 50
                gains = []
                losses = []
                for i in range(1, len(prices)):
                    change = prices[i] - prices[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                
                avg_gain = sum(gains[-period:]) / period
                avg_loss = sum(losses[-period:]) / period
                if avg_loss == 0:
                    return 100
                rs = avg_gain / avg_loss
                return 100 - (100 / (1 + rs))
            
            rsi = calculate_rsi(df['close'].values)
            
            market_data = {
                'current_price': current_price,
                'price_change_7d': price_change_7d,
                'price_change_30d': price_change_30d,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'trend': 'bullish' if price_change_7d > 0 else 'bearish',
                'momentum': 'strong' if abs(price_change_7d) > 5 else 'weak'
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"[{self.coin}] 시장 분석 실패: {e}")
            return {}
    
    def ai_optimize_grid_settings(self) -> Tuple[float, int]:
        """AI를 사용한 그리드 범위와 갯수 최적화"""
        try:
            if not OPENAI_API_KEY:
                # AI가 없으면 기존 로직 사용
                adaptive_range = self.adaptive_grid_range()
                return adaptive_range, self.grid_count
            
            market_data = self.get_market_analysis()
            if not market_data:
                adaptive_range = self.adaptive_grid_range()
                return adaptive_range, self.grid_count
            
            # AI 프롬프트 생성
            prompt = f"""
다음은 {self.coin} 코인의 현재 시장 데이터입니다:

- 현재 가격: {market_data['current_price']:,.0f}원
- 7일 가격 변동: {market_data['price_change_7d']:.2f}%
- 30일 가격 변동: {market_data['price_change_30d']:.2f}%
- 변동성: {market_data['volatility']:.2f}%
- 거래량 비율: {market_data['volume_ratio']:.2f}
- RSI: {market_data['rsi']:.1f}
- 추세: {market_data['trend']}
- 모멘텀: {market_data['momentum']}

이 데이터를 기반으로 그리드 트레이딩을 위한 최적의 설정을 추천해주세요:

1. 그리드 범위 (가격 변동 범위 %): 현재 가격을 기준으로 상하 몇 % 범위에서 그리드를 설정할지
2. 그리드 갯수: 해당 범위 내에서 몇 개의 그리드를 설정할지

고려사항:
- 변동성이 높으면 범위를 넓게, 갯수를 적게
- 변동성이 낮으면 범위를 좁게, 갯수를 많게  
- 상승 추세에서는 매수 중심으로, 하락 추세에서는 매도 중심으로
- RSI가 70 이상이면 과매수, 30 이하면 과매도로 고려
- 모멘텀이 강하면 더 적극적인 설정

응답 형식:
범위: X.X%
갯수: X
설명: 간단한 추천 이유
"""
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content.strip()
            logger.info(f"[{self.coin}] AI 그리드 최적화 응답: {ai_response}")
            
            # 응답 파싱
            lines = ai_response.split('\n')
            range_percent = None
            grid_count = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('범위:') and '%' in line:
                    try:
                        range_str = line.split(':')[1].strip().replace('%', '')
                        range_percent = float(range_str) / 100
                    except:
                        pass
                elif line.startswith('갯수:'):
                    try:
                        count_str = line.split(':')[1].strip()
                        grid_count = int(count_str)
                    except:
                        pass
            
            # 파싱 실패 시 기본값 사용
            if range_percent is None or range_percent <= 0:
                range_percent = self.adaptive_grid_range()
                logger.warning(f"[{self.coin}] AI 범위 파싱 실패, 기본값 사용: {range_percent:.3f}")
            
            if grid_count is None or grid_count <= 0:
                grid_count = self.grid_count
                logger.warning(f"[{self.coin}] AI 갯수 파싱 실패, 기본값 사용: {grid_count}")
            
            # 합리적인 범위로 제한
            range_percent = max(0.01, min(0.15, range_percent))  # 1% ~ 15%
            grid_count = max(2, min(10, grid_count))  # 2개 ~ 10개
            
            logger.info(f"[{self.coin}] AI 최적화 결과 - 범위: {range_percent:.3f}, 갯수: {grid_count}")
            send_discord_notification(f"🤖 **{self.coin} AI 그리드 최적화**\n• 범위: {range_percent:.1%}\n• 갯수: {grid_count}개\n• 변동성: {market_data.get('volatility', 0):.2f}%", color=3447003)
            
            return range_percent, grid_count
            
        except Exception as e:
            logger.error(f"[{self.coin}] AI 그리드 최적화 실패: {e}")
            # 실패 시 기존 로직 사용
            adaptive_range = self.adaptive_grid_range()
            return adaptive_range, self.grid_count
    
    def adaptive_grid_range(self) -> float:
        volatility = self.check_volatility()
        if volatility == 0:
            return self.price_range_percent
        
        if volatility < 3:
            return 0.03
        elif volatility < 5:
            return 0.05
        elif volatility < 10:
            return 0.07
        else:
            return 0.10
    
    def initialize_grids(self) -> bool:
        try:
            current_price = self.get_current_price()
            if not current_price:
                return False
            
            # AI 기반 그리드 설정 최적화
            ai_range, ai_grid_count = self.ai_optimize_grid_settings()
            
            # AI 결과를 사용하되, 설정된 최대값을 초과하지 않도록 제한
            self.grid_count = min(ai_grid_count, self.max_grids)
            
            self.upper_price = round_price(current_price * (1 + ai_range), self.coin)
            self.lower_price = round_price(current_price * (1 - ai_range), self.coin)
            
            step = (self.upper_price - self.lower_price) / self.grid_count
            self.grids = []
            
            for i in range(self.grid_count + 1):
                grid_price = self.lower_price + (step * i)
                price = round_price(grid_price, self.coin)
                self.grids.append({'price': price, 'has_position': False, 'buy_order_uuid': None, 'sell_order_uuid': None, 'volume': 0.0, 'buy_price': 0.0})
            
            logger.info(f"[{self.coin}] AI 최적화 그리드 설정: {self.lower_price:,.0f} ~ {self.upper_price:,.0f} ({self.grid_count}개)")
            send_discord_notification(f"🎯 **{self.coin_name} ({self.coin}) AI 그리드 설정 완료**\n• 범위: {self.lower_price:,.0f}원 ~ {self.upper_price:,.0f}원\n• 그리드: {self.grid_count}개\n• AI 최적화 적용됨", color=5763719)
            return True
        except Exception as e:
            logger.error(f"[{self.coin}] 그리드 초기화 실패: {e}")
            return False
    
    def place_grid_orders(self):
        try:
            current_price = self.get_current_price()
            if not current_price:
                return
            
            balances = self.upbit.get_balances()
            available_balance = 0
            for balance in balances:
                if balance['currency'] == 'KRW':
                    available_balance = float(balance['balance'])
                    break
            
            for i, grid in enumerate(self.grids[:-1]):
                buy_price = grid['price']
                sell_price = self.grids[i + 1]['price']

                # 그리드별 매수 주문 여부 결정
                if PREPLACE_ALL_GRIDS:
                    place_buy = (not grid['has_position'] and not grid['buy_order_uuid'])
                else:
                    place_buy = (current_price > buy_price and not grid['has_position'] and not grid['buy_order_uuid'])

                if not place_buy:
                    # 로깅: 왜 스킵했는지 간단히 기록
                    if grid['has_position']:
                        logger.debug(f"[{self.coin}] 그리드 {i} 스킵: 이미 포지션 보유")
                    elif grid['buy_order_uuid']:
                        logger.debug(f"[{self.coin}] 그리드 {i} 스킵: 이미 주문 존재 (uuid={grid['buy_order_uuid']})")
                    else:
                        logger.debug(f"[{self.coin}] 그리드 {i} 스킵: 가격 조건 미충족 (current={current_price:,.0f}, grid={buy_price:,.0f})")
                    continue

                amount_per_grid = self.capital / self.grid_count
                if amount_per_grid < self.min_order_amount:
                    logger.info(f"[{self.coin}] 그리드 {i} 매수 스킵: 그리드당 금액 {amount_per_grid:,.0f}원 < 최소주문 {self.min_order_amount:,.0f}원")
                    continue

                if available_balance < amount_per_grid:
                    logger.info(f"[{self.coin}] 그리드 {i} 매수 스킵: 잔고 부족 ({available_balance:,.0f}원 < 필요 {amount_per_grid:,.0f}원)")
                    continue

                volume = amount_per_grid / buy_price
                volume = float(Decimal(str(volume)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
                result = execute_buy_limit_order(self.upbit, self.market, buy_price, volume)
                if result and 'uuid' in result:
                    grid['buy_order_uuid'] = result['uuid']
                    # pending metadata: created time and expected volume
                    self.pending_orders[result['uuid']] = {'grid_index': i, 'side': 'buy', 'created_at': datetime.now(), 'expected_volume': volume}
                    available_balance -= amount_per_grid
                    logger.info(f"[{self.coin}] 그리드 {i} 지정매수 주문 배치: price={buy_price:,.0f}, amount={amount_per_grid:,.0f}원")
                else:
                    logger.warning(f"[{self.coin}] 그리드 {i} 지정매수 주문 실패: price={buy_price:,.0f}, volume={volume:.8f}, result={result}")
                time.sleep(API_CALL_INTERVAL)
                
                if grid['has_position'] and grid['volume'] > 0 and not grid['sell_order_uuid']:
                    result = execute_sell_limit_order(self.upbit, self.market, sell_price, grid['volume'])
                    if result and 'uuid' in result:
                        grid['sell_order_uuid'] = result['uuid']
                        self.pending_orders[result['uuid']] = {'grid_index': i, 'side': 'sell', 'created_at': datetime.now(), 'expected_volume': grid['volume']}
                    time.sleep(API_CALL_INTERVAL)
        except Exception as e:
            logger.error(f"[{self.coin}] 주문 배치 실패: {e}")
    
    def check_filled_orders(self):
        try:
            # 오래된 주문 자동 취소 먼저 실행
            cancel_stale_orders(self)

            open_orders = self.upbit.get_order(self.market, state='wait')
            if open_orders is None:
                open_orders = []

            open_order_uuids = {order['uuid'] for order in open_orders}
            filled_uuids = set(self.pending_orders.keys()) - open_order_uuids

            for uuid in list(filled_uuids):
                order_info = self.pending_orders.get(uuid, {})
                grid_index = order_info.get('grid_index')
                side = order_info.get('side')
                expected_volume = float(order_info.get('expected_volume') or 0)

                if grid_index is None or grid_index >= len(self.grids):
                    try:
                        del self.pending_orders[uuid]
                    except KeyError:
                        pass
                    continue

                grid = self.grids[grid_index]

                if side == 'buy' and grid.get('buy_order_uuid') == uuid:
                    # 부분체결 처리: 가능한 경우 주문 상세에서 executed_volume 조회
                    executed_volume = None
                    try:
                        # some pyupbit versions allow querying by uuid; wrap in try
                        detail = None
                        try:
                            detail = self.upbit.get_order(uuid)
                        except Exception:
                            detail = None

                        if detail and isinstance(detail, dict):
                            executed_volume = float(detail.get('executed_volume') or detail.get('executed_units') or 0)
                    except Exception:
                        executed_volume = None

                    # fallback to expected_volume if no detail
                    if not executed_volume:
                        executed_volume = expected_volume

                    try:
                        if executed_volume and executed_volume > 0:
                            grid['has_position'] = True
                            grid['volume'] += executed_volume
                            grid['buy_price'] = grid['price']
                            grid['buy_order_uuid'] = None
                            logger.info(f"[{self.coin}] ✅ 매수 체결(부분/전체 반영): {grid['price']:,.0f}원 | 체결량: {executed_volume:.6f}")

                            # 잔여가 있고 옵션 활성화 시 시장가로 잔여 채움 시도
                            if FILL_REMAINING_MARKET and expected_volume and executed_volume < expected_volume:
                                remaining = expected_volume - executed_volume
                                if remaining / expected_volume <= MARKET_FILL_IF_LESS_THAN_RATIO:
                                    logger.info(f"[{self.coin}] 잔여량 시장가 채움 시도: remaining={remaining:.8f}")
                                    morder = None
                                    try:
                                        morder = self.upbit.buy_market_order(self.market, remaining)
                                    except Exception as e:
                                        logger.warning(f"[{self.coin}] 시장가 잔여 매수 실패: {e}")

                                    if morder:
                                        grid['volume'] += remaining
                                        logger.info(f"[{self.coin}] 잔여량 시장가 매수 완료: {remaining:.8f}")
                                    else:
                                        logger.warning(f"[{self.coin}] 잔여량 시장가 매수 실패: remaining={remaining:.8f}")
                        else:
                            grid['buy_order_uuid'] = None
                            logger.info(f"[{self.coin}] 매수 주문 체결됨(체결량 확인 불가) uuid={uuid}")
                    except Exception as e:
                        logger.error(f"[{self.coin}] 매수 체결 반영 실패: {e}")

                elif side == 'sell' and grid.get('sell_order_uuid') == uuid:
                    if grid_index + 1 < len(self.grids):
                        sell_price = self.grids[grid_index + 1]['price']
                        profit = (sell_price - grid['buy_price']) * grid['volume']
                        net_profit = profit * 0.999
                        self.total_profit += net_profit
                        self.trade_count += 1
                        logger.info(f"[{self.coin}] ✅ 매도 체결: {sell_price:,.0f}원 | 수익: +{net_profit:,.0f}원")
                        send_discord_notification(f"**{self.coin} 매도 체결! 💰**\n• 수익: +{net_profit:,.0f}원\n• 누적: {self.total_profit:,.0f}원\n• 거래: {self.trade_count}회", color=3066993)

                    grid['has_position'] = False
                    grid['volume'] = 0.0
                    grid['buy_price'] = 0.0
                    grid['sell_order_uuid'] = None

                try:
                    del self.pending_orders[uuid]
                except KeyError:
                    pass
        except Exception as e:
            logger.error(f"[{self.coin}] 체결 확인 실패: {e}")
    
    def reset_grids_if_needed(self):
        try:
            current_price = self.get_current_price()
            if not current_price:
                return
            
            if current_price > self.upper_price * 1.05 or current_price < self.lower_price * 0.95:
                logger.info(f"[{self.coin}] 그리드 재설정...")
                open_orders = self.upbit.get_order(self.market, state='wait')
                if open_orders:
                    for order in open_orders:
                        try:
                            self.upbit.cancel_order(order['uuid'])
                            time.sleep(API_CALL_INTERVAL)
                        except:
                            pass
                
                total_volume = sum(grid['volume'] for grid in self.grids if grid['has_position'])
                if total_volume > 0:
                    execute_sell_market_order(self.upbit, self.market, total_volume)
                    time.sleep(API_CALL_INTERVAL)
                
                self.pending_orders.clear()
                time.sleep(1)
                self.initialize_grids()
        except Exception as e:
            logger.error(f"[{self.coin}] 그리드 재설정 실패: {e}")
    
    def run_cycle(self):
        self.reset_grids_if_needed()
        self.place_grid_orders()
        time.sleep(API_CALL_INTERVAL)
        self.check_filled_orders()
        
        # 고급 기능 실행
        self.adjust_portfolio_balance()
        self.check_trailing_stop()
        self.add_waterfall_grids()
    
    def get_stats(self) -> Dict:
        runtime = datetime.now() - self.start_time
        return {
            'coin': self.coin,
            'coin_name': self.coin_name,
            'total_profit': self.total_profit,
            'trade_count': self.trade_count,
            'runtime': str(runtime).split('.')[0],
            'profit_rate': (self.total_profit / self.initial_capital * 100) if self.initial_capital > 0 else 0,
            'capital': self.capital,
            'volatility': self.volatility
        }
    
    def adjust_portfolio_balance(self):
        """포트폴리오 밸런스 조정 - 각 코인의 목표 비율 유지"""
        try:
            total_balance = sum(bot.capital for bot in self.manager.bots)
            target_capital = total_balance * self.portion
            
            if abs(self.capital - target_capital) > self.min_order_amount * 10:  # 최소 조정 금액
                if self.capital > target_capital:
                    # 자본이 많으면 매도
                    excess = self.capital - target_capital
                    current_price = self.get_current_price()
                    if current_price:
                        sell_volume = excess / current_price
                        # 현재 가격으로 시장가 매도
                        order = self.upbit.sell_market_order(self.market, sell_volume)
                        if order:
                            logger.info(f"[{self.coin}] 포트폴리오 밸런스 조정: {excess:,.0f}원 매도")
                            self.capital -= excess
                else:
                    # 자본이 적으면 매수
                    deficit = target_capital - self.capital
                    current_price = self.get_current_price()
                    if current_price:
                        buy_volume = deficit / current_price
                        # 현재 가격으로 시장가 매수
                        order = self.upbit.buy_market_order(self.market, buy_volume)
                        if order:
                            logger.info(f"[{self.coin}] 포트폴리오 밸런스 조정: {deficit:,.0f}원 매수")
                            self.capital += deficit
        except Exception as e:
            logger.error(f"[{self.coin}] 포트폴리오 밸런스 조정 실패: {e}")
    
    def check_trailing_stop(self):
        """트레일링 스탑 - 수익 보호"""
        try:
            if not hasattr(self, 'highest_price'):
                self.highest_price = self.get_current_price() or 0
                return
            
            current_price = self.get_current_price()
            if not current_price:
                return
            
            # 최고가 업데이트
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # 하락률 계산
            drop_rate = (self.highest_price - current_price) / self.highest_price
            
            # 1.1% 하락 시 부분 매도 (50%)
            if drop_rate >= 0.011 and not hasattr(self, 'partial_sold'):
                total_position_value = sum(grid['volume'] * grid['buy_price'] for grid in self.grids if grid['has_position'])
                if total_position_value > 0:
                    sell_value = total_position_value * 0.5  # 50% 매도
                    sell_volume = sell_value / current_price
                    
                    order = self.upbit.sell_market_order(self.market, sell_volume)
                    if order:
                        logger.info(f"[{self.coin}] 트레일링 스탑 부분 매도: {sell_value:,.0f}원 (1.1% 하락)")
                        send_discord_notification(f"**{self.coin} 트레일링 스탑! ⚠️**\n• 부분 매도: {sell_value:,.0f}원\n• 하락률: {drop_rate:.2f}%", color=15105570)
                        self.partial_sold = True
            
            # 0.3% 추가 하락 시 전체 스탑 (3.3% 총 하락)
            if drop_rate >= 0.033:
                # 모든 포지션 청산
                for grid in self.grids:
                    if grid['has_position'] and grid['volume'] > 0:
                        order = self.upbit.sell_market_order(self.market, grid['volume'])
                        if order:
                            logger.info(f"[{self.coin}] 트레일링 스탑 전체 청산: 그리드 {grid['index']}")
                
                logger.info(f"[{self.coin}] 트레일링 스탑 발동: 전체 포지션 청산 ({drop_rate:.2f}% 하락)")
                send_discord_notification(f"**{self.coin} 트레일링 스탑 발동! 🚨**\n• 전체 청산 완료\n• 총 하락률: {drop_rate:.2f}%", color=15158332)
                
                # 봇 중지 또는 재시작 로직
                self.stop_trading = True
                
        except Exception as e:
            logger.error(f"[{self.coin}] 트레일링 스탑 체크 실패: {e}")
    
    def add_waterfall_grids(self):
        """워터폴 그리드 - 가격 하락 시 추가 그리드 생성"""
        try:
            current_price = self.get_current_price()
            if not current_price:
                return
            
            # 현재 가격이 하단 가격보다 5% 이상 낮으면 추가 그리드 생성
            if self.lower_price and current_price < self.lower_price * 0.95:
                # 잔고 확인
                balances = self.upbit.get_balances()
                available_balance = 0
                for balance in balances:
                    if balance['currency'] == 'KRW':
                        available_balance = float(balance['balance'])
                        break
                
                logger.info(f"[{self.coin}] 워터폴 그리드 잔고 확인: {available_balance:,.0f}원")
                
                # 새로운 그리드 가격 계산 (현재 가격보다 2% 낮게)
                new_grid_price = current_price * 0.98
                # 최소 가격 단위로 반올림
                new_grid_price = round_price(new_grid_price, self.coin)
                new_grid_index = len(self.grids)
                
                # 새로운 그리드에 필요한 금액 계산
                temp_grid_count = self.grid_count + 1
                amount_per_grid = self.capital / temp_grid_count
                
                logger.info(f"[{self.coin}] 워터폴 그리드 계산: 현재가 {current_price:,.0f}원, 새그리드가격 {new_grid_price:,.0f}원, 금액 {amount_per_grid:,.0f}원, 최소주문 {self.min_order_amount:,.0f}원")
                
                # 최소 주문 금액 및 잔고 확인
                if amount_per_grid < self.min_order_amount:
                    logger.info(f"[{self.coin}] 워터폴 그리드 추가 취소: 최소 주문 금액 미달 ({amount_per_grid:,.0f} < {self.min_order_amount:,.0f})")
                    return
                    
                if available_balance < amount_per_grid:
                    logger.info(f"[{self.coin}] 워터폴 그리드 추가 취소: 잔고 부족 ({available_balance:,.0f} < {amount_per_grid:,.0f})")
                    return
                
                # 새로운 그리드 추가
                new_grid = {
                    'index': new_grid_index,
                    'price': new_grid_price,
                    'has_position': False,
                    'volume': 0.0,
                    'buy_price': 0.0,
                    'buy_order_uuid': None,
                    'sell_order_uuid': None
                }
                
                self.grids.append(new_grid)
                self.grid_count += 1
                
                # 새로운 그리드에 매수 주문
                volume = amount_per_grid / new_grid_price
                volume = float(Decimal(str(volume)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
                
                # 최소 수량 제한 적용
                min_volume = self.min_order_amount / new_grid_price
                if volume < min_volume:
                    volume = min_volume
                    logger.info(f"[{self.coin}] 최소 수량으로 조정: {volume:.8f}")
                
                logger.info(f"[{self.coin}] 워터폴 그리드 주문 시도: 가격 {new_grid_price:,.0f}원, 수량 {volume:.8f}")
                
                order = execute_buy_limit_order(self.upbit, self.market, new_grid_price, volume)
                if order and 'uuid' in order:
                    new_grid['buy_order_uuid'] = order['uuid']
                    self.pending_orders[order['uuid']] = {
                        'grid_index': new_grid_index,
                        'side': 'buy',
                        'price': new_grid_price,
                        'volume': volume
                    }
                    
                    logger.info(f"[{self.coin}] 워터폴 그리드 추가 성공: {new_grid_price:,.0f}원 (총 {self.grid_count}개)")
                    send_discord_notification(f"**{self.coin} 워터폴 그리드 추가! 🌊**\n• 현재가: {current_price:,.0f}원\n• 새 그리드: {new_grid_price:,.0f}원\n• 총 그리드: {self.grid_count}개", color=1752220)
                else:
                    # 주문 실패 시 그리드 제거
                    self.grids.pop()
                    self.grid_count -= 1
                    logger.warning(f"[{self.coin}] 워터폴 그리드 주문 실패로 그리드 제거")
                    
        except Exception as e:
            logger.error(f"[{self.coin}] 워터폴 그리드 추가 실패: {e}")

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
            
            capital_per_coin = INITIAL_CAPITAL / len(COINS)
            for coin in COINS:
                config = COIN_CONFIGS.get(coin, {})
                portion = config.get('portion', 1.0 / len(COINS))  # 기본적으로 균등 분배
                capital_for_coin = INITIAL_CAPITAL * portion
                bot = GridTradingBot(self.upbit, coin, capital_for_coin, self, portion)
                if bot.initialize_grids():
                    self.bots.append(bot)
                    time.sleep(API_CALL_INTERVAL * 2)
            
            if not self.bots:
                logger.error("초기화된 봇이 없습니다!")
                return False
            
            logger.info(f"총 {len(self.bots)}개 봇 초기화 완료")
            send_discord_notification(f"🚀 **그리드 봇 시작!**\n• 코인: {', '.join([bot.coin for bot in self.bots])}\n• 총 투자금: {INITIAL_CAPITAL:,}원", color=3447003)
            return True
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            return False
    
    def run(self):
        logger.info("그리드 트레이딩 봇 시작!")
        last_report_time = datetime.now()
        
        try:
            while True:
                peak_time = is_peak_time()
                
                for bot in self.bots:
                    try:
                        bot.run_cycle()
                        time.sleep(API_CALL_INTERVAL if peak_time else API_CALL_INTERVAL * 1.5)
                    except Exception as e:
                        logger.error(f"[{bot.coin}] 사이클 오류: {e}")
                
                if datetime.now() - last_report_time > timedelta(hours=1):
                    self.send_report()
                    last_report_time = datetime.now()
                
                time.sleep(5 if peak_time else 10)
        
        except KeyboardInterrupt:
            logger.info("봇 종료 중...")
            self.send_final_report()
        except Exception as e:
            logger.error(f"치명적 오류: {e}")
    
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