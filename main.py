import os
import time
import requests
import threading
import numpy as np
import pandas as pd
import uvicorn
from xgboost import XGBClassifier
from fastapi import FastAPI
from datetime import datetime, timedelta
from collections import deque
import json

app = FastAPI()

# =============================================================================
# CONFIGURATION
# =============================================================================
PROXY_URL = os.environ.get("PROXY_URL", "https://crypto-proxy-bot.uijayasingha.workers.dev")

# Risk Management
RISK_PER_TRADE_PCT = 0.01          # 1% risk per trade
MAX_TRADES_PER_SESSION = 2         # Max 2 trades per killzone
MAX_DAILY_TRADES = 4               # Max 4 trades per day
MIN_RR_RATIO = 2.0                 # Minimum 1:2 Risk:Reward

# Session Probability Thresholds (24/7 with session weighting)
SESSION_THRESHOLDS = {
    "ASIAN":      {"buy_prob": 0.82, "sell_prob": 0.18},    # Very strict
    "LONDON":     {"buy_prob": 0.68, "sell_prob": 0.32},    # Normal
    "LONDON_NY":  {"buy_prob": 0.62, "sell_prob": 0.38},    # Best session
    "NY":         {"buy_prob": 0.65, "sell_prob": 0.35},    # Good
    "NY_CLOSE":   {"buy_prob": 0.75, "sell_prob": 0.25},    # Strict
}

FEATURE_COLUMNS = [
    'footprint_delta', 'cvd_divergence', 'wick_absorption', 'liquidity_sweep',
    'fvg_imbalance', 'volatility_squeeze', 'vsa_anomaly', 'anchored_vwap_dev',
    'session_killzone', 'premium_discount_zone'
]

# =============================================================================
# THREAD-SAFE GLOBAL STATE
# =============================================================================
state_lock = threading.Lock()
active_trade = None
active_entry_price = 0.0
active_stop_loss = 0.0
active_target_price = 0.0
active_position_size = 0.0
active_trade_risk = 0.0
latest_bot_state = "INITIALIZING SMC ENGINE..."
scalp_model_5m = None
last_model_train_time = None
is_using_fallback = False

# Performance Tracking
trade_history = deque(maxlen=100)
daily_stats = {"wins": 0, "losses": 0, "be": 0, "total_r": 0.0, "date": datetime.utcnow().date()}
account_balance = 1000.0  # USD - Update via webhook or env var

@app.get("/")
def home():
    with state_lock:
        win_rate = (daily_stats["wins"] / max(daily_stats["wins"] + daily_stats["losses"], 1)) * 100
        total_trades = daily_stats["wins"] + daily_stats["losses"] + daily_stats["be"]
        avg_r = daily_stats["total_r"] / max(total_trades, 1)

        return {
            "status": "SMC INSTITUTIONAL QUANT ENGINE",
            "version": "3.0-SMC",
            "current_state": latest_bot_state,
            "active_trade": active_trade,
            "entry": active_entry_price,
            "stop_loss": active_stop_loss,
            "target": active_target_price,
            "position_size": active_position_size,
            "using_fallback": is_using_fallback,
            "proxy": PROXY_URL,
            "performance": {
                "win_rate_pct": round(win_rate, 1),
                "total_trades_today": total_trades,
                "wins": daily_stats["wins"],
                "losses": daily_stats["losses"],
                "avg_r": round(avg_r, 2),
                "account_balance": account_balance
            },
            "trade_history": list(trade_history)[-10:]
        }

@app.post("/webhook")
def receive_signal(data: dict):
    global account_balance
    if "balance" in data:
        account_balance = float(data["balance"])
        print(f"💰 Account balance updated: ${account_balance}", flush=True)
    print(f"📥 [WEBHOOK]: {data}", flush=True)
    return {"status": "success"}

# =============================================================================
# DATA FETCHING
# =============================================================================
def generate_fallback_data(limit=200):
    np.random.seed(int(time.time()) % 10000)
    base_price = 65000.0
    dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=limit, freq='5min')
    returns = np.random.normal(0.00005, 0.001, limit)
    price_path = base_price * np.exp(np.cumsum(returns))
    volume = np.random.uniform(300, 1500, limit)
    taker_buy = volume * np.random.uniform(0.45, 0.55, limit)
    df = pd.DataFrame({
        'open_time': dates,
        'open': price_path * (1 + np.random.normal(0, 0.0001, limit)),
        'high': price_path * (1 + np.abs(np.random.normal(0, 0.0005, limit))),
        'low': price_path * (1 - np.abs(np.random.normal(0, 0.0005, limit))),
        'close': price_path,
        'volume': volume,
        'taker_buy_base_vol': taker_buy
    })
    return df

def fetch_from_proxy(symbol="BTCUSDT", interval="5m", limit=200):
    url = f"{PROXY_URL}/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    try:
        res = requests.get(url, params=params, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                cols = ['open', 'high', 'low', 'close', 'volume']
                df[cols] = df[cols].astype(float)
                df['taker_buy_base_vol'] = df['taker_buy_base'].astype(float)
                df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms', utc=True)
                return df
    except Exception as e:
        print(f"❌ Proxy Error: {e}", flush=True)
    return pd.DataFrame()

def fetch_crypto_data(symbol="BTCUSDT", interval="5m", limit=200):
    global is_using_fallback
    df = fetch_from_proxy(symbol, interval, limit)
    if not df.empty and len(df) >= 30:
        with state_lock:
            is_using_fallback = False
        return df
    print("⚠️ Using Fallback Data...", flush=True)
    with state_lock:
        is_using_fallback = True
    return generate_fallback_data(limit=limit)

# =============================================================================
# MARKET STRUCTURE DETECTION (SMC CORE)
# =============================================================================
def detect_swing_points(df, left=3, right=3):
    """Detect swing highs and lows using fractal logic"""
    highs = df['high']
    lows = df['low']

    swing_highs = pd.Series(False, index=df.index)
    swing_lows = pd.Series(False, index=df.index)

    for i in range(left, len(df) - right):
        # Swing High: higher than left and right candles
        if all(highs.iloc[i] >= highs.iloc[i-j] for j in range(1, left+1)) and \
           all(highs.iloc[i] >= highs.iloc[i+j] for j in range(1, right+1)):
            swing_highs.iloc[i] = True

        # Swing Low: lower than left and right candles
        if all(lows.iloc[i] <= lows.iloc[i-j] for j in range(1, left+1)) and \
           all(lows.iloc[i] <= lows.iloc[i+j] for j in range(1, right+1)):
            swing_lows.iloc[i] = True

    return swing_highs, swing_lows

def detect_market_structure(df):
    """
    Detect BOS (Break of Structure) and CHoCH (Change of Character)
    Returns: structure_bias, last_bos_price, last_choch_price, structure_type
    """
    swing_highs, swing_lows = detect_swing_points(df)

    sh_indices = df.index[swing_highs].tolist()
    sl_indices = df.index[swing_lows].tolist()

    if len(sh_indices) < 2 or len(sl_indices) < 2:
        return "NEUTRAL", 0, 0, "NONE"

    structure_bias = "NEUTRAL"
    last_bos = 0
    last_choch = 0
    struct_type = "NONE"

    # Bullish BOS: Price breaks above previous swing high
    recent_highs = [df.loc[idx, 'high'] for idx in sh_indices[-5:]]
    recent_lows = [df.loc[idx, 'low'] for idx in sl_indices[-5:]]

    current_close = df['close'].iloc[-1]
    prev_high = recent_highs[-2] if len(recent_highs) >= 2 else 0
    prev_low = recent_lows[-2] if len(recent_lows) >= 2 else 0

    # BOS Bullish: Close above previous swing high
    if current_close > prev_high and prev_high > 0:
        structure_bias = "BULLISH"
        last_bos = prev_high
        struct_type = "BOS_BULLISH"

    # BOS Bearish: Close below previous swing low
    elif current_close < prev_low and prev_low > 0:
        structure_bias = "BEARISH"
        last_bos = prev_low
        struct_type = "BOS_BEARISH"

    # CHoCH Bullish: Higher low after lower lows (trend reversal up)
    if len(recent_lows) >= 3:
        if recent_lows[-1] > recent_lows[-2] and recent_lows[-2] < recent_lows[-3]:
            if structure_bias != "BULLISH":
                structure_bias = "BULLISH_CHoCH"
                last_choch = recent_lows[-2]
                struct_type = "CHoCH_BULLISH"

    # CHoCH Bearish: Lower high after higher highs (trend reversal down)
    if len(recent_highs) >= 3:
        if recent_highs[-1] < recent_highs[-2] and recent_highs[-2] > recent_highs[-3]:
            if structure_bias != "BEARISH":
                structure_bias = "BEARISH_CHoCH"
                last_choch = recent_highs[-2]
                struct_type = "CHoCH_BEARISH"

    return structure_bias, last_bos, last_choch, struct_type

def get_major_structure_levels(df, lookback=50):
    """Get major swing highs/lows as key levels"""
    swing_highs, swing_lows = detect_swing_points(df, left=5, right=5)

    major_highs = df.loc[swing_highs, 'high'].tail(3).tolist()
    major_lows = df.loc[swing_lows, 'low'].tail(3).tolist()

    return {
        'resistance_levels': sorted(major_highs, reverse=True),
        'support_levels': sorted(major_lows),
        'recent_swing_high': major_highs[-1] if major_highs else df['high'].max(),
        'recent_swing_low': major_lows[-1] if major_lows else df['low'].min()
    }

# =============================================================================
# SMC FEATURES: FVG, ORDER BLOCKS, LIQUIDITY, VOLUME PROFILE
# =============================================================================
def build_smc_features(df):
    """Complete SMC feature set for institutional-grade analysis"""
    if df is None or df.empty or len(df) < 30:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # ─── ORDER FLOW & FOOTPRINT ───
    df['taker_sell_vol'] = np.maximum(0, df['volume'] - df['taker_buy_base_vol'])
    df['footprint_delta'] = df['taker_buy_base_vol'] - df['taker_sell_vol']
    df['delta_percent'] = df['footprint_delta'] / np.maximum(df['volume'], 1e-8)
    df['cvd'] = df['footprint_delta'].cumsum()
    df['cvd_divergence'] = df['cvd'].pct_change().fillna(0) - df['close'].pct_change().fillna(0)

    # ─── WICK ANALYSIS (Absorption / Rejection) ───
    candle_range = np.maximum(df['high'] - df['low'], 1e-8)
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    df['wick_absorption'] = (upper_wick - lower_wick) / candle_range
    df['upper_wick_pct'] = upper_wick / candle_range
    df['lower_wick_pct'] = lower_wick / candle_range

    # ─── LIQUIDITY SWEEPS ───
    df['prev_high_20'] = df['high'].rolling(20).max().shift(1)
    df['prev_low_20'] = df['low'].rolling(20).min().shift(1)
    df['liquidity_sweep'] = np.where(
        df['high'] > df['prev_high_20'], 1,
        np.where(df['low'] < df['prev_low_20'], -1, 0)
    )
    df['sweep_strength'] = np.where(
        df['liquidity_sweep'] == 1, (df['high'] - df['prev_high_20']) / df['close'],
        np.where(df['liquidity_sweep'] == -1, (df['prev_low_20'] - df['low']) / df['close'], 0)
    )

    # ─── FAIR VALUE GAPS (FVG) ───
    # Bullish FVG: Low[i] > High[i-2]
    df['fvg_bullish'] = np.maximum(0, df['low'] - df['high'].shift(2))
    # Bearish FVG: Low[i-2] > High[i]
    df['fvg_bearish'] = np.maximum(0, df['low'].shift(2) - df['high'])
    df['fvg_imbalance'] = (df['fvg_bullish'] - df['fvg_bearish']) / df['close']
    df['fvg_total'] = (df['fvg_bullish'] + df['fvg_bearish']) / df['close']

    # FVG Fill Status (for entry timing)
    df['fvg_bull_filled'] = df['low'] <= df['high'].shift(2)
    df['fvg_bear_filled'] = df['high'] >= df['low'].shift(2)

    # ─── ORDER BLOCK APPROXIMATION ───
    # Bullish OB: Strong bearish candle before bullish move
    df['body'] = df['close'] - df['open']
    df['prev_body'] = df['body'].shift(1)
    df['bullish_ob'] = (df['prev_body'] < 0) & (df['body'] > 0) & (abs(df['prev_body']) > candle_range * 0.6)
    df['bearish_ob'] = (df['prev_body'] > 0) & (df['body'] < 0) & (abs(df['prev_body']) > candle_range * 0.6)

    # ─── VOLATILITY & SQUEEZE ───
    df['atr'] = candle_range.rolling(14).mean()
    df['atr_percentile'] = df['atr'].rolling(100).rank(pct=True).fillna(0.5)
    df['volatility_squeeze'] = np.where(df['atr_percentile'] < 0.20, 1, 0)

    # ─── VOLUME PROFILE / HEAT MAP APPROXIMATION ───
    vol_mean = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / np.maximum(vol_mean, 1e-8)
    df['vsa_anomaly'] = df['volume_ratio'] / (candle_range / df['close'])

    # Volume at price level (recent window)
    df['vpoc'] = df['close'].rolling(20).apply(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[-1], raw=False)

    # ─── VWAP (Rolling Anchor) ───
    vwap_window = 48
    pv = df['close'] * df['volume']
    rolling_pv = pv.rolling(window=vwap_window).sum()
    rolling_vol = df['volume'].rolling(window=vwap_window).sum()
    vwap = rolling_pv / np.maximum(rolling_vol, 1e-8)
    vwap_std = df['close'].rolling(vwap_window).std().fillna(1.0)
    df['anchored_vwap_dev'] = (df['close'] - vwap) / np.maximum(vwap_std, 1e-8)
    df['vwap'] = vwap

    # ─── PREMIUM / DISCOUNT ZONES (True SMC Style) ───
    range_high = df['high'].rolling(30).max()
    range_low = df['low'].rolling(30).min()
    price_range = np.maximum(range_high - range_low, 1e-8)
    df['premium_discount_zone'] = (df['close'] - range_low) / price_range

    # Zone classification
    df['is_premium'] = df['premium_discount_zone'] > 0.70
    df['is_discount'] = df['premium_discount_zone'] < 0.30
    df['is_equilibrium'] = (df['premium_discount_zone'] >= 0.30) & (df['premium_discount_zone'] <= 0.70)

    # ─── SESSION / KILLZONE ───
    hours = df['open_time'].dt.hour
    df['session_killzone'] = np.where(
        (hours >= 8) & (hours <= 11), 1,      # London
        np.where((hours >= 13) & (hours <= 16), 2,   # NY
        np.where((hours >= 7) & (hours <= 10), 1,    # Pre-London
        0))  # Asian / Other
    )

    df['session_name'] = np.where(
        (hours >= 0) & (hours < 7), "ASIAN",
        np.where((hours >= 7) & (hours < 12), "LONDON",
        np.where((hours >= 12) & (hours < 17), "LONDON_NY",
        np.where((hours >= 17) & (hours < 21), "NY", "NY_CLOSE")))
    )

    return df


def get_session_threshold(session_name):
    """Get probability thresholds based on trading session"""
    return SESSION_THRESHOLDS.get(session_name, SESSION_THRESHOLDS["ASIAN"])

# =============================================================================
# MULTI-TIMEFRAME ANALYSIS (HTF Bias + LTF Execution)
# =============================================================================
def get_htf_bias(symbol="BTCUSDT", htf="4h", limit=100):
    """Higher Timeframe Bias: Bullish / Bearish / Neutral"""
    df = fetch_crypto_data(symbol, htf, limit)
    if df.empty or len(df) < 20:
        return "NEUTRAL", 0.50, {}

    df = build_smc_features(df)
    bias, last_bos, last_choch, struct_type = detect_market_structure(df)
    levels = get_major_structure_levels(df)

    # EMA alignment
    ema20 = df['close'].ewm(span=20).mean().iloc[-1]
    ema50 = df['close'].ewm(span=50).mean().iloc[-1]
    curr_close = df['close'].iloc[-1]

    ema_aligned = (curr_close > ema20 > ema50) or (curr_close < ema20 < ema50)

    # PD zone on HTF
    pd_zone = df['premium_discount_zone'].iloc[-1]

    # Determine bias
    if "BULLISH" in bias and curr_close > ema20:
        htf_bias = "BULLISH"
    elif "BEARISH" in bias and curr_close < ema20:
        htf_bias = "BEARISH"
    else:
        htf_bias = "NEUTRAL"

    return htf_bias, pd_zone, {
        'structure_type': struct_type,
        'last_bos': last_bos,
        'last_choch': last_choch,
        'ema20': ema20,
        'ema50': ema50,
        'levels': levels
    }

def get_itf_direction(symbol="BTCUSDT", itf="1h", limit=100):
    """Intermediate Timeframe: Directional bias + PD arrays"""
    df = fetch_crypto_data(symbol, itf, limit)
    if df.empty or len(df) < 20:
        return "NEUTRAL", 0.50, {}

    df = build_smc_features(df)
    bias, last_bos, last_choch, struct_type = detect_market_structure(df)
    levels = get_major_structure_levels(df)

    pd_zone = df['premium_discount_zone'].iloc[-1]
    curr_close = df['close'].iloc[-1]

    # Key PD arrays
    recent_high = df['high'].rolling(20).max().iloc[-1]
    recent_low = df['low'].rolling(20).min().iloc[-1]

    if "BULLISH" in bias:
        direction = "BULLISH"
    elif "BEARISH" in bias:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return direction, pd_zone, {
        'structure_type': struct_type,
        'recent_high': recent_high,
        'recent_low': recent_low,
        'levels': levels
    }

# =============================================================================
# ENTRY ENGINE: SMC CONFLUENCE SYSTEM
# =============================================================================
def evaluate_entry_setup(live_df, htf_bias, htf_info, itf_direction, itf_info):
    """
    A+ Setup Requirements:
    1. HTF Bias aligns with trade direction
    2. Price at PD zone (Discount for long, Premium for short)
    3. LTF Structure shift (CHoCH / MSS)
    4. FVG present (for entry precision)
    5. Liquidity sweep confirmation
    6. Order flow confirmation (delta divergence)
    7. Killzone / Session appropriate
    8. Volume confirmation
    """
    if live_df.empty or len(live_df) < 10:
        return None, 0.0, "NO_DATA"

    current = live_df.iloc[-1]
    prev = live_df.iloc[-2] if len(live_df) >= 2 else current

    price = current['close']
    pd_zone = current['premium_discount_zone']
    session = current['session_name']

    # Get session thresholds
    thresholds = get_session_threshold(session)

    # Structure detection on LTF
    ltf_bias, ltf_bos, ltf_choch, ltf_struct = detect_market_structure(live_df)

    # ─── LONG SETUP EVALUATION ───
    long_score = 0.0
    long_reasons = []

    # 1. HTF Bias (25 points)
    if htf_bias == "BULLISH":
        long_score += 25
        long_reasons.append("HTF_BULLISH")

    # 2. PD Zone - Discount (25 points)
    if pd_zone < 0.35:
        long_score += 25
        long_reasons.append("DISCOUNT_ZONE")
    elif pd_zone < 0.50:
        long_score += 15
        long_reasons.append("NEAR_DISCOUNT")

    # 3. LTF Structure Shift (20 points)
    if "BULLISH" in ltf_bias or "CHoCH_BULLISH" == ltf_struct:
        long_score += 20
        long_reasons.append("LTF_CHoCH")
    elif "BOS_BULLISH" == ltf_struct:
        long_score += 15
        long_reasons.append("LTF_BOS")

    # 4. FVG Present (10 points)
    if current['fvg_bullish'] > 0:
        long_score += 10
        long_reasons.append("BULLISH_FVG")

    # 5. Liquidity Sweep + Reclaim (10 points)
    if current['liquidity_sweep'] == -1 and price > prev['low']:  # Swept low, reclaimed
        long_score += 10
        long_reasons.append("SWEEP_RECLAIM")

    # 6. Order Flow (5 points)
    if current['delta_percent'] > 0.10:  # Strong buying delta
        long_score += 5
        long_reasons.append("BUYING_DELTA")

    # 7. Volume (5 points)
    if current['volume_ratio'] > 1.2:
        long_score += 5
        long_reasons.append("VOLUME_CONFIRM")

    long_prob = long_score / 100.0

    # ─── SHORT SETUP EVALUATION ───
    short_score = 0.0
    short_reasons = []

    if htf_bias == "BEARISH":
        short_score += 25
        short_reasons.append("HTF_BEARISH")

    if pd_zone > 0.65:
        short_score += 25
        short_reasons.append("PREMIUM_ZONE")
    elif pd_zone > 0.50:
        short_score += 15
        short_reasons.append("NEAR_PREMIUM")

    if "BEARISH" in ltf_bias or "CHoCH_BEARISH" == ltf_struct:
        short_score += 20
        short_reasons.append("LTF_CHoCH")
    elif "BOS_BEARISH" == ltf_struct:
        short_score += 15
        short_reasons.append("LTF_BOS")

    if current['fvg_bearish'] > 0:
        short_score += 10
        short_reasons.append("BEARISH_FVG")

    if current['liquidity_sweep'] == 1 and price < prev['high']:
        short_score += 10
        short_reasons.append("SWEEP_RECLAIM")

    if current['delta_percent'] < -0.10:
        short_score += 5
        short_reasons.append("SELLING_DELTA")

    if current['volume_ratio'] > 1.2:
        short_score += 5
        short_reasons.append("VOLUME_CONFIRM")

    short_prob = short_score / 100.0

    # ─── DECISION ───
    if long_prob >= thresholds['buy_prob'] and long_prob > short_prob + 0.15:
        return "BUY", long_prob, "|".join(long_reasons)
    elif short_prob >= (1 - thresholds['sell_prob']) and short_prob > long_prob + 0.15:
        return "SELL", short_prob, "|".join(short_reasons)

    return None, max(long_prob, short_prob), "NO_SETUP"

# =============================================================================
# RISK MANAGEMENT & POSITION SIZING
# =============================================================================
def calculate_position_size(entry_price, stop_loss, account_bal=None):
    """Calculate position size based on 1% risk rule"""
    global account_balance
    bal = account_bal if account_bal else account_balance
    risk_amount = bal * RISK_PER_TRADE_PCT
    sl_distance = abs(entry_price - stop_loss)

    if sl_distance <= 0:
        return 0.0

    position_size = risk_amount / sl_distance
    return position_size

def calculate_stop_loss(entry, direction, live_df, htf_levels):
    """Structure-based stop loss"""
    if direction == "BUY":
        # SL below recent structure low or swing low
        recent_low = live_df['low'].tail(5).min()
        swing_lows = live_df['low'].rolling(3).min().shift(1).tail(3)
        struct_sl = swing_lows.min() if not swing_lows.empty else recent_low * 0.995
        # Also consider HTF support
        htf_support = htf_levels.get('support_levels', [recent_low])[0] if htf_levels else recent_low
        sl = min(struct_sl, htf_support * 0.998)
        return sl
    else:
        recent_high = live_df['high'].tail(5).max()
        swing_highs = live_df['high'].rolling(3).max().shift(1).tail(3)
        struct_sl = swing_highs.max() if not swing_highs.empty else recent_high * 1.005
        htf_resist = htf_levels.get('resistance_levels', [recent_high])[0] if htf_levels else recent_high
        sl = max(struct_sl, htf_resist * 1.002)
        return sl

def calculate_take_profit(entry, stop_loss, direction, live_df, htf_levels, min_rr=2.0):
    """TP at opposing PD zone or structure level with minimum R:R"""
    risk = abs(entry - stop_loss)

    if direction == "BUY":
        # Target: Premium zone or next resistance
        recent_high = live_df['high'].rolling(20).max().iloc[-1]
        htf_resist = htf_levels.get('resistance_levels', [recent_high * 1.01])[0] if htf_levels else recent_high * 1.01

        # Minimum RR
        min_tp = entry + (risk * min_rr)
        tp = max(htf_resist, min_tp)
        return tp
    else:
        recent_low = live_df['low'].rolling(20).min().iloc[-1]
        htf_support = htf_levels.get('support_levels', [recent_low * 0.99])[0] if htf_levels else recent_low * 0.99

        min_tp = entry - (risk * min_rr)
        tp = min(htf_support, min_tp)
        return tp

# =============================================================================
# TREND RIDING & TRADE MANAGEMENT
# =============================================================================
def manage_active_trade(current_price, live_df, htf_bias, htf_info):
    """
    Trend Riding Logic:
    - Trail stop based on structure (swing lows for longs, swing highs for shorts)
    - Move to breakeven at 1R
    - Take partial profits at 2R, 3R
    - Exit on HTF structure break (CHoCH on HTF)
    """
    global active_trade, active_entry_price, active_stop_loss, active_target_price

    if active_trade is None:
        return "SEARCHING", 0.0

    entry = active_entry_price
    sl = active_stop_loss
    tp = active_target_price
    risk = abs(entry - sl)

    if risk <= 0:
        return "ERROR", 0.0

    if active_trade == "BUY":
        current_r = (current_price - entry) / risk

        # Full target hit
        if current_price >= tp:
            return "TARGET_HIT", current_r

        # HTF structure broke bearish — exit immediately
        if "BEARISH" in htf_bias and htf_info.get('structure_type', '').startswith('CHoCH'):
            return "HTF_STRUCTURE_BREAK", current_r

        # Trailing stop: Below recent swing low
        recent_swing_low = live_df['low'].tail(3).min()
        if recent_swing_low > sl and current_r > 1.5:
            # Trail SL to below recent swing low, but never below BE
            new_sl = max(recent_swing_low * 0.999, entry)
            active_stop_loss = new_sl

        # Breakeven at 1R
        if current_r >= 1.0 and sl < entry:
            active_stop_loss = entry
            return "BE_TO_BREAKEVEN", current_r

        # Partial profit levels
        if current_r >= 3.0:
            return "TP_3R", current_r
        elif current_r >= 2.0:
            return "TP_2R", current_r
        elif current_r >= 1.0:
            return "TP_1R", current_r
        elif current_r <= -1.0:
            return "STOPPED_OUT", current_r
        else:
            return f"LONG_R{current_r:.1f}", current_r

    else:  # SELL
        current_r = (entry - current_price) / risk

        if current_price <= tp:
            return "TARGET_HIT", current_r

        if "BULLISH" in htf_bias and htf_info.get('structure_type', '').startswith('CHoCH'):
            return "HTF_STRUCTURE_BREAK", current_r

        recent_swing_high = live_df['high'].tail(3).max()
        if recent_swing_high < sl and current_r > 1.5:
            new_sl = min(recent_swing_high * 1.001, entry)
            active_stop_loss = new_sl

        if current_r >= 1.0 and sl > entry:
            active_stop_loss = entry
            return "BE_TO_BREAKEVEN", current_r

        if current_r >= 3.0:
            return "TP_3R", current_r
        elif current_r >= 2.0:
            return "TP_2R", current_r
        elif current_r >= 1.0:
            return "TP_1R", current_r
        elif current_r <= -1.0:
            return "STOPPED_OUT", current_r
        else:
            return f"SHORT_R{current_r:.1f}", current_r

# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================
def record_trade_exit(direction, entry, exit_price, sl, tp, reason, r_multiple):
    """Record completed trade for analytics"""
    global daily_stats, trade_history

    pnl_pct = ((exit_price - entry) / entry * 100) if direction == "BUY" else ((entry - exit_price) / entry * 100)

    trade_record = {
        "time": datetime.utcnow().isoformat(),
        "direction": direction,
        "entry": round(entry, 2),
        "exit": round(exit_price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "exit_reason": reason,
        "r_multiple": round(r_multiple, 2),
        "pnl_pct": round(pnl_pct, 3)
    }

    trade_history.append(trade_record)

    # Update daily stats
    today = datetime.utcnow().date()
    if daily_stats["date"] != today:
        daily_stats = {"wins": 0, "losses": 0, "be": 0, "total_r": 0.0, "date": today}

    if r_multiple >= 2.0:
        daily_stats["wins"] += 1
    elif r_multiple <= -1.0:
        daily_stats["losses"] += 1
    else:
        daily_stats["be"] += 1

    daily_stats["total_r"] += r_multiple

    print(f"📊 TRADE CLOSED: {direction} | Entry: {entry:.2f} | Exit: {exit_price:.2f} | R: {r_multiple:.2f} | Reason: {reason}", flush=True)

# =============================================================================
# ML MODEL TRAINING (SMC-Enhanced)
# =============================================================================
def train_5m_quant_model():
    print("⏳ Fetching Multi-Timeframe Data & Training SMC Model...", flush=True)

    # Fetch 5m data via proxy
    raw_df = fetch_crypto_data(symbol="BTCUSDT", interval="5m", limit=500)
    df = build_smc_features(raw_df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    # Enhanced target: Future 6-candle return (30 min) with trend context
    future_return = (df['close'].shift(-6) - df['close']) / df['close']

    # Only label as 1 if strong bullish move AND in discount zone
    df['Target'] = np.where(
        (future_return > 0.005) & (df['premium_discount_zone'] < 0.40), 1,
        np.where(
            (future_return < -0.005) & (df['premium_discount_zone'] > 0.60), 0, 
            np.nan
        )
    )

    # Drop neutral cases for cleaner training
    df = df.dropna(subset=['Target'])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(df) < 50:
        print("⚠️ Insufficient data for training, using default model", flush=True)
        return None

    X = df[FEATURE_COLUMNS]
    y = df['Target'].astype(int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.03,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X, y)
    print(f"✅ SMC Model Trained! Pos: {n_pos}, Neg: {n_neg}, Scale: {scale_pos_weight:.2f}", flush=True)
    return model

# =============================================================================
# MAIN EXECUTION LOOP — SMC INSTITUTIONAL ENGINE
# =============================================================================
def quant_execution_loop():
    global active_trade, active_entry_price, active_stop_loss, active_target_price
    global active_position_size, latest_bot_state, scalp_model_5m
    global last_model_train_time, is_using_fallback, account_balance

    # Initial training
    scalp_model_5m = train_5m_quant_model()
    last_model_train_time = datetime.utcnow()

    loop_counter = 0
    session_trade_count = {"ASIAN": 0, "LONDON": 0, "LONDON_NY": 0, "NY": 0, "NY_CLOSE": 0}

    while True:
        try:
            loop_counter += 1
            now = datetime.utcnow()
            current_hour = now.hour

            # Reset session counts at day start
            if current_hour == 0:
                session_trade_count = {k: 0 for k in session_trade_count}

            # Retrain model every 6 hours
            if loop_counter % 4320 == 0:
                print("🔄 Scheduled model retraining...", flush=True)
                new_model = train_5m_quant_model()
                with state_lock:
                    scalp_model_5m = new_model
                    last_model_train_time = now

            # ─── MULTI-TIMEFRAME ANALYSIS ───
            print("🔍 Analyzing HTF Bias...", flush=True)
            htf_bias, htf_pd, htf_info = get_htf_bias(symbol="BTCUSDT", htf="4h", limit=100)

            print("🔍 Analyzing ITF Direction...", flush=True)
            itf_direction, itf_pd, itf_info = get_itf_direction(symbol="BTCUSDT", itf="1h", limit=100)

            print("🔍 Fetching LTF Execution Data...", flush=True)
            live_df = fetch_crypto_data(symbol="BTCUSDT", interval="5m", limit=100)

            if live_df.empty or len(live_df) < 30:
                print("⚠️ Insufficient LTF data", flush=True)
                time.sleep(10)
                continue

            live_df = build_smc_features(live_df)
            if live_df.empty:
                time.sleep(10)
                continue

            for col in FEATURE_COLUMNS:
                if col not in live_df.columns:
                    live_df[col] = 0.0

            current = live_df.iloc[-1]
            current_price = current['close']
            session_name = current['session_name']

            # ─── MANAGE ACTIVE TRADE ───
            if active_trade is not None:
                status, current_r = manage_active_trade(current_price, live_df, htf_bias, htf_info)

                if status == "TARGET_HIT":
                    record_trade_exit(active_trade, active_entry_price, current_price,
                                    active_stop_loss, active_target_price, "FULL_TARGET", current_r)
                    with state_lock:
                        active_trade = None
                    latest_bot_state = f"🎉 TARGET HIT! R{current_r:.1f} | Closed at ${current_price:.2f}"

                elif status == "STOPPED_OUT":
                    record_trade_exit(active_trade, active_entry_price, current_price,
                                    active_stop_loss, active_target_price, "STOP_LOSS", current_r)
                    with state_lock:
                        active_trade = None
                    latest_bot_state = f"🛑 STOPPED OUT! R{current_r:.1f} | Closed at ${current_price:.2f}"

                elif status == "HTF_STRUCTURE_BREAK":
                    record_trade_exit(active_trade, active_entry_price, current_price,
                                    active_stop_loss, active_target_price, "HTF_BREAK", current_r)
                    with state_lock:
                        active_trade = None
                    latest_bot_state = f"⚠️ HTF STRUCTURE BREAK! R{current_r:.1f} | Closed at ${current_price:.2f}"

                elif status == "BE_TO_BREAKEVEN":
                    latest_bot_state = f"🛡️ SL MOVED TO BREAKEVEN | R{current_r:.1f} | ${current_price:.2f}"

                elif "TP_" in status:
                    tp_level = status.replace("TP_", "")
                    latest_bot_state = f"📈 {tp_level} REACHED! | R{current_r:.1f} | ${current_price:.2f} | Trailing..."

                else:
                    latest_bot_state = f"{'🟢' if active_trade == 'BUY' else '🔴'} {active_trade} ACTIVE | R{current_r:.1f} | ${current_price:.2f} | SL: ${active_stop_loss:.2f}"

                print(f"[LIVE]: {latest_bot_state}", flush=True)
                time.sleep(5)
                continue

            # ─── NEW ENTRY EVALUATION ───
            print("🔎 Evaluating SMC Entry Setup...", flush=True)

            signal, prob, reasons = evaluate_entry_setup(
                live_df, htf_bias, htf_info, itf_direction, itf_info
            )

            # Check session trade limits
            if signal and session_trade_count.get(session_name, 0) >= MAX_TRADES_PER_SESSION:
                print(f"⛔ Session limit reached for {session_name}", flush=True)
                signal = None

            # Check daily trade limits
            total_daily = sum(session_trade_count.values())
            if signal and total_daily >= MAX_DAILY_TRADES:
                print(f"⛔ Daily trade limit reached ({total_daily})", flush=True)
                signal = None

            # ─── EXECUTE ENTRY ───
            if signal == "BUY":
                # Calculate SL and TP
                sl = calculate_stop_loss(current_price, "BUY", live_df, htf_info.get('levels', {}))
                tp = calculate_take_profit(current_price, sl, "BUY", live_df, htf_info.get('levels', {}))
                pos_size = calculate_position_size(current_price, sl)

                # Validate R:R
                risk = abs(current_price - sl)
                reward = abs(tp - current_price)
                rr = reward / risk if risk > 0 else 0

                if rr >= MIN_RR_RATIO:
                    with state_lock:
                        active_trade = "BUY"
                        active_entry_price = current_price
                        active_stop_loss = sl
                        active_target_price = tp
                        active_position_size = pos_size

                    session_trade_count[session_name] = session_trade_count.get(session_name, 0) + 1

                    latest_bot_state = (
                        f"🔥 BUY TRIGGERED @ ${current_price:.2f} | "
                        f"SL: ${sl:.2f} | TP: ${tp:.2f} | R:R {rr:.1f}:1 | "
                        f"Size: {pos_size:.4f} | Score: {prob:.0%} | {reasons}"
                    )
                    print(f"🚀 {latest_bot_state}", flush=True)
                else:
                    print(f"⛔ R:R too low ({rr:.1f}:1), skipping BUY", flush=True)

            elif signal == "SELL":
                sl = calculate_stop_loss(current_price, "SELL", live_df, htf_info.get('levels', {}))
                tp = calculate_take_profit(current_price, sl, "SELL", live_df, htf_info.get('levels', {}))
                pos_size = calculate_position_size(current_price, sl)

                risk = abs(current_price - sl)
                reward = abs(tp - current_price)
                rr = reward / risk if risk > 0 else 0

                if rr >= MIN_RR_RATIO:
                    with state_lock:
                        active_trade = "SELL"
                        active_entry_price = current_price
                        active_stop_loss = sl
                        active_target_price = tp
                        active_position_size = pos_size

                    session_trade_count[session_name] = session_trade_count.get(session_name, 0) + 1

                    latest_bot_state = (
                        f"🔻 SELL TRIGGERED @ ${current_price:.2f} | "
                        f"SL: ${sl:.2f} | TP: ${tp:.2f} | R:R {rr:.1f}:1 | "
                        f"Size: {pos_size:.4f} | Score: {prob:.0%} | {reasons}"
                    )
                    print(f"🚀 {latest_bot_state}", flush=True)
                else:
                    print(f"⛔ R:R too low ({rr:.1f}:1), skipping SELL", flush=True)

            else:
                # No setup
                htf_str = htf_bias[:4] if htf_bias else "N/A"
                itf_str = itf_direction[:4] if itf_direction else "N/A"
                pd_str = f"PD:{current['premium_discount_zone']:.2f}"

                latest_bot_state = (
                    f"⏳ SCANNING | BTC: ${current_price:.2f} | "
                    f"HTF:{htf_str} | ITF:{itf_str} | {pd_str} | "
                    f"Session:{session_name} | Score:{prob:.0%}"
                )
                print(f"[SCAN]: {latest_bot_state}", flush=True)

            time.sleep(10)

        except Exception as e:
            print(f"🔥 Loop Exception: {e}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(10)

@app.on_event("startup")
def startup_event():
    print("="*60, flush=True)
    print("🚀 SMC INSTITUTIONAL QUANT ENGINE v3.0", flush=True)
    print("📊 Style: Smart Money Concepts + Trend Capture", flush=True)
    print("🎯 Goal: Catch trends from start, ride to end", flush=True)
    print("🛡️ Risk: 1% per trade | Min R:R 2:1 | Structure-based SL", flush=True)
    print("⏰ Mode: 24/7 with session-aware probability", flush=True)
    print("="*60, flush=True)
    print(f"🔗 Proxy: {PROXY_URL}", flush=True)
    threading.Thread(target=quant_execution_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
