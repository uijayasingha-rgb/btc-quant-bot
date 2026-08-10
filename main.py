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

app = FastAPI()

FEATURE_COLUMNS = [
    'footprint_delta', 'cvd_divergence', 'wick_absorption', 'liquidity_sweep',
    'fvg_imbalance', 'volatility_squeeze', 'vsa_anomaly', 'anchored_vwap_dev',
    'session_killzone', 'premium_discount_zone'
]

# =============================================================================
# THREAD-SAFE GLOBAL STATE (Fix #1: Thread Safety)
# =============================================================================
state_lock = threading.Lock()
active_trade = None
active_entry_price = 0.0
active_target_price = 0.0
latest_bot_state = "INITIALIZING ENGINE & TRAINING MODEL..."
scalp_model_5m = None
last_model_train_time = None

@app.get("/")
def home():
    with state_lock:
        return {
            "status": "BTC Institutional Quant Engine Active",
            "current_state": latest_bot_state,
            "active_trade": active_trade,
            "entry_price": active_entry_price,
            "target_price": active_target_price
        }

@app.post("/webhook")
def receive_signal(data: dict):
    print(f"📥 [WEBHOOK RECEIVED]: {data}", flush=True)
    return {"status": "success"}

# =============================================================================
# DATA FETCHING
# =============================================================================
def generate_fallback_data(limit=200):
    """Generates realistic synthetic data so the engine never gets stuck"""
    np.random.seed(42)
    base_price = 65000.0
    dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=limit, freq='5min')

    returns = np.random.normal(0, 0.001, limit)
    price_path = base_price * np.exp(np.cumsum(returns))

    volume = np.random.uniform(100, 800, limit)  # More realistic volume range
    taker_buy = volume * np.random.uniform(0.45, 0.55, limit)

    df = pd.DataFrame({
        'open_time': dates,
        'open': price_path,
        'high': price_path * (1 + np.abs(np.random.normal(0, 0.0005, limit))),
        'low': price_path * (1 - np.abs(np.random.normal(0, 0.0005, limit))),
        'close': price_path * (1 + np.random.normal(0, 0.0002, limit)),
        'volume': volume,
        'taker_buy_base_vol': taker_buy
    })
    return df

def fetch_crypto_quant_data(symbol="BTCUSDT", interval="5", limit=200):
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={interval}&limit={limit}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            list_data = data.get("result", {}).get("list", [])
            if list_data and len(list_data) > 0:
                list_data.reverse()  # Oldest first
                df = pd.DataFrame(list_data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                cols = ['open', 'high', 'low', 'close', 'volume']
                df[cols] = df[cols].astype(float)
                # If taker_buy data not available, estimate it
                df['taker_buy_base_vol'] = df['volume'] * 0.52
                df['open_time'] = pd.to_datetime(df['open_time'].astype(float), unit='ms', utc=True)
                return df
            else:
                print(f"❌ [API ERR]: Empty Response from Bybit", flush=True)
        else:
            print(f"❌ [HTTP ERR]: Status Code {res.status_code}", flush=True)
    except Exception as e:
        print(f"❌ [FETCH EXCEPTION]: {str(e)}", flush=True)

    print("⚠️ Using Fallback Market Data...", flush=True)
    return generate_fallback_data(limit=limit)

# =============================================================================
# FEATURE ENGINEERING (Fix #2: Rolling VWAP instead of cumulative)
# =============================================================================
def build_institutional_features(df):
    if df is None or df.empty or 'close' not in df.columns or len(df) < 30:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    df['taker_sell_vol'] = np.maximum(0, df['volume'] - df['taker_buy_base_vol'])
    df['footprint_delta'] = df['taker_buy_base_vol'] - df['taker_sell_vol']
    df['cvd'] = df['footprint_delta'].cumsum()
    df['cvd_divergence'] = df['cvd'].pct_change().fillna(0) - df['close'].pct_change().fillna(0)

    candle_range = np.maximum(df['high'] - df['low'], 1e-8)
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    df['wick_absorption'] = (upper_wick - lower_wick) / candle_range

    df['liquidity_sweep'] = np.where(
        df['high'] > df['high'].rolling(20).max().shift(1), 1,
        np.where(df['low'] < df['low'].rolling(20).min().shift(1), -1, 0)
    )

    df['fvg_bullish'] = np.maximum(0, df['low'] - df['high'].shift(2))
    df['fvg_bearish'] = np.maximum(0, df['low'].shift(2) - df['high'])
    df['fvg_imbalance'] = (df['fvg_bullish'] - df['fvg_bearish']) / df['close']

    # ATR for volatility squeeze and targets
    df['atr'] = candle_range.rolling(14).mean()
    atr_percentile = df['atr'].rolling(100).rank(pct=True).fillna(0.5)
    df['volatility_squeeze'] = np.where(atr_percentile < 0.20, 1, 0)

    vol_mean = df['volume'].rolling(20).mean()
    df['vsa_anomaly'] = (df['volume'] / np.maximum(vol_mean, 1e-8)) / (candle_range / df['close'])

    # Fix #2: Rolling VWAP (48-period = 4 hours for 5min candles) instead of cumulative
    vwap_window = 48
    pv = df['close'] * df['volume']
    rolling_pv = pv.rolling(window=vwap_window).sum()
    rolling_vol = df['volume'].rolling(window=vwap_window).sum()
    vwap = rolling_pv / np.maximum(rolling_vol, 1e-8)
    vwap_std = df['close'].rolling(vwap_window).std().fillna(1.0)
    df['anchored_vwap_dev'] = (df['close'] - vwap) / np.maximum(vwap_std, 1e-8)

    range_high = df['high'].rolling(30).max()
    range_low = df['low'].rolling(30).min()
    price_range = np.maximum(range_high - range_low, 1e-8)
    df['premium_discount_zone'] = (df['close'] - range_low) / price_range

    hours = df['open_time'].dt.hour
    df['session_killzone'] = np.where(
        (hours >= 7) & (hours <= 10), 1,
        np.where((hours >= 13) & (hours <= 16), 2, 0)
    )
    return df

def get_15m_htf_structure():
    df_15m = fetch_crypto_quant_data(symbol="BTCUSDT", interval="15", limit=100)
    if df_15m.empty or 'close' not in df_15m.columns or len(df_15m) < 20:
        return 0, 0.50
    df_15m = build_institutional_features(df_15m)
    if df_15m.empty or 'close' not in df_15m.columns:
        return 0, 0.50
    ema20 = df_15m['close'].ewm(span=20).mean().iloc[-1]
    curr_close = df_15m['close'].iloc[-1]
    trend = 1 if curr_close > ema20 else -1
    htf_pd_zone = df_15m['premium_discount_zone'].iloc[-1]
    return trend, htf_pd_zone

# =============================================================================
# MODEL TRAINING (Fix #4 & #5: Periodic retrain + class imbalance handling)
# =============================================================================
def train_5m_quant_model():
    print("⏳ Fetching Market Data & Training Model...", flush=True)
    raw_df = fetch_crypto_quant_data(symbol="BTCUSDT", interval="5", limit=400)
    df = build_institutional_features(raw_df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns: 
            df[col] = 0.0

    # Future return target (3 candles ahead = 15 min)
    future_return = (df['close'].shift(-3) - df['close']) / df['close']
    df['Target'] = np.where(future_return > 0.003, 1, 0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df[FEATURE_COLUMNS]
    y = df['Target']

    # Fix #5: Handle class imbalance with scale_pos_weight
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=150, 
        max_depth=4, 
        learning_rate=0.04, 
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )
    model.fit(X, y)
    print(f"✅ AI Model Trained! Positives: {n_pos}, Negatives: {n_neg}, Scale: {scale_pos_weight:.2f}", flush=True)
    return model

# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def quant_execution_loop():
    global active_trade, active_entry_price, active_target_price
    global latest_bot_state, scalp_model_5m, last_model_train_time

    # Initial training
    scalp_model_5m = train_5m_quant_model()
    last_model_train_time = datetime.utcnow()

    loop_counter = 0

    while True:
        try:
            loop_counter += 1

            # Fix #4: Retrain model every 6 hours (4320 loops at 5s sleep)
            if loop_counter % 4320 == 0:
                print("🔄 Scheduled model retraining...", flush=True)
                new_model = train_5m_quant_model()
                with state_lock:
                    scalp_model_5m = new_model
                    last_model_train_time = datetime.utcnow()

            htf_trend, htf_pd_zone = get_15m_htf_structure()
            live_df = fetch_crypto_quant_data(symbol="BTCUSDT", interval="5", limit=100)

            if not live_df.empty and 'close' in live_df.columns and len(live_df) >= 30:
                live_df = build_institutional_features(live_df)
                if not live_df.empty and 'close' in live_df.columns:
                    for col in FEATURE_COLUMNS:
                        if col not in live_df.columns: 
                            live_df[col] = 0.0

                    latest_features = live_df[FEATURE_COLUMNS].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0)

                    with state_lock:
                        current_model = scalp_model_5m

                    prob = current_model.predict_proba(latest_features)[0][1]
                    current_price = live_df['close'].iloc[-1]

                    # Fix #6: Normalized delta (z-score style) instead of raw -100
                    delta_series = live_df['footprint_delta']
                    delta_mean = delta_series.rolling(50).mean().iloc[-1]
                    delta_std = delta_series.rolling(50).std().iloc[-1]
                    delta_std = max(delta_std, 1e-8)
                    delta_z = (delta_series.iloc[-1] - delta_mean) / delta_std

                    pd_zone_val = live_df['premium_discount_zone'].iloc[-1]
                    current_atr = live_df['atr'].iloc[-1]

                    # Fix #3: ATR-based target prices (2x ATR) instead of 30-candle high/low
                    est_side = "BUY" if htf_trend == 1 else "SELL"

                    with state_lock:
                        state_msg = "SEARCHING FOR ENTRY..."

                        # Calculate target only when entering new trade
                        if active_trade == "BUY":
                            tot_dist = active_target_price - active_entry_price
                            curr_dist = current_price - active_entry_price
                            progress = (curr_dist / tot_dist) if tot_dist > 0 else 0

                            if progress >= 1.0 or current_price >= active_target_price:
                                state_msg = "🎉 100% FULL TARGET ACHIEVED!"
                                active_trade = None
                            # Fix #6: Normalized exit (delta_z < -2.0 OR prob < 0.35)
                            elif delta_z < -2.0 or prob < 0.35 or htf_trend == -1:
                                state_msg = "🚨 EXIT NOW / INVALIDATED!"
                                active_trade = None
                            elif progress >= 0.80: 
                                state_msg = "⚠️ TAKE PARTIAL PROFIT (80% Reached)!"
                            elif progress >= 0.50: 
                                state_msg = "🎯 50% TARGET ACHIEVED!"
                            elif progress >= 0.25: 
                                state_msg = "🎯 25% TARGET ACHIEVED!"
                            else: 
                                state_msg = f"🟢 BUY ACTIVE ({progress*100:.1f}%)"

                        elif active_trade == "SELL":
                            tot_dist = active_entry_price - active_target_price
                            curr_dist = active_entry_price - current_price
                            progress = (curr_dist / tot_dist) if tot_dist > 0 else 0

                            if progress >= 1.0 or current_price <= active_target_price:
                                state_msg = "🎉 100% FULL TARGET ACHIEVED!"
                                active_trade = None
                            # Fix #6: Normalized exit (delta_z > +2.0 OR prob > 0.65)
                            elif delta_z > 2.0 or prob > 0.65 or htf_trend == 1:
                                state_msg = "🚨 EXIT NOW / INVALIDATED!"
                                active_trade = None
                            elif progress >= 0.80: 
                                state_msg = "⚠️ TAKE PARTIAL PROFIT (80% Reached)!"
                            elif progress >= 0.50: 
                                state_msg = "🎯 50% TARGET ACHIEVED!"
                            elif progress >= 0.25: 
                                state_msg = "🎯 25% TARGET ACHIEVED!"
                            else: 
                                state_msg = f"🔴 SELL ACTIVE ({progress*100:.1f}%)"

                        # Entry Logic - Fix #3: ATR-based targets
                        if prob > 0.65 and pd_zone_val < 0.50 and htf_trend == 1 and active_trade is None:
                            active_trade = "BUY"
                            active_entry_price = current_price
                            active_target_price = current_price + (2.0 * current_atr)  # 2x ATR target
                            print(f"🔥 BUY TRIGGERED @ ${current_price:.2f} | Target: ${active_target_price:.2f}", flush=True)

                        elif prob < 0.35 and pd_zone_val > 0.50 and htf_trend == -1 and active_trade is None:
                            active_trade = "SELL"
                            active_entry_price = current_price
                            active_target_price = current_price - (2.0 * current_atr)  # 2x ATR target
                            print(f"🔻 SELL TRIGGERED @ ${current_price:.2f} | Target: ${active_target_price:.2f}", flush=True)

                        latest_bot_state = (
                            f"BTC: ${current_price:.2f} | "
                            f"Target: ${active_target_price:.2f if active_trade else 0:.2f} | "
                            f"Prob: {prob:.2f} | DeltaZ: {delta_z:.2f} | "
                            f"State: {state_msg}"
                        )

                    print(f"[LIVE LOG]: {latest_bot_state}", flush=True)

            time.sleep(5)  # Faster loop for more responsive exits
        except Exception as e:
            print(f"Loop Exception: {e}", flush=True)
            time.sleep(5)

@app.on_event("startup")
def startup_event():
    print("🚀 App Initialized! Starting Quant Background Process...", flush=True)
    threading.Thread(target=quant_execution_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
