import os
import time
import requests
import threading
import numpy as np
import pandas as pd
import uvicorn
from xgboost import XGBClassifier
from fastapi import FastAPI

app = FastAPI()

FEATURE_COLUMNS = [
    'footprint_delta', 'cvd_divergence', 'wick_absorption', 'liquidity_sweep',
    'fvg_imbalance', 'volatility_squeeze', 'vsa_anomaly', 'anchored_vwap_dev',
    'session_killzone', 'premium_discount_zone'
]

# Global Active Trade State
active_trade = None
active_entry_price = 0.0
active_target_price = 0.0
latest_bot_state = "INITIALIZING ENGINE & TRAINING MODEL..."
scalp_model_5m = None

@app.get("/")
def home():
    return {
        "status": "BTC Institutional Quant Engine Active",
        "current_state": latest_bot_state
    }

@app.post("/webhook")
def receive_signal(data: dict):
    print(f"📥 [WEBHOOK RECEIVED]: {data}", flush=True)
    return {"status": "success"}

def fetch_crypto_quant_data(symbol="BTC", interval="5", limit=300):
    url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={symbol}&tsym=USD&limit={limit}&aggregate={interval}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            raw_list = data.get("Data", {}).get("Data", [])
            if raw_list and len(raw_list) > 0:
                df = pd.DataFrame(raw_list)
                df = df.rename(columns={
                    'time': 'open_time',
                    'volumeto': 'volume',
                    'volumefrom': 'taker_buy_base_vol'
                })
                cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_vol']
                df[cols] = df[cols].astype(float)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
                return df
    except Exception as e:
        print(f"Fetch Exception: {e}", flush=True)
        
    return pd.DataFrame()

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
    
    df['liquidity_sweep'] = np.where(df['high'] > df['high'].rolling(20).max().shift(1), 1,
                             np.where(df['low'] < df['low'].rolling(20).min().shift(1), -1, 0))
                             
    df['fvg_bullish'] = np.maximum(0, df['low'] - df['high'].shift(2))
    df['fvg_bearish'] = np.maximum(0, df['low'].shift(2) - df['high'])
    df['fvg_imbalance'] = (df['fvg_bullish'] - df['fvg_bearish']) / df['close']
    
    atr = candle_range.rolling(14).mean()
    atr_percentile = atr.rolling(100).rank(pct=True).fillna(0.5)
    df['volatility_squeeze'] = np.where(atr_percentile < 0.20, 1, 0)
    
    vol_mean = df['volume'].rolling(20).mean()
    df['vsa_anomaly'] = (df['volume'] / np.maximum(vol_mean, 1e-8)) / (candle_range / df['close'])
    
    pv = df['close'] * df['volume']
    cum_pv = pv.cumsum()
    cum_vol = df['volume'].cumsum()
    vwap = cum_pv / np.maximum(cum_vol, 1e-8)
    vwap_std = df['close'].rolling(20).std().fillna(1.0)
    df['anchored_vwap_dev'] = (df['close'] - vwap) / np.maximum(vwap_std, 1e-8)
    
    range_high = df['high'].rolling(30).max()
    range_low = df['low'].rolling(30).min()
    price_range = np.maximum(range_high - range_low, 1e-8)
    df['premium_discount_zone'] = (df['close'] - range_low) / price_range
    
    hours = df['open_time'].dt.hour
    df['session_killzone'] = np.where((hours >= 7) & (hours <= 10), 1,
                             np.where((hours >= 13) & (hours <= 16), 2, 0))
    return df

def get_15m_htf_structure():
    df_15m = fetch_crypto_quant_data(interval="15", limit=100)
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

def train_5m_quant_model():
    print("⏳ Training AI Model in Background...", flush=True)
    df = pd.DataFrame()
    attempts = 0
    while True:
        attempts += 1
        raw_df = fetch_crypto_quant_data(interval="5", limit=300)
        if not raw_df.empty and 'close' in raw_df.columns:
            df = build_institutional_features(raw_df)
            if not df.empty and 'close' in df.columns and len(df) >= 30:
                print("✅ Crypto Market Data Received Successfully!", flush=True)
                break
        print(f"⚠️ Data Fetching Issue. Retrying in 5s... (Attempt {attempts})", flush=True)
        time.sleep(5)

    for col in FEATURE_COLUMNS:
        if col not in df.columns: 
            df[col] = 0.0
            
    future_return = (df['close'].shift(-3) - df['close']) / df['close']
    df['Target'] = np.where(future_return > 0.003, 1, 0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df[FEATURE_COLUMNS]
    y = df['Target']
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.04, random_state=42)
    model.fit(X, y)
    print("✅ AI Model Trained Successfully!", flush=True)
    return model

def quant_execution_loop():
    global active_trade, active_entry_price, active_target_price, latest_bot_state, scalp_model_5m
    
    scalp_model_5m = train_5m_quant_model()
    
    while True:
        try:
            htf_trend, htf_pd_zone = get_15m_htf_structure()
            live_df = fetch_crypto_quant_data(interval="5", limit=100)
            if not live_df.empty and 'close' in live_df.columns and len(live_df) >= 30:
                live_df = build_institutional_features(live_df)
                if not live_df.empty and 'close' in live_df.columns:
                    for col in FEATURE_COLUMNS:
                        if col not in live_df.columns: 
                            live_df[col] = 0.0
                    
                    latest_features = live_df[FEATURE_COLUMNS].iloc[-1:].replace([np.inf, -np.inf], np.nan).fillna(0)
                    prob = scalp_model_5m.predict_proba(latest_features)[0][1]
                    current_price = live_df['close'].iloc[-1]
                    delta = live_df['footprint_delta'].iloc[-1]
                    pd_zone_val = live_df['premium_discount_zone'].iloc[-1]
                    
                    est_side = "BUY" if htf_trend == 1 else "SELL"
                    t_price = live_df['high'].rolling(30).max().iloc[-1] if est_side == "BUY" else live_df['low'].rolling(30).min().iloc[-1]
                    
                    state_msg = "SEARCHING FOR ENTRY..."
                    if active_trade == "BUY":
                        tot_dist = active_target_price - active_entry_price
                        curr_dist = current_price - active_entry_price
                        progress = (curr_dist / tot_dist) if tot_dist > 0 else 0
                        if progress >= 1.0 or current_price >= active_target_price:
                            state_msg = "🎉 100% FULL TARGET ACHIEVED!"; active_trade = None
                        elif delta < -100 or prob < 0.40 or htf_trend == -1:
                            state_msg = "🚨 EXIT NOW / INVALIDATED!"; active_trade = None
                        elif progress >= 0.80: state_msg = "⚠️ TAKE PARTIAL PROFIT (80% Reached)!"
                        elif progress >= 0.50: state_msg = "🎯 50% TARGET ACHIEVED!"
                        elif progress >= 0.25: state_msg = "🎯 25% TARGET ACHIEVED!"
                        else: state_msg = f"🟢 BUY ACTIVE ({progress*100:.1f}%)"
                    elif active_trade == "SELL":
                        tot_dist = active_entry_price - active_target_price
                        curr_dist = active_entry_price - current_price
                        progress = (curr_dist / tot_dist) if tot_dist > 0 else 0
                        if progress >= 1.0 or current_price <= active_target_price:
                            state_msg = "🎉 100% FULL TARGET ACHIEVED!"; active_trade = None
                        elif delta > 100 or prob > 0.60 or htf_trend == 1:
                            state_msg = "🚨 EXIT NOW / INVALIDATED!"; active_trade = None
                        elif progress >= 0.80: state_msg = "⚠️ TAKE PARTIAL PROFIT (80% Reached)!"
                        elif progress >= 0.50: state_msg = "🎯 50% TARGET ACHIEVED!"
                        elif progress >= 0.25: state_msg = "🎯 25% TARGET ACHIEVED!"
                        else: state_msg = f"🔴 SELL ACTIVE ({progress*100:.1f}%)"

                    if prob > 0.65 and pd_zone_val < 0.50 and htf_trend == 1 and active_trade != "BUY":
                        active_trade, active_entry_price, active_target_price = "BUY", current_price, t_price
                        print(f"🔥 BUY TRIGGERED @ ${current_price}", flush=True)
                    elif prob < 0.35 and pd_zone_val > 0.50 and htf_trend == -1 and active_trade != "SELL":
                        active_trade, active_entry_price, active_target_price = "SELL", current_price, t_price
                        print(f"🔻 SELL TRIGGERED @ ${current_price}", flush=True)

                    latest_bot_state = f"BTC: ${current_price:.2f} | Target: ${t_price:.2f} | State: {state_msg}"
                    print(f"[LIVE LOG]: {latest_bot_state}", flush=True)

            time.sleep(10)
        except Exception as e:
            print(f"Loop Exception: {e}", flush=True)
            time.sleep(5)

# Safe Startup Event Trigger
@app.on_event("startup")
def startup_event():
    print("🚀 App Initialized Successfully! Starting Quant Background Process...", flush=True)
    threading.Thread(target=quant_execution_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
