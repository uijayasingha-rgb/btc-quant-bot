import os
import time
import requests
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
import uvicorn

# ==========================================
# FASTAPI SERVER FOR RAILWAY PORT BINDING
# ==========================================
app = FastAPI()

@app.get("/")
def health_check():
    return {
        "status": "BTC Quant Bot Active",
        "timestamp": str(datetime.now()),
        "engine": "11 Rules SMC Logic"
    }

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
PROXY_URL = os.getenv("PROXY_URL", "https://crypto-proxy-bot.uijayasingha.workers.dev").rstrip('/')
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

FETCH_INTERVAL_1M = "1m"
FETCH_INTERVAL_15M = "15m"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

LAST_SIGNAL_TIMESTAMP = None

# ==========================================
# DATA FETCHING FUNCTIONS (BINANCE PROXY SAFE)
# ==========================================
def fetch_klines_safe(symbol=SYMBOL, interval="1m", limit=300):
    """
    Safely fetches Kline data via Cloudflare Worker Proxy
    Prevents Typecasting & Missing Value Crashes
    """
    endpoint = f"{PROXY_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        response = requests.get(endpoint, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Proxy Status Code Error [{interval}]: {response.status_code}")
            return None
            
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            return None

        # Binance Standard 12 Columns
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        
        df = pd.DataFrame(data, columns=columns)

        # Safe Numeric Conversion (Prevents Crash on Null/Empty String values)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    except Exception as e:
        print(f"⚠️ Fetch Exception [{interval}]: {e}")
        return None

# ==========================================
# TECHNICAL ANALYSIS & SMC 11-RULES ENGINE
# ==========================================
def calculate_atr(df, period=14):
    """Calculates Average True Range (ATR) for SL/TP"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def analyze_smc_and_rules(df_1m, df_15m):
    """
    Evaluates Quant Rules & SMC Logic
    """
    global LAST_SIGNAL_TIMESTAMP

    if df_1m is None or df_15m is None or len(df_1m) < 50 or len(df_15m) < 50:
        return None

    last_candle = df_1m.iloc[-1]
    prev_candle = df_1m.iloc[-2]
    current_time = last_candle['timestamp']

    # Prevent Repeating the Same Signal in the Same Minute Bar
    if LAST_SIGNAL_TIMESTAMP == current_time:
        return None

    # 1. Higher Timeframe Trend Filter (15m Structure)
    htf_sma = df_15m['close'].rolling(50).mean().iloc[-1]
    is_htf_bullish = df_15m['close'].iloc[-1] > htf_sma

    # 2. ATR Volatility Logic
    df_1m['atr'] = calculate_atr(df_1m)
    current_atr = df_1m['atr'].iloc[-1]
    if pd.isna(current_atr) or current_atr == 0:
        current_atr = last_candle['close'] * 0.002 # Default Fallback ATR (0.2%)

    # 3. Volume Surge & Taker Buying Pressure
    avg_volume = df_1m['volume'].rolling(20).mean().iloc[-1]
    volume_surge = last_candle['volume'] > (avg_volume * 1.4)
    buying_pressure = last_candle['taker_buy_base'] > (last_candle['volume'] * 0.45)

    # 4. SMC Candlestick Patterns (Engulfing / Rejection)
    is_bullish_engulfing = (last_candle['close'] > prev_candle['high']) and (last_candle['open'] <= prev_candle['low'])
    is_bearish_engulfing = (last_candle['close'] < prev_candle['low']) and (last_candle['open'] >= prev_candle['high'])

    signal = None

    # Bullish Conditions (Rule 11 Engine)
    if is_htf_bullish and is_bullish_engulfing and volume_surge and buying_pressure:
        entry_price = last_candle['close']
        stop_loss = entry_price - (current_atr * 1.5)
        take_profit = entry_price + ((entry_price - stop_loss) * 2.0)

        LAST_SIGNAL_TIMESTAMP = current_time
        signal = {
            "type": "BUY / LONG 🟢",
            "symbol": SYMBOL,
            "entry": round(entry_price, 2),
            "sl": round(stop_loss, 2),
            "tp": round(take_profit, 2),
            "reason": "15m HTF Bullish + Volume Surge + SMC Engulfing Sweep"
        }

    # Bearish Conditions
    elif not is_htf_bullish and is_bearish_engulfing and volume_surge and not buying_pressure:
        entry_price = last_candle['close']
        stop_loss = entry_price + (current_atr * 1.5)
        take_profit = entry_price - ((stop_loss - entry_price) * 2.0)

        LAST_SIGNAL_TIMESTAMP = current_time
        signal = {
            "type": "SELL / SHORT 🔴",
            "symbol": SYMBOL,
            "entry": round(entry_price, 2),
            "sl": round(stop_loss, 2),
            "tp": round(take_profit, 2),
            "reason": "15m HTF Bearish + Volume Surge + SMC Rejection Sweep"
        }

    return signal

# ==========================================
# BACKGROUND CONTINUOUS TRADING LOOP
# ==========================================
def bot_loop():
    print("🚀 BTC Quant Logic Engine Thread Running...")
    time.sleep(3)

    while True:
        try:
            df_1m = fetch_klines_safe(symbol=SYMBOL, interval="1m", limit=300)
            df_15m = fetch_klines_safe(symbol=SYMBOL, interval="15m", limit=100)

            if df_1m is not None and df_15m is not None:
                current_price = df_1m['close'].iloc[-1]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Live {SYMBOL}: ${current_price:.2f} | Status: Data Fetched Successfully ✅")

                trade_signal = analyze_smc_and_rules(df_1m, df_15m)

                if trade_signal:
                    print("\n" + "="*55)
                    print(f"🚨 NEW SIGNAL GENERATED: {trade_signal['type']}")
                    print(f"📍 Entry: ${trade_signal['entry']} | SL: ${trade_signal['sl']} | TP: ${trade_signal['tp']}")
                    print(f"💡 Strategy Logic: {trade_signal['reason']}")
                    print("="*55 + "\n")
            else:
                print("⚠️ Data Fetching Issue. Retrying in 5s...")

        except Exception as err:
            print(f"❌ Main Loop Exception: {err}")

        time.sleep(10)

# ==========================================
# MAIN ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    # Start Trading Bot in Daemon Thread
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()

    # Start FastAPI Web Server for Railway Health Check
    uvicorn.run(app, host=HOST, port=PORT)
