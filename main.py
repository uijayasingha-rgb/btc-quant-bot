import os
import time
import threading
import requests
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
import uvicorn

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# REAL BROWSER USER-AGENT HEADERS (TO BYPASS 403/451 BLOCKS)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache"
}

LAST_SIGNAL_TIMESTAMP = None

# REAL-TIME BOT STATUS MONITORING MEMORY
bot_state = {
    "status": "Initializing...",
    "symbol": SYMBOL,
    "last_price": 0.0,
    "last_update": "N/A",
    "total_checks": 0,
    "last_signal": "No Signal Yet",
    "data_source": "Not Connected"
}

# ==========================================
# DATA FETCHING FUNCTIONS (BYBIT / BINANCE SAFE)
# ==========================================
def fetch_klines_safe(symbol=SYMBOL, interval="1", limit=200):
    """Fetches Kline Data using Bybit API (No Geo-Block & 100% Reliable for Cloud)"""
    
    # Bybit Interval Format Mapping
    bybit_interval = "1" if interval == "1m" or interval == "1" else "15"
    
    url = f"https://api.bybit.com/v5/market/kline?category=spot&symbol={symbol}&interval={bybit_interval}&limit={limit}"
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code != 200:
            print(f"⚠️ API Error [{interval}]: Status Code {response.status_code}", flush=True)
            bot_state["data_source"] = f"Error HTTP {response.status_code}"
            return None
            
        data = response.json()
        
        if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
            bot_state["data_source"] = "Invalid Response Data"
            return None

        # Bybit returns data in reverse chronological order: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        raw_klines = data["result"]["list"]
        raw_klines.reverse() # Convert to chronological order

        df = pd.DataFrame(raw_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
        df['taker_buy_base'] = df['volume'] * 0.5  # Estimated volume flow

        bot_state["data_source"] = "Bybit Public API Connected ✅"
        return df

    except Exception as e:
        print(f"⚠️ Fetch Exception [{interval}]: {e}", flush=True)
        bot_state["data_source"] = f"Exception: {str(e)}"
        return None

# ==========================================
# TECHNICAL ANALYSIS & SMC ENGINE
# ==========================================
def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def analyze_smc_and_rules(df_1m, df_15m):
    global LAST_SIGNAL_TIMESTAMP

    if df_1m is None or df_15m is None or len(df_1m) < 50 or len(df_15m) < 50:
        return None

    last_candle = df_1m.iloc[-1]
    prev_candle = df_1m.iloc[-2]
    current_time = last_candle['timestamp']

    if LAST_SIGNAL_TIMESTAMP == current_time:
        return None

    # 1. 15m HTF Trend
    htf_sma = df_15m['close'].rolling(50).mean().iloc[-1]
    is_htf_bullish = df_15m['close'].iloc[-1] > htf_sma

    # 2. ATR Calculation
    df_1m['atr'] = calculate_atr(df_1m)
    current_atr = df_1m['atr'].iloc[-1]
    if pd.isna(current_atr) or current_atr == 0:
        current_atr = last_candle['close'] * 0.002

    # 3. Volume Surge & Order Flow
    avg_volume = df_1m['volume'].rolling(20).mean().iloc[-1]
    volume_surge = last_candle['volume'] > (avg_volume * 1.3)

    # 4. Patterns
    is_bullish_engulfing = (last_candle['close'] > prev_candle['high']) and (last_candle['open'] <= prev_candle['low'])
    is_bearish_engulfing = (last_candle['close'] < prev_candle['low']) and (last_candle['open'] >= prev_candle['high'])

    signal = None

    if is_htf_bullish and is_bullish_engulfing and volume_surge:
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

    elif not is_htf_bullish and is_bearish_engulfing and volume_surge:
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
# TRADING BOT CONTINUOUS THREAD LOOP
# ==========================================
def run_quant_bot():
    print("🚀 BTC Quant Engine Worker Thread Started!", flush=True)
    bot_state["status"] = "Worker Thread Running 🏃"
    time.sleep(3)

    while True:
        try:
            df_1m = fetch_klines_safe(symbol=SYMBOL, interval="1m", limit=200)
            df_15m = fetch_klines_safe(symbol=SYMBOL, interval="15m", limit=100)

            if df_1m is not None and df_15m is not None:
                current_price = float(df_1m['close'].iloc[-1])
                
                # Update Global State for Endpoint Checking
                bot_state["status"] = "Active & Data Fetching ✅"
                bot_state["last_price"] = current_price
                bot_state["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                bot_state["total_checks"] += 1

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Live {SYMBOL}: ${current_price:.2f} | Status: Data Fetched Successfully ✅", flush=True)

                trade_signal = analyze_smc_and_rules(df_1m, df_15m)

                if trade_signal:
                    bot_state["last_signal"] = trade_signal
                    print("\n" + "="*55, flush=True)
                    print(f"🚨 NEW SIGNAL GENERATED: {trade_signal['type']}", flush=True)
                    print(f"📍 Entry: ${trade_signal['entry']} | SL: ${trade_signal['sl']} | TP: ${trade_signal['tp']}", flush=True)
                    print(f"💡 Strategy Logic: {trade_signal['reason']}", flush=True)
                    print("="*55 + "\n", flush=True)
            else:
                bot_state["status"] = "Data Fetching Issue ⚠️"
                print("⚠️ Data Fetching Issue. Retrying in 10s...", flush=True)

        except Exception as err:
            bot_state["status"] = f"Loop Exception Error: {str(err)}"
            print(f"❌ Worker Thread Exception: {err}", flush=True)

        time.sleep(10)

# ==========================================
# DIRECT THREAD INITIALIZATION (MODULE LOAD)
# ==========================================
bot_thread = threading.Thread(target=run_quant_bot, daemon=True)
bot_thread.start()

# ==========================================
# FASTAPI APP FOR HEALTH CHECK & MONITORING
# ==========================================
app = FastAPI(title="BTC Quant Bot Monitor")

@app.get("/")
def health_check():
    """Live Status Endpoint"""
    return {
        "engine_state": bot_state,
        "server_time": str(datetime.now())
    }
@app.get("/signal")
def get_latest_signal():
    """TradingView Connector Endpoint"""
    return {
        "symbol": bot_state["symbol"],
        "price": bot_state["last_price"],
        "signal": bot_state["last_signal"],
        "timestamp": bot_state["last_update"]
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
