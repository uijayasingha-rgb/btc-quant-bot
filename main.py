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
PROXY_URL = os.getenv("PROXY_URL", "https://crypto-proxy-bot.uijayasingha.workers.dev").rstrip('/')
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

LAST_SIGNAL_TIMESTAMP = None

# ==========================================
# DATA FETCHING FUNCTIONS (BINANCE PROXY)
# ==========================================
def fetch_klines_safe(symbol=SYMBOL, interval="1m", limit=300):
    """Fetches Kline data via Cloudflare Worker Proxy safely"""
    endpoint = f"{PROXY_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    
    try:
        response = requests.get(endpoint, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Proxy Status Error [{interval}]: {response.status_code}", flush=True)
            return None
            
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            return None

        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        
        df = pd.DataFrame(data, columns=columns)

        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    except Exception as e:
        print(f"⚠️ Fetch Exception [{interval}]: {e}", flush=True)
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
    volume_surge = last_candle['volume'] > (avg_volume * 1.4)
    buying_pressure = last_candle['taker_buy_base'] > (last_candle['volume'] * 0.45)

    # 4. Patterns
    is_bullish_engulfing = (last_candle['close'] > prev_candle['high']) and (last_candle['open'] <= prev_candle['low'])
    is_bearish_engulfing = (last_candle['close'] < prev_candle['low']) and (last_candle['open'] >= prev_candle['high'])

    signal = None

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
# TRADING BOT CONTINUOUS THREAD LOOP
# ==========================================
def run_quant_bot():
    print("🚀 BTC Quant Engine Worker Thread Started!", flush=True)
    time.sleep(3)

    while True:
        try:
            df_1m = fetch_klines_safe(symbol=SYMBOL, interval="1m", limit=300)
            df_15m = fetch_klines_safe(symbol=SYMBOL, interval="15m", limit=100)

            if df_1m is not None and df_15m is not None:
                current_price = df_1m['close'].iloc[-1]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Live {SYMBOL}: ${current_price:.2f} | Status: Data Fetched Successfully ✅", flush=True)

                trade_signal = analyze_smc_and_rules(df_1m, df_15m)

                if trade_signal:
                    print("\n" + "="*55, flush=True)
                    print(f"🚨 NEW SIGNAL GENERATED: {trade_signal['type']}", flush=True)
                    print(f"📍 Entry: ${trade_signal['entry']} | SL: ${trade_signal['sl']} | TP: ${trade_signal['tp']}", flush=True)
                    print(f"💡 Strategy Logic: {trade_signal['reason']}", flush=True)
                    print("="*55 + "\n", flush=True)
            else:
                print("⚠️ Data Fetching Issue. Retrying in 10s...", flush=True)

        except Exception as err:
            print(f"❌ Worker Thread Exception: {err}", flush=True)

        time.sleep(10)

# ==========================================
# DIRECT THREAD INITIALIZATION (MODULE LOAD)
# ==========================================
# Python File එක Import/Run වෙන පළමු තත්පරයේදීම Bot එක Background එකේ වැඩ පටන් ගන්නවා
bot_thread = threading.Thread(target=run_quant_bot, daemon=True)
bot_thread.start()

# ==========================================
# FASTAPI APP FOR RAILWAY HEALTH CHECK
# ==========================================
app = FastAPI()

@app.get("/")
def health_check():
    return {
        "status": "BTC Quant Bot Active",
        "timestamp": str(datetime.now())
    }

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
