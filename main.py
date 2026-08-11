import os, time, requests, threading, numpy as np, pandas as pd, uvicorn
from xgboost import XGBClassifier
from fastapi import FastAPI
from datetime import datetime

app = FastAPI()
PROXY_URL = "https://crypto-proxy-bot.uijayasingha.workers.dev"

state_lock = threading.Lock()
active_trade = None
active_entry = 0.0
active_sl = 0.0
active_tp = 0.0
latest_state = "INITIALIZING..."

@app.get("/")
def home():
    with state_lock:
        return {"status": "SMC BOT", "state": latest_state, "trade": active_trade}

def fetch_data(symbol="BTCUSDT", interval="5m", limit=200):
    try:
        r = requests.get(f"{PROXY_URL}/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=['ot','o','h','l','c','v','ct','qv','tr','tb','tq','ig'])
                for col in ['o','h','l','c','v']: df[col] = df[col].astype(float)
                df['tbv'] = df['tb'].astype(float)
                df['ot'] = pd.to_datetime(df['ot'].astype(float), unit='ms', utc=True)
                df.columns = ['open_time','open','high','low','close','volume','close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore','taker_buy_base_vol']
                return df
    except Exception as e: print(f"Proxy err: {e}", flush=True)
    return pd.DataFrame()

def build_features(df):
    df = df.copy()
    df['delta'] = df['taker_buy_base_vol'] - (df['volume'] - df['taker_buy_base_vol'])
    df['cvd'] = df['delta'].cumsum()
    df['cvd_div'] = df['cvd'].pct_change().fillna(0) - df['close'].pct_change().fillna(0)
    cr = np.maximum(df['high'] - df['low'], 1e-8)
    df['wick'] = ((df['high'] - np.maximum(df['open'],df['close'])) - (np.minimum(df['open'],df['close']) - df['low'])) / cr
    df['liq'] = np.where(df['high'] > df['high'].rolling(20).max().shift(1), 1, np.where(df['low'] < df['low'].rolling(20).min().shift(1), -1, 0))
    df['fvg'] = (np.maximum(0, df['low'] - df['high'].shift(2)) - np.maximum(0, df['low'].shift(2) - df['high'])) / df['close']
    df['atr'] = cr.rolling(14).mean()
    df['vsq'] = np.where(df['atr'].rolling(100).rank(pct=True).fillna(0.5) < 0.20, 1, 0)
    vm = df['volume'].rolling(20).mean()
    df['vsa'] = (df['volume']/np.maximum(vm,1e-8)) / (cr/df['close'])
    vw = 48; pv = df['close']*df['volume']; rp = pv.rolling(vw).sum(); rv = df['volume'].rolling(vw).sum()
    vwap = rp/np.maximum(rv,1e-8); vstd = df['close'].rolling(vw).std().fillna(1.0)
    df['vwap_dev'] = (df['close']-vwap)/np.maximum(vstd,1e-8)
    rh = df['high'].rolling(30).max(); rl = df['low'].rolling(30).min()
    df['pd'] = (df['close']-rl)/np.maximum(rh-rl,1e-8)
    hr = df['open_time'].dt.hour
    df['kz'] = np.where((hr>=8)&(hr<=11),1,np.where((hr>=13)&(hr<=16),2,0))
    return df

def train_model():
    print("Training...", flush=True)
    df = fetch_data(limit=400)
    if df.empty: return None
    df = build_features(df)
    fr = (df['close'].shift(-3)-df['close'])/df['close']
    df['T'] = np.where(fr>0.003,1,0)
    df = df.replace([np.inf,-np.inf],np.nan).fillna(0)
    X = df[['delta','cvd_div','wick','liq','fvg','vsq','vsa','vwap_dev','kz','pd']]
    y = df['T']
    m = XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.04,random_state=42,
                      scale_pos_weight=(len(y)-y.sum())/max(y.sum(),1),eval_metric='logloss')
    m.fit(X,y)
    print(f"Trained: Pos={y.sum()}, Neg={len(y)-y.sum()}", flush=True)
    return m

def loop():
    global active_trade, active_entry, active_sl, active_tp, latest_state
    model = train_model()
    while True:
        try:
            df = fetch_data(limit=100)
            if df.empty or len(df)<30: time.sleep(10); continue
            df = build_features(df)
            f = df[['delta','cvd_div','wick','liq','fvg','vsq','vsa','vwap_dev','kz','pd']].iloc[-1:].fillna(0)
            prob = model.predict_proba(f)[0][1] if model else 0.5
            price = df['close'].iloc[-1]
            pdz = df['pd'].iloc[-1]
            
            with state_lock:
                if active_trade == "BUY":
                    prog = (price-active_entry)/(active_tp-active_entry) if active_tp>active_entry else 0
                    if prog>=1 or price>=active_tp: latest_state="TARGET HIT"; active_trade=None
                    elif price<=active_sl or prob<0.35: latest_state="STOPPED"; active_trade=None
                    elif prog>=0.8: latest_state=f"BUY 80%"
                    elif prog>=0.5: latest_state=f"BUY 50%"
                    else: latest_state=f"BUY {prog*100:.0f}%"
                elif active_trade == "SELL":
                    prog = (active_entry-price)/(active_entry-active_tp) if active_entry>active_tp else 0
                    if prog>=1 or price<=active_tp: latest_state="TARGET HIT"; active_trade=None
                    elif price>=active_sl or prob>0.65: latest_state="STOPPED"; active_trade=None
                    elif prog>=0.8: latest_state=f"SELL 80%"
                    elif prog>=0.5: latest_state=f"SELL 50%"
                    else: latest_state=f"SELL {prog*100:.0f}%"
                else:
                    if prob>0.65 and pdz<0.35:
                        active_trade="BUY"; active_entry=price
                        active_sl=df['low'].tail(5).min()*0.998
                        active_tp=price+(price-active_sl)*2
                        latest_state=f"BUY @{price:.0f} SL:{active_sl:.0f} TP:{active_tp:.0f}"
                        print(f"FIRE: {latest_state}", flush=True)
                    elif prob<0.35 and pdz>0.65:
                        active_trade="SELL"; active_entry=price
                        active_sl=df['high'].tail(5).max()*1.002
                        active_tp=price-(active_sl-price)*2
                        latest_state=f"SELL @{price:.0f} SL:{active_sl:.0f} TP:{active_tp:.0f}"
                        print(f"FIRE: {latest_state}", flush=True)
                    else:
                        latest_state=f"SCAN BTC:{price:.0f} P:{prob:.2f} PD:{pdz:.2f}"
                print(f"[LIVE] {latest_state}", flush=True)
            time.sleep(10)
        except Exception as e:
            print(f"ERR: {e}", flush=True)
            time.sleep(10)

@app.on_event("startup")
def start():
    print("SMC BOT STARTING...", flush=True)
    threading.Thread(target=loop, daemon=True).start()

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
