
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch, torch.nn as nn
import onnxruntime as ort
import pickle, json, os, glob
from datetime import datetime

# ── 常量 ──────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/ml-projects/stock-ranker/data")
OUTPUT_DIR = os.path.expanduser("~/ml-projects/stock-ranker/output")
FEATURE_COLS = ['mom_5d','mom_10d','mom_20d','vol_10d','vol_20d',
                'rsi_dist','high_20d_ratio','vol_ratio']
DEFAULT_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA",
    "META","TSLA","AMD","INTC","CRM",
    "JPM","BAC","GS","MS","WFC",
    "JNJ","UNH","PFE","MRK","ABBV"
]

# ── 加载资源（缓存）──────────────────────────────────
@st.cache_resource
def load_resources():
    sess = ort.InferenceSession(f"{DATA_DIR}/listnet_v2.onnx")
    with open(f"{DATA_DIR}/scaler_v2.pkl", "rb") as f:
        scaler = pickle.load(f)
    return sess, scaler

# ── 特征工程 ─────────────────────────────────────────
def make_features(close, volume):
    f = pd.DataFrame(index=close.index)
    f["mom_5d"]         = close.pct_change(5)
    f["mom_10d"]        = close.pct_change(10)
    f["mom_20d"]        = close.pct_change(20)
    f["vol_10d"]        = close.pct_change().rolling(10).std()
    f["vol_20d"]        = close.pct_change().rolling(20).std()
    f["rsi_dist"]       = (close - close.rolling(20).mean()) / close.rolling(20).std()
    f["high_20d_ratio"] = close / close.rolling(20).max()
    f["vol_ratio"]      = volume / volume.rolling(10).mean()
    return f

def get_watch_list(tickers, sess, scaler, top_k=5):
    raw    = yf.download(tickers, period="2mo", interval="1d",
                         auto_adjust=True, progress=False)
    close  = raw["Close"]
    volume = raw["Volume"]

    rows = []
    for t in tickers:
        feat = make_features(close[t], volume[t]).dropna()
        if len(feat) == 0: continue
        latest = feat.iloc[[-1]]
        latest["ticker"] = t
        rows.append(latest)

    df = pd.concat(rows)
    latest_date = df.index.max()
    feats_scaled = scaler.transform(df[FEATURE_COLS].values)
    scores = sess.run(["scores"], {"features": feats_scaled.astype(np.float32)})[0]
    df["score"] = scores
    result = (df[["ticker","score"]]
              .sort_values("score", ascending=False)
              .head(top_k)
              .reset_index(drop=True))
    result.index += 1
    return result, latest_date

# ── UI ───────────────────────────────────────────────
st.set_page_config(page_title="Stock Ranker", page_icon="📈", layout="wide")
st.title("📈 Stock Watch List")
st.caption("Learning-to-Rank · ListNet · yfinance")

sess, scaler = load_resources()

with st.sidebar:
    st.header("设置")
    ticker_input = st.text_area(
        "股票池（每行一个）",
        value="\n".join(DEFAULT_TICKERS), height=300
    )
    top_k  = st.slider("Watch list 数量", 3, 10, 5)
    run_btn = st.button("🔄 立即运行", use_container_width=True)

# ── 主区域 ───────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("今日 Watch List")
    if run_btn:
        tickers = [t.strip().upper() for t in ticker_input.split("\n") if t.strip()]
        with st.spinner("拉取数据 & 推理中..."):
            watch_list, date = get_watch_list(tickers, sess, scaler, top_k)
        st.success(f"数据日期：{date.date()}")
        st.dataframe(watch_list, use_container_width=True)

        # 保存 JSON
        out = {
            "date": str(date.date()),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "watch_list": watch_list.rename(
                columns={"score":"model_score"}).to_dict(orient="records")
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(f"{OUTPUT_DIR}/{date.date()}.json", "w") as f:
            json.dump(out, f, indent=2)
    else:
        st.info("点击左侧「立即运行」生成 watch list")

with col2:
    st.subheader("历史记录")
    files = sorted(glob.glob(f"{OUTPUT_DIR}/*.json"), reverse=True)
    if files:
        for fp in files[:7]:
            with open(fp) as f:
                rec = json.load(f)
            with st.expander(f"📅 {rec['date']}  （生成于 {rec['generated_at']}）"):
                st.dataframe(pd.DataFrame(rec["watch_list"]), use_container_width=True)
    else:
        st.info("暂无历史记录")
