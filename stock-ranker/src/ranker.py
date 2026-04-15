import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from datetime import datetime

class ListNet(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.scorer(x).squeeze(1)

class StockRanker:
    FEATURE_COLS = ['mom_5d', 'mom_10d', 'mom_20d', 'vol_10d', 'vol_20d',
                    'rsi_dist', 'high_20d_ratio', 'vol_ratio']

    def __init__(self, model_path, scaler_path, tickers):
        self.tickers = tickers

        # 加载模型
        ckpt = torch.load(model_path, map_location="cpu")
        self.model = ListNet(input_dim=8)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # 加载 scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        print(f"[StockRanker] 加载完成，股票池: {len(tickers)} 只")

    def _fetch_features(self):
        """拉取最新数据并计算特征，返回最新一天的特征 DataFrame"""
        raw = yf.download(self.tickers, period="2mo", interval="1d",
                          auto_adjust=True, progress=False)
        close  = raw["Close"]
        volume = raw["Volume"]

        all_feats = []
        for ticker in self.tickers:
            f = pd.DataFrame(index=close.index)
            f["mom_5d"]         = close[ticker].pct_change(5)
            f["mom_10d"]        = close[ticker].pct_change(10)
            f["mom_20d"]        = close[ticker].pct_change(20)
            f["vol_10d"]        = close[ticker].pct_change().rolling(10).std()
            f["vol_20d"]        = close[ticker].pct_change().rolling(20).std()
            f["rsi_dist"]       = (close[ticker] - close[ticker].rolling(20).mean()) / close[ticker].rolling(20).std()
            f["high_20d_ratio"] = close[ticker] / close[ticker].rolling(20).max()
            f["vol_ratio"]      = volume[ticker] / volume[ticker].rolling(10).mean()
            f["ticker"]         = ticker
            all_feats.append(f)

        df = pd.concat(all_feats).dropna()
        # 取最新一天
        latest_date = df.index.max()
        latest = df.loc[latest_date].reset_index(drop=True)
        return latest, latest_date

    def run(self, top_k=5):
        """执行一次推理，返回 watch list"""
        latest, latest_date = self._fetch_features()

        # 标准化
        feats_scaled = self.scaler.transform(latest[self.FEATURE_COLS].values)
        feats_tensor = torch.tensor(feats_scaled, dtype=torch.float32)

        # 打分排序
        with torch.no_grad():
            scores = self.model(feats_tensor).numpy()

        latest["score"] = scores
        watch_list = (latest[["ticker", "score"]]
                      .sort_values("score", ascending=False)
                      .head(top_k)
                      .reset_index(drop=True))
        watch_list.index += 1  # rank 从1开始

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Watch List ({latest_date.date()})")
        print(watch_list.to_string())
        return watch_list, latest_date
