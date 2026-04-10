# src/app.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from recommender import Recommender

# ── 路径 ──
BASE = Path(__file__).parent.parent
DATA_DIR    = BASE / "data"
MODEL_PATH  = DATA_DIR / "two_tower_v2_best.pth"
ONNX_PATH   = DATA_DIR / "user_tower.onnx"
NCF_PATH    = DATA_DIR / "ncf_clean.pth"

# ── NCF 定义 ──
class NCF(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=32, mlp_dims=[64,32,16]):
        super().__init__()
        self.mf_user_emb   = nn.Embedding(n_users,  emb_dim)
        self.mf_movie_emb  = nn.Embedding(n_movies, emb_dim)
        self.mlp_user_emb  = nn.Embedding(n_users,  emb_dim)
        self.mlp_movie_emb = nn.Embedding(n_movies, emb_dim)
        layers = []
        in_dim = emb_dim * 2
        for out_dim in mlp_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(emb_dim + mlp_dims[-1], 1)

    def forward(self, user_idx, movie_idx):
        mf = self.mf_user_emb(user_idx) * self.mf_movie_emb(movie_idx)
        mlp_in = torch.cat([self.mlp_user_emb(user_idx),
                            self.mlp_movie_emb(movie_idx)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        out = self.out(torch.cat([mf, mlp_out], dim=-1))
        return torch.sigmoid(out).squeeze(-1)

# ── 加载（cached，只跑一次）──
@st.cache_resource
def load_data():
    ratings = pd.read_csv(DATA_DIR / "ml-1m/ratings.dat", sep="::",
        names=["user_id","movie_id","rating","timestamp"], engine="python")
    movies = pd.read_csv(DATA_DIR / "ml-1m/movies.dat", sep="::",
        names=["movie_id","title","genres"], engine="python", encoding="latin-1")
    movie_info = movies.set_index("movie_id")[["title","genres"]]
    all_movies = sorted(ratings["movie_id"].unique())
    movie2idx  = {m: i for i, m in enumerate(all_movies)}
    idx2movie  = {i: m for m, i in movie2idx.items()}
    return ratings, movie_info, movie2idx, idx2movie

@st.cache_resource
def load_ncf():
    ratings, _, _, _ = load_data()
    all_users  = sorted(ratings["user_id"].unique())
    user2idx   = {u: i for i, u in enumerate(all_users)}
    ckpt = torch.load(NCF_PATH, map_location="cpu")
    hp   = ckpt["hparams"]
    ncf  = NCF(hp["n_users"], hp["n_movies"], hp["emb_dim"], hp["mlp_dims"])
    ncf.load_state_dict(ckpt["model_state"])
    ncf.eval()
    return ncf, user2idx

@st.cache_resource
def load_two_tower():
    return Recommender(
        data_dir=str(DATA_DIR),
        model_path=str(MODEL_PATH),
        onnx_path=str(ONNX_PATH)
    )

def ncf_recommend(user_id, top_k=10):
    ratings, movie_info, movie2idx, idx2movie = load_data()
    ncf, user2idx = load_ncf()
    uidx     = user2idx[user_id]
    all_midx = torch.arange(len(movie2idx))
    uidx_t   = torch.full_like(all_midx, uidx)
    with torch.no_grad():
        scores = ncf(uidx_t, all_midx).numpy()
    seen     = set(ratings[ratings["user_id"] == user_id]["movie_id"].tolist())
    seen_idx = [movie2idx[m] for m in seen if m in movie2idx]
    scores[seen_idx] = -1.0
    top_idx  = np.argsort(scores)[::-1][:top_k]
    results  = []
    for idx in top_idx:
        mid = idx2movie[idx]
        results.append({
            "title":  movie_info.loc[mid, "title"],
            "genres": movie_info.loc[mid, "genres"],
            "score":  float(scores[idx])
        })
    return results

def tt_recommend(user_id, top_k=10):
    _, movie_info, _, _ = load_data()
    rec  = load_two_tower()
    resp = rec.recommend(user_id=user_id, top_k=top_k)
    if "error" in resp:
        return []
    results = []
    for r in resp["recommendations"]:
        mid = r["movie_id"]
        genres = movie_info.loc[mid, "genres"] if mid in movie_info.index else "N/A"
        results.append({
            "title":  r["title"],
            "genres": genres,
            "score":  r["score"]
        })
    return results

# ── UI ──
st.set_page_config(page_title="电影推荐引擎", page_icon="🎬", layout="wide")
st.title("🎬 电影推荐引擎")
st.caption("NCF vs Two-Tower A/B 对比")

user_id = st.number_input("输入用户 ID（1–6040）", min_value=1, max_value=6040,
                           value=1, step=1)
top_k   = st.slider("推荐数量", min_value=5, max_value=20, value=10)

if st.button("推荐", type="primary"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("NCF")
        st.caption("协同过滤，纯 ID 信号")
        with st.spinner("推理中..."):
            ncf_recs = ncf_recommend(user_id, top_k)
        df_ncf = pd.DataFrame(ncf_recs)
        df_ncf.index += 1
        df_ncf.columns = ["电影", "类型", "评分"]
        st.dataframe(df_ncf, use_container_width=True)

    with col2:
        st.subheader("Two-Tower")
        st.caption("向量召回，含用户属性 + 电影类型")
        with st.spinner("推理中..."):
            tt_recs = tt_recommend(user_id, top_k)
        df_tt = pd.DataFrame(tt_recs)
        df_tt.index += 1
        df_tt.columns = ["电影", "类型", "评分"]
        st.dataframe(df_tt, use_container_width=True)

    # 重叠分析
    ncf_titles = {r["title"] for r in ncf_recs}
    tt_titles  = {r["title"] for r in tt_recs}
    overlap    = ncf_titles & tt_titles
    st.divider()
    st.markdown(f"**两模型重叠：{len(overlap)}/{top_k}**")
    if overlap:
        st.write("共同推荐：" + "、".join(sorted(overlap)))
    else:
        st.write("无重叠——两个模型推荐结果完全不同")