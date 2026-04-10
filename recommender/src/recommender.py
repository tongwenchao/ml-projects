import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import faiss
from pathlib import Path

# ── 模型定义 ──────────────────────────────────────────────────
class TwoTowerV2(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=32, tower_dims=[64, 32], n_occ=21):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, emb_dim)
        self.occ_emb   = nn.Embedding(n_occ, 16)
        self.movie_emb = nn.Embedding(n_movies, emb_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(emb_dim + 16 + 1 + 1, tower_dims[0]), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(tower_dims[0], tower_dims[1])
        )
        self.movie_tower = nn.Sequential(
            nn.Linear(emb_dim + 18, tower_dims[0]), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(tower_dims[0], tower_dims[1])
        )
        self.register_buffer("genre_matrix", torch.zeros(n_movies, 18))
        self.register_buffer("age_min",   torch.tensor(0.0))
        self.register_buffer("age_range", torch.tensor(1.0))

    def get_item_vec(self, movie_idx):
        emb   = self.movie_emb(movie_idx)
        genre = self.genre_matrix[movie_idx]
        return self.movie_tower(torch.cat([emb, genre], dim=-1))

    def get_user_vec(self, user_idx, gender, age, occ):
        age_norm = (age - self.age_min) / self.age_range
        emb      = self.user_emb(user_idx)
        o_emb    = self.occ_emb(occ)
        return self.user_tower(torch.cat([emb, o_emb, gender, age_norm], dim=-1))


# ── 核心推理类 ────────────────────────────────────────────────
class Recommender:
    def __init__(self, data_dir: str, model_path: str):
        data_dir   = Path(data_dir)
        model_path = Path(model_path)

        # 加载数据
        ratings = pd.read_csv(data_dir / "ml-1m/ratings.dat", sep="::",
            names=["user_id","movie_id","rating","timestamp"], engine="python")
        self.movies = pd.read_csv(data_dir / "ml-1m/movies.dat", sep="::",
            names=["movie_id","title","genres"], engine="python", encoding="latin-1")
        self.users  = pd.read_csv(data_dir / "ml-1m/users.dat", sep="::",
            names=["user_id","gender","age","occupation","zip"], engine="python")

        # index 体系
        all_users  = sorted(ratings["user_id"].unique())
        all_movies = sorted(ratings["movie_id"].unique())
        self.user2idx  = {u: i for i, u in enumerate(all_users)}
        self.movie2idx = {m: i for i, m in enumerate(all_movies)}
        self.idx2movie = {i: m for m, i in self.movie2idx.items()}

        # 每个用户看过的电影
        self.watched = ratings.groupby("user_id")["movie_id"].apply(set).to_dict()

        # 加载模型
        ckpt = torch.load(model_path, map_location="cpu")
        hp   = ckpt["hparams"]
        self.model = TwoTowerV2(n_users=hp["n_users"], n_movies=hp["n_movies"],
                                emb_dim=hp["emb_dim"], tower_dims=hp["tower_dims"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # 建 Faiss 索引
        all_idx = torch.arange(len(self.movie2idx))
        with torch.no_grad():
            item_vecs = self.model.get_item_vec(all_idx).numpy().astype("float32")
        faiss.normalize_L2(item_vecs)
        self.index = faiss.IndexFlatIP(item_vecs.shape[1])
        self.index.add(item_vecs)

        print(f"Recommender ready — {len(self.user2idx)} users, "
              f"{self.index.ntotal} items in Faiss index")

    def recommend(self, user_id: int, top_k: int = 10, recall_k: int = 50):
        if user_id not in self.user2idx:
            return {"error": f"user_id {user_id} not found"}

        u = self.users[self.users["user_id"] == user_id].iloc[0]
        gender_val = 0.0 if u["gender"] == "F" else 1.0
        age_val    = float(u["age"])
        occ_val    = int(u["occupation"])

        with torch.no_grad():
            u_vec = self.model.get_user_vec(
                user_idx = torch.tensor([self.user2idx[user_id]]),
                gender   = torch.tensor([[gender_val]]),
                age      = torch.tensor([[age_val]]),
                occ      = torch.tensor([occ_val]),
            )
        u_vec_np = u_vec.numpy().astype("float32")
        faiss.normalize_L2(u_vec_np)

        # 召回 recall_k，过滤已看，取 top_k
        scores, indices = self.index.search(u_vec_np, recall_k)
        watched = self.watched.get(user_id, set())

        results = []
        for idx_, score in zip(indices[0], scores[0]):
            movie_id = self.idx2movie[idx_]
            if movie_id in watched:
                continue
            title = self.movies[self.movies["movie_id"] == movie_id]["title"].values[0]
            results.append({"movie_id": int(movie_id), "title": title, "score": round(float(score), 4)})
            if len(results) >= top_k:
                break

        return {"user_id": user_id, "recommendations": results}