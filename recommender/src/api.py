import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from recommender import Recommender

# ── 启动时加载模型，关闭时释放 ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rec = Recommender(
        data_dir   = "../data",
        model_path = "../data/two_tower_v2_best.pth"
    )
    yield
    del app.state.rec

app = FastAPI(title="Movie Recommender", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 10):
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k 必须在 1-50 之间")
    return app.state.rec.recommend(user_id, top_k=top_k)