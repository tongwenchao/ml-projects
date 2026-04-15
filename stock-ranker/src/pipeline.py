
import os, sys, json
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.insert(0, os.path.dirname(__file__))
from ranker import StockRanker

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD",  "INTC", "CRM",
    "JPM",  "BAC",  "GS",   "MS",   "WFC",
    "JNJ",  "UNH",  "PFE",  "MRK",  "ABBV"
]

DATA_DIR   = os.path.expanduser("~/ml-projects/stock-ranker/data")
OUTPUT_DIR = os.path.expanduser("~/ml-projects/stock-ranker/output")

ranker = StockRanker(
    model_path  = f"{DATA_DIR}/listnet_v2.pth",
    scaler_path = f"{DATA_DIR}/scaler_v2.pkl",
    tickers     = TICKERS
)

def daily_job():
    watch_list, date = ranker.run(top_k=5)

    # 保存到 output/YYYY-MM-DD.json
    out = {
        "date": str(date.date()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "watch_list": watch_list.rename(columns={"score": "model_score"})
                                .to_dict(orient="records")
    }
    out_path = f"{OUTPUT_DIR}/{date.date()}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[pipeline] 已保存 {out_path}")

if __name__ == "__main__":
    # 立即跑一次
    print("[pipeline] 启动，立即执行一次...")
    daily_job()

    # 每个工作日 18:00 触发（美股收盘后）
    scheduler = BlockingScheduler(timezone="America/New_York")
    scheduler.add_job(daily_job, "cron",
                      day_of_week="mon-fri", hour=18, minute=0)
    print("[pipeline] 调度器启动，每个工作日 18:00 ET 执行")
    print("[pipeline] Ctrl+C 停止")
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("[pipeline] 已停止")
