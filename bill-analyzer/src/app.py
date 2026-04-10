import streamlit as st
import pandas as pd
import sys
sys.path.append(".")
from predict import BillPredictor

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="账单分析工具",
    page_icon="💳",
    layout="wide"
)

# ── 加载模型（只加载一次）────────────────────────────────
@st.cache_resource
def load_predictor():
    return BillPredictor(model_dir="../data")

predictor = load_predictor()

# ── 标题 ──────────────────────────────────────────────────
st.title("💳 个人账单分析")
st.caption("上传账单 CSV，自动分类消费类别并检测异常交易")

# ── 侧边栏：设置 ──────────────────────────────────────────
with st.sidebar:
    st.header("设置")
    threshold = st.slider(
        "异常检测阈值",
        min_value=0.1,
        max_value=2.0,
        value=0.50,
        step=0.05,
        help="异常分超过此值的交易会被标记，调高可减少误报"
    )
    predictor.threshold = threshold

    st.divider()
    st.header("数据格式")
    st.caption("CSV 需包含以下列：")
    st.code("Description\nAmount\nDate\nTransaction Type")

# ── 文件上传 ──────────────────────────────────────────────
uploaded = st.file_uploader("上传账单 CSV", type="csv")

if uploaded is None:
    # 没有上传时，提供一个用示例数据的按钮
    st.info("还没有上传文件，可以先用示例数据试试")
    use_sample = st.button("使用示例数据")

    if use_sample:
        uploaded = "../data/personal_transactions.csv"
    else:
        st.stop()

# ── 读取和推理 ────────────────────────────────────────────
@st.cache_data
def run_inference(file, threshold):
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    df = df[df["Transaction Type"] == "debit"].copy()
    df = df[~df["Category"].isin(["Credit Card Payment", "Paycheck"])].copy()
    df = df.reset_index(drop=True)

    transactions = df[["Description", "Amount", "Date"]].rename(columns={
        "Description": "description",
        "Amount": "amount",
        "Date": "date"
    }).to_dict("records")

    results = predictor.predict_batch(transactions)
    df_out = pd.DataFrame(results)

    # 把原始 Category 也带上，方便对比
    df_out["original_category"] = df["Category"].values
    return df_out

with st.spinner("推理中..."):
    df_result = run_inference(uploaded, threshold)

# ── 概览指标 ──────────────────────────────────────────────
st.divider()
n_total    = len(df_result)
n_anomaly  = df_result["is_anomaly"].sum()
total_amt  = df_result["amount"].str.replace("$", "", regex=False).astype(float).sum()
anomaly_amt = df_result[df_result["is_anomaly"]]["amount"].str.replace(
    "$", "", regex=False).astype(float).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("总交易数",   f"{n_total} 条")
col2.metric("总消费金额", f"${total_amt:,.2f}")
col3.metric("异常交易",   f"{n_anomaly} 条",
            delta=f"{n_anomaly/n_total:.1%}" if n_anomaly > 0 else None,
            delta_color="inverse")
col4.metric("异常金额",   f"${anomaly_amt:,.2f}")

# ── 类别分布图 ────────────────────────────────────────────
st.divider()
st.subheader("消费类别分布")

col_chart, col_table = st.columns([1, 1])

with col_chart:
    cat_counts = df_result["category"].value_counts().reset_index()
    cat_counts.columns = ["类别", "笔数"]
    st.bar_chart(cat_counts.set_index("类别"))

with col_table:
    # 各类别金额汇总
    df_result["amount_float"] = df_result["amount"].str.replace(
        "$", "", regex=False).astype(float)
    cat_summary = df_result.groupby("category").agg(
        笔数=("amount_float", "count"),
        总金额=("amount_float", "sum"),
        平均金额=("amount_float", "mean")
    ).round(2).sort_values("总金额", ascending=False)
    cat_summary["总金额"] = cat_summary["总金额"].map("${:,.2f}".format)
    cat_summary["平均金额"] = cat_summary["平均金额"].map("${:,.2f}".format)
    st.dataframe(cat_summary, use_container_width=True)

# ── 异常交易 ──────────────────────────────────────────────
st.divider()
st.subheader("⚠️ 异常交易")

df_anomaly = df_result[df_result["is_anomaly"]].sort_values(
    "anomaly_score", ascending=False)

if len(df_anomaly) == 0:
    st.success("未检测到异常交易")
else:
    st.dataframe(
        df_anomaly[["date", "description", "amount",
                    "category", "anomaly_score"]].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "anomaly_score": st.column_config.ProgressColumn(
                "异常分",
                min_value=0,
                max_value=2.0,
                format="%.4f"
            )
        }
    )

# ── 完整明细 ──────────────────────────────────────────────
st.divider()
st.subheader("完整交易明细")

# 过滤器
col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_cats = st.multiselect(
        "按类别筛选",
        options=sorted(df_result["category"].unique()),
        default=sorted(df_result["category"].unique())
    )
with col_f2:
    show_anomaly_only = st.checkbox("只显示异常交易")

df_display = df_result[df_result["category"].isin(selected_cats)]
if show_anomaly_only:
    df_display = df_display[df_display["is_anomaly"]]

# 异常行高亮
def highlight_anomaly(row):
    if row["is_anomaly"]:
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)

st.dataframe(
    df_display[["date", "description", "amount", "category",
                "confidence", "anomaly_score", "is_anomaly"]].reset_index(drop=True),
    use_container_width=True,
    height=400
)

# ── 下载 ──────────────────────────────────────────────────
st.divider()
csv = df_result.drop(columns=["amount_float"]).to_csv(index=False)
st.download_button(
    label="下载完整结果 CSV",
    data=csv,
    file_name="bill_analysis.csv",
    mime="text/csv"
)