# ML Hands-on Learning Progress

## 学习者背景
- 有 ML/AI 理论背景，做过一些项目
- 无 hands-on 工程经验
- 机器：MacBook Apple Silicon (M系列)，conda 已安装
- AWS 账号已有（region: us-west-2），暂时不用 AWS infrastructure 继续学习
- 后续需要强大 GPU 时再决定用 AWS 还是 GCP，到时候 Claude 会把 setup 作为学习计划的一部分

---

## 总体学习计划

### Phase 1（已完成）：PyTorch 基础 + 部署
- 项目 B：个人账单智能分析（Week 1–6）✅
- 项目 A：个人影视推荐引擎（Week 7–12）✅
- 项目 C：股票/基金自选池助手（Week 13–18）✅

### Phase 2（未开始）：Fine-tuning + 量化蒸馏
- 平台：
  - 前期验证/代码调试：Google Colab Free / Pro（有会话时长限制，适合快速跑通逻辑）
  - 正式训练：Vast.ai / Lambda Labs（完整 Linux 环境，SSH + tmux，按小时付费）
  - 费用参考：Vast.ai RTX 4090 ~$0.3–0.5/hr，A100 ~$1.5–2.5/hr；Lambda Labs A100 80GB ~$1.25/hr；Colab Pro $10/月，Pro+ $50/月
- SFT / LoRA / QLoRA fine-tune LLM（7B 量级，如 Mistral/Llama）
- 模型量化：INT8/INT4，bitsandbytes
- ONNX export + TensorRT 推理加速
- 知识蒸馏：teacher-student，loss 设计

### Phase 3（未开始）：对齐 + 多模态
- 平台：
  - 前期验证：Google Colab Free / Pro
  - DPO/RLHF 正式训练：Vast.ai / Lambda Labs
  - Stable Diffusion fine-tune / ControlNet：直接用 Vast.ai（显存需求高，Colab 不稳定）
  - CogVideoX fine-tune：Vast.ai（需 40GB+ 显存，Colab 不支持）
  - TPU 对比实验：Google Colab TPU（免费有 TPU v2 额度）或 Google Cloud TPU（按需付费）
- DPO / RLHF：偏好对齐，从 DPO 切入（比 RLHF 实现简单）
- 图像模型：Stable Diffusion fine-tune（DreamBooth / LoRA）+ ControlNet
- 视频模型：CogVideoX fine-tune（小规模）+ 推理实验
- TPU 对比实验：同一任务在 GPU vs TPU 上的速度和成本对比

### Phase 3.5（未开始）：分布式训练 PoC
- 平台：Vast.ai 租 2x 同机 GPU（如 2x RTX 4090，~$0.8–1/hr）
- 目标：理解分布式训练核心概念，跑通 PyTorch DDP，不做生产级工程化
- 内容：
  - `torch.distributed` 初始化（init_process_group、rank、world_size）
  - `DistributedDataParallel` 包装模型
  - `DistributedSampler` 数据分片
  - 梯度在多卡间同步的过程（AllReduce）
  - 单卡 vs 多卡速度和显存对比实验
- 不做：多节点跨机器通信、fault tolerance、生产级调度（留 Phase 4）
- 用已有模型（如 Phase 2/3 fine-tune 过的 LLM）作为训练对象，聚焦分布式机制本身

### Phase 4（未开始）：大规模训练 + 工程化
- 平台：AWS（已有账号，us-west-2）或 GCP，Claude 届时提供完整 setup 指南
  - 不建议用 Vast.ai：多节点 NCCL 通信不稳定，且缺乏持久化基础设施
  - 推荐实例：p3.8xlarge（4x V100）或 p4d.24xlarge（8x A100）
- Continuous pre-training：domain adaptation，数据 pipeline
- 分布式训练：多节点，跨机器通信，fault tolerance
- MLOps：监控、serving、A/B testing
- vLLM / TGI 推理框架

---

## 项目 B：账单分析工具（已完成）✅

### 项目目录结构
```
~/ml-projects/bill-analyzer/
├── data/
│   ├── personal_transactions.csv     # 原始数据（Kaggle Personal Finance Dataset）
│   ├── bill_classifier_bert.pth      # 分类器权重
│   ├── bill_autoencoder_v2.pth       # 异常检测权重
│   ├── bert_emb_matrix.pth           # BERT embedding 矩阵 [63, 768]
│   ├── scaler_v2.pkl                 # StandardScaler（19维过滤后8维）
│   ├── keep_dims.npy                 # 有效维度索引 [1,3,9,12,15,16,17,18]
│   ├── desc_encoder.pkl              # 商家名 LabelEncoder（63个商家）
│   ├── label_encoder.pkl             # 类别 LabelEncoder（5类）
│   ├── bill_classifier.onnx          # ONNX 导出版分类器
│   └── predictions.csv              # 最新推理结果
├── notebooks/
│   ├── week1.ipynb                   # PyTorch 基础 + 分类器训练
│   ├── week2.ipynb                   # Autoencoder 异常检测
│   └── week2b.ipynb                  # BERT embedding 版本
└── src/
    ├── predict.py                    # 推理封装类 BillPredictor
    ├── app.py                        # Streamlit UI
    ├── export_onnx.py                # ONNX 导出 + 速度对比
    ├── inference.py                  # SageMaker inference 入口
    ├── deploy.py                     # SageMaker 部署脚本（已验证可用）
    └── test_endpoint.py              # SageMaker endpoint 测试
```

### 模型架构

**分类器 BillClassifierBERT**
- 输入：商家 ID（整数）+ 数值特征（金额log、星期几、是否周末）
- Embedding：预计算 bert-base-uncased CLS token [63, 768] → proj Linear(768, 16) + ReLU
- MLP：Linear(19, 32) + ReLU + Dropout(0.3) + Linear(32, 5)
- 输出：5类消费类别（食饮/购物/出行/居家/娱乐）
- val acc: 0.98–0.99，可训练参数：13109

**异常检测 BillAutoencoder**
- 输入：分类器 proj 输出(16维) + 数值特征(3维) = 19维，过滤近零方差后保留8维，StandardScaler 标准化
- Encoder：Linear(8,8) + ReLU + Linear(8,4)
- Decoder：Linear(4,8) + ReLU + Linear(8,8)
- 异常阈值：0.50（手动设定，95–99分位数之间）
- 异常判据：重建误差 > 阈值

### 关键踩坑
1. **近零方差维度导致标准化爆炸**：BERT proj 输出 16 维里有 11 维几乎全为 0，StandardScaler 除以极小 std 后产生天文数字。解决：用 VarianceThreshold(1e-6) 过滤，保留 8 维。
2. **模型迭代导致 pipeline 不一致**：升级到 BERT 后 Autoencoder 输入维度从 11 变 19，必须重训。
3. **布尔 mask 当整数索引**：True/False 被当成 0/1 index，取到错误误差值。
4. **SageMaker SDK 版本**：需要 `pip install "sagemaker<3.0"`。
5. **optimizer.zero_grad() 漏写**：梯度累加导致 loss 爆炸。
6. **Autoencoder 需要标准化，分类器不需要**：MSE loss 直接受特征值域影响。

### ONNX 导出结果
- PyTorch vs ONNX 最大误差：0.000005
- 推理速度：PyTorch 0.021ms → ONNX Runtime 0.007ms，加速 3.21x

### SageMaker 部署（已验证，暂不继续用 AWS）
- IAM Role：SageMakerExecutionRole（AmazonSageMakerFullAccess + AmazonS3FullAccess）
- 实例：ml.t2.medium（~$0.05/hr）
- 删除命令：`aws sagemaker delete-endpoint --endpoint-name bill-analyzer --region us-west-2`

### GitHub
- 仓库：https://github.com/tongwenchao/bill-analyzer（Private）
- 不提交：.pth / .pkl / .npy / .csv / .DS_Store

---

## 项目 A：影视推荐引擎（已完成）✅

### 项目目录
```
~/ml-projects/recommender/
├── data/
│   ├── ml-1m/
│   │   ├── movies.dat               # 3883 部电影（3706 有评分，177 无评分）
│   │   ├── ratings.dat              # 1,000,209 条评分
│   │   └── users.dat                # 6040 个用户
│   ├── mf_best.pth                  # MF 最佳权重（epoch 8，val RMSE 0.8934）
│   ├── ncf_clean.pth                # NCF 重训权重（含 hparams）
│   ├── two_tower_v2_best.pth        # Two-Tower 最佳权重（含 hparams）
│   └── user_tower.onnx              # ONNX 导出的 user tower
├── notebooks/
│   ├── week7.ipynb                  # 数据探索 + Matrix Factorization
│   ├── week8.ipynb                  # NCF + 负采样
│   ├── week9.ipynb                  # 内容特征 + Two-Tower
│   ├── week10.ipynb                 # Faiss 召回 + FastAPI
│   ├── week11.ipynb                 # ONNX 优化
│   └── week12.ipynb                 # Streamlit UI + 收尾
└── src/
    ├── recommender.py               # 核心推理类（ONNX + Faiss）
    ├── api.py                       # FastAPI 推理接口
    └── app.py                       # Streamlit A/B 对比 UI
```

### 完整技术栈
```
数据探索 → Matrix Factorization → NCF → Two-Tower
→ Faiss 向量召回 → FastAPI → ONNX 加速 → Streamlit UI
```

### Index 体系（重要）
```python
all_users  = sorted(ratings["user_id"].unique())
all_movies = sorted(ratings["movie_id"].unique())
user2idx  = {u: i for i, u in enumerate(all_users)}   # 6040
movie2idx = {m: i for i, m in enumerate(all_movies)}  # 3706（有评分的）
# 177 部无评分电影直接排除
```

### 模型权重
- `ncf_clean.pth`：`{"model_state": ..., "hparams": {"emb_dim":32, "mlp_dims":[64,32,16], "n_users":6040, "n_movies":3706}}`
- `two_tower_v2_best.pth`：`{"model_state": ..., "hparams": {"emb_dim":32, "tower_dims":[64,32], "n_users":6040, "n_movies":3706}}`
- `user_tower.onnx`：Two-Tower user tower ONNX 导出版，2.8 KB

### NCF 架构（重要：与训练时完全一致）
```python
class NCF(nn.Module):
    def __init__(self, n_users, n_movies, emb_dim=32, mlp_dims=[64,32,16]):
        # MF 路径：mf_user_emb, mf_movie_emb
        # MLP 路径：mlp_user_emb, mlp_movie_emb
        # MLP 每组：Linear → ReLU → Dropout（下标跳3，所以有 mlp.0/3/6）
        # 输出层名为 self.out（不是 output_layer）
        # Linear(emb_dim + mlp_dims[-1], 1) = Linear(48, 1)
```

### Two-Tower 架构
- User Tower：user_emb(32) + occ_emb(16) + gender scalar(1) + age归一化(1) → Linear(50→64)→ReLU→Dropout→Linear(64→32)
- Item Tower：movie_emb(32) + genre多热(18) → Linear(50→64)→ReLU→Dropout→Linear(64→32)
- 输出：内积 → sigmoid
- state_dict 层名：`movie_tower`（不是 item_tower），buffer：`age_min`、`age_range`、`genre_matrix`

### Recommender 接口
```python
rec = Recommender(data_dir="../data", model_path="...", onnx_path="...")
resp = rec.recommend(user_id=1, top_k=10)
# 返回：{"user_id": 1, "recommendations": [{"movie_id", "title", "score"}, ...]}
# genres 不在返回里，在 app 层用 movie_info.loc[mid, "genres"] 补查
```

### 最终评估结果

| 模型 | HR@10 | NDCG@10 | 参数量 | 备注 |
|------|-------|---------|--------|------|
| 随机基准 | 0.100 | — | — | |
| NCF | 0.404 | 0.208 | 631k | 纯 ID，无内容特征 |
| Two-Tower | 0.410 | 0.219 | 323k | user属性+genre，参数更少 |

A/B 对比（用户1）：NCF 动作/惊悚为主（score 0.90+），Two-Tower 喜剧/剧情为主（余弦相似度 0.39–0.49），重叠仅 1/6。

### 关键踩坑（全 12 周汇总）
1. **clamp 梯度陷阱**：值落在边界时梯度为0，初始化必须让预测落在合理范围内。
2. **Data leakage**：推荐系统必须按时间戳切分，不能随机切分。
3. **两套 index 并存是静默 bug 的来源**：全程只维护一套 movie2idx，混用不报错但结果完全错误。
4. **dying ReLU**：`nn.init.normal_(std=0.01)` + 多层 ReLU → 输出全零，loss 卡 0.693。修复：kaiming_uniform_，tower 最后一层不加 ReLU。
5. **评估随机性**：随机负采样导致 HR@10 抖动 ±0.02，跨模型对比必须用固定评估集。
6. **faiss-cpu pip 版在 Apple Silicon 上 segfault**：必须用 `conda install -c conda-forge faiss-cpu`。
7. **MPS 上 Faiss 会 crash**：全程用 CPU，对 32 维小模型无速度损失。
8. **OpenMP 冲突**：torch + faiss 并存报 OMP Error #15。永久修复：写入 `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`。
9. **Jupyter kernel crash 难定位**：terminal 里 `python -c` 最小化复现，segfault 会明确打印。
10. **NCF 定义必须和训练时完全一致**：层名（`out` 不是 `output_layer`）、有无 Dropout 都影响 load_state_dict，不匹配会导致 notebook kernel restart。
11. **保存模型必须带 hparams**：`torch.save({"model_state": ..., "hparams": {...}}, path)`。
12. **ONNX 只优化热路径**：user tower（每次请求跑）走 ONNX，item tower（启动时跑一次）保持 PyTorch；fallback 机制保证 ONNX 文件不存在时服务仍可用。

### 服务启动
```bash
# FastAPI
cd ~/ml-projects/recommender/src
uvicorn api:app --reload --port 8000

# Streamlit
cd ~/ml-projects/recommender/src
streamlit run app.py
```

---

## 项目 C：股票/基金自选池助手（已完成）✅

### 项目目录
```
~/ml-projects/stock-ranker/
├── data/
│   ├── features_labeled.csv         # 特征+标签数据（200行，Week13产出）
│   ├── pairs_train.csv              # pairwise 训练数据（400对，5只股票版）
│   ├── ranknet_v1.pth               # RankNet v1（5只股票，3个月数据）
│   ├── listnet_v1.pth               # ListNet v1（5只股票，3个月数据）
│   ├── ranknet_v2.pth               # RankNet v2（20只股票，1年数据）
│   ├── listnet_v2.pth               # ListNet v2（20只股票，1年数据）✅最终使用
│   ├── scaler_v2.pkl                # StandardScaler（8维特征）
│   └── listnet_v2.onnx              # ONNX 导出版 ListNet
├── notebooks/
│   ├── week13.ipynb                 # 数据获取 + 时序特征工程
│   ├── week14.ipynb                 # Pairwise 数据构造 + RankNet
│   ├── week15.ipynb                 # ListNet 实现
│   ├── week16.ipynb                 # 扩大股票池 + 模型对比
│   ├── week17.ipynb                 # 定时推理 pipeline
│   └── week18.ipynb                 # ONNX 导出 + Streamlit UI
├── src/
│   ├── ranker.py                    # StockRanker 推理类
│   ├── pipeline.py                  # APScheduler 定时调度
│   └── app.py                       # Streamlit UI
└── output/
    └── YYYY-MM-DD.json              # 每日 watch list 输出
```

### 完整技术栈
```
yfinance 数据拉取
→ 时序特征工程（动量/波动率/相对强弱/成交量）
→ Learning-to-Rank（RankNet pairwise + ListNet listwise）
→ ONNX 导出加速（5.49x）
→ APScheduler 定时 pipeline（每个工作日 18:00 ET）
→ Streamlit UI
```

### 特征工程（8维）

| 特征 | 计算方式 | 含义 |
|------|----------|------|
| mom_5d | pct_change(5) | 5日动量 |
| mom_10d | pct_change(10) | 10日动量 |
| mom_20d | pct_change(20) | 20日动量 |
| vol_10d | rolling(10).std() | 10日波动率 |
| vol_20d | rolling(20).std() | 20日波动率 |
| rsi_dist | (close - ma20) / std20 | 距20日均线标准差 |
| high_20d_ratio | close / rolling(20).max() | 现价/20日高点 |
| vol_ratio | volume / rolling(10).mean() | 量比 |

### 模型架构

**RankNet（pairwise）**
```python
class RankNet(nn.Module):
    # scorer: Linear(8,32) → ReLU → Dropout(0.2) → Linear(32,16) → ReLU → Linear(16,1)
    # 训练：sigmoid(score_i - score_j) → BCELoss
    # 推理：scorer(f) 单独打分，按分数排序
    # 参数量：833
```

**ListNet（listwise）**
```python
class ListNet(nn.Module):
    # scorer: Linear(8,32) → ReLU → Dropout(0.2) → Linear(32,16) → ReLU → Linear(16,1)
    # loss: -sum(softmax(-ranks) * log(softmax(scores)))  # KL 散度
    # 推理：同 RankNet，scorer 单独打分
```

### 训练数据规模对比

| 版本 | 股票数 | 历史 | pairwise对数 | listwise天数 |
|------|--------|------|-------------|-------------|
| v1 | 5只 | 3个月 | 400 | 40 |
| v2 | 20只 | 1年 | 43510 | 229 |

### 最终评估结果（v2，val集）

| 模型 | NDCG@5 | val_acc | 备注 |
|------|--------|---------|------|
| 随机基准 | 0.200 | 0.500 | |
| RankNet | 0.540 | 0.490 | pairwise 边界对噪声大 |
| **ListNet** | **0.603** | — | listwise 对整体排序更鲁棒 |

### ONNX 导出结果
- PyTorch vs ONNX 最大误差：0.00000012
- 推理速度：PyTorch 0.027ms → ONNX Runtime 0.005ms，加速 **5.49x**
- `dynamic_axes` 设置让股票池大小可变，推理时无需重训

### 服务启动
```bash
# 定时 pipeline（立即执行一次 + 每日18:00 ET 自动触发）
cd ~/ml-projects/stock-ranker/src
python pipeline.py

# Streamlit UI
cd ~/ml-projects/stock-ranker/src
streamlit run app.py
```

### 关键踩坑
1. **rolling(52) ≠ 52周**：日线数据 rolling(52) 是52个交易日，约2.5个月，不是1年。数据窗口必须和特征窗口匹配。
2. **标准化对 MLP 至关重要**：各特征值域差异大（mom 0.01级 vs rsi_dist 1–3级），加 StandardScaler 后 val_acc 0.56→0.66。
3. **ListNet 是梯度累积而非真正 batch**：每天是独立列表，"batch"是多天 loss 求和后一起 backward，不是把多天股票混在一起算 softmax。
4. **数据量决定算法选择**：数据量小→RankNet 更稳（pairwise 等效样本量更大）；数据量大→ListNet 更好（listwise 直接优化排序目标）。
5. **pairwise 边界对噪声**：20只股票时 rank 相邻的两只本来就难区分，强行要求分对边界对会干扰学习，RankNet val_acc 跌破0.5。
6. **LTR 模型不依赖股票 ID**：输入是特征向量，新加股票无需重训，天然没有冷启动问题（与推荐系统 Embedding 的本质区别）。
7. **推理和训练形式不同**：训练输入 (fi, fj) 对，推理时每只股票单独过 scorer 得分，再排序。

---

## 重要概念总结（已掌握）

### PyTorch 核心
- `requires_grad=True`：追踪该变量用于自动求导
- `loss.backward()`：反向传播，沿计算图求导
- `optimizer.zero_grad()`：清空梯度（必须在 backward 前调用）
- `optimizer.step()`：用梯度更新参数
- `model.train()` / `model.eval()`：切换训练/推理模式（影响 Dropout、BatchNorm）
- `torch.no_grad()`：推理时关闭梯度计算，省内存
- `torch.save({"model_state": ..., "hparams": {...}})` / `model.load_state_dict(ckpt["model_state"])`：保存/加载权重，必须带 hparams
- `register_buffer`：注册不需要训练的固定张量，保存加载时自动处理，自动跟模型走同一个 device

### 模型设计
- `nn.Embedding`：离散 ID → 连续向量，本质是可训练的查找表
- `nn.Linear` + `nn.ReLU`：基本 MLP 构建块
- `nn.Dropout`：训练时随机丢弃神经元，缓解 overfit
- `nn.CrossEntropyLoss`：分类任务 loss
- `F.mse_loss`：回归/重建任务 loss
- `nn.BCELoss`：二分类任务 loss（配合 sigmoid 输出使用）
- `nn.init.kaiming_uniform_(nonlinearity='relu')`：ReLU 网络的正确初始化，避免 dying ReLU
- 特征类型决定编码方式：二值特征用 scalar，有序连续特征归一化后 concat，无序类别特征用 embedding

### 工程概念
- Overfitting：train loss 降、val loss 升，模型记住了噪声
- 标准化（StandardScaler）vs 归一化（MinMaxScaler）：前者对异常值更鲁棒
- 迁移学习：冻结预训练权重，只训练后面的 head
- ONNX：模型格式标准化，脱离 PyTorch 依赖，推理更快；导出时必须在 eval() 模式，Dropout 会被消除
- ONNX 计算图：PyTorch 的 Python 操作被追踪并展开为算子级节点（Gather/Gemm/Relu等），Netron 可视化
- ONNX dynamic_axes：让指定维度（如 batch size、股票池大小）在推理时可变
- SageMaker endpoint 内部：TorchServe + inference.py，不是 Streamlit
- batch size 影响：越大梯度越平滑、更新频率越低；小 batch 的噪声有正则化效果
- 梯度累积：多步 loss 累加后统一 backward，等效增大 batch size，不等于真正的 batched tensor 计算
- Early stopping：val loss 连续 N 步不改善则停止，配合保存 best_state 使用

### 推荐系统专项
- 矩阵分解（MF）：user/item embedding 点积预测评分，压缩至隐空间
- Bias 设计：global_bias 承担全局均值，user/item bias 学习偏离量，职责分离
- 显式反馈 vs 隐式反馈：显式用 RMSE，隐式用 HR@K / NDCG@K
- 负采样：正负比例 1:4（NCF 原论文经验值）
- BCE loss 随机基准：loss = ln(2) ≈ 0.693，低于此值说明模型在学习
- Two-Tower：user/item 各自独立建模；item tower 输出离线建 Faiss 索引，天然支持大规模召回
- 冷启动：user 属性特征弥补稀疏 embedding；item 内容特征弥补新物品无交互历史
- Faiss IndexFlatIP + L2 归一化 = 余弦相似度

### Learning-to-Rank 专项
- LTR 三种范式：pointwise（回归/分类）、pairwise（比较两两）、listwise（优化整个列表）
- RankNet：pairwise，sigmoid(si - sj) → BCELoss，等效样本量大，数据量小时更稳
- ListNet：listwise，cross-entropy(softmax(pred), softmax(-rank))，数据量大时更好
- NDCG@K：排序质量指标，考虑位置折扣，比 accuracy 更贴近真实排序目标
- LTR 无冷启动：输入是特征向量而非 ID，新候选项直接推理，无需重训
- 训练与推理形式分离：训练输入 (fi, fj) 对，推理时单独打分再排序

### 定时任务
- APScheduler BlockingScheduler：阻塞式调度，`cron` trigger 指定 day_of_week/hour/minute
- 模型加载与推理分离：`__init__` 加载一次，job 函数只做推理（与 FastAPI lifespan 同一思路）

### 环境问题专项（Apple Silicon）
- **faiss**：`conda install -c conda-forge faiss-cpu`，不要用 pip
- **MPS**：小模型推理用 CPU 更稳定
- **OpenMP 冲突**：永久方案：写入 `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`
- **Jupyter crash 定位**：terminal 里 `python -c` 复现，segfault 会明确打印

### API 服务设计
- FastAPI lifespan：启动时加载模型/索引，避免每次请求重复加载
- `GET /health`：健康检查，部署验证必备
- 召回与过滤分离：Faiss 召回 recall_k（如50），过滤已看后取 top_k（如10）
- ONNX 集成模式：`_get_user_vec` 统一封装 ONNX/PyTorch 两条路径，外部调用无感知

---

## 新 Chat 继续的说明

把以下内容贴给新的 Claude 即可继续：

---

我正在系统学习 ML hands-on 技能，请扮演我的老师，按照以下进度继续。

**当前进度**：Phase 1 全部完成，准备开始 Phase 2（Fine-tuning + 量化蒸馏）。

**已完成**：
- 项目 B 账单分析工具（PyTorch 分类器 + Autoencoder 异常检测 + Streamlit + SageMaker 部署 + ONNX 导出）
- 项目 A 影视推荐引擎（Matrix Factorization → NCF → Two-Tower → Faiss → FastAPI → ONNX → Streamlit，Week 7–12 全部完成）
- 项目 C 股票/基金自选池助手（时序特征工程 + RankNet + ListNet + APScheduler + ONNX + Streamlit，Week 13–18 全部完成）

**Phase 2 定位**：
- 平台：前期验证用 Google Colab，正式训练用 Vast.ai / Lambda Labs
- 目标：SFT / LoRA / QLoRA fine-tune LLM（7B 量级，如 Mistral/Llama）
- 技术点：模型量化（INT8/INT4，bitsandbytes）、ONNX + TensorRT 推理加速、知识蒸馏

**环境**：MacBook Apple Silicon，conda，mlenv 环境
- faiss 必须用 conda-forge 版：`conda install -c conda-forge faiss-cpu`
- KMP_DUPLICATE_LIB_OK=TRUE 已永久写入 conda activate.d

**注意事项**：
- 暂不使用 AWS infrastructure
- 教学风格：每次只给一个 cell 的代码，跑完看结果再继续，遇到问题用数据验证而不是猜测

完整学习进度文档见附件 `ml_learning_progress_v10.md`。
