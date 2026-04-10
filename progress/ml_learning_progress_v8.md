# ML Hands-on Learning Progress

## 学习者背景
- 有 ML/AI 理论背景，做过一些项目
- 无 hands-on 工程经验
- 机器：MacBook Apple Silicon (M系列)，conda 已安装
- AWS 账号已有（region: us-west-2），暂时不用 AWS infrastructure 继续学习
- 后续需要强大 GPU 时再决定用 AWS 还是 GCP，到时候 Claude 会把 setup 作为学习计划的一部分

---

## 总体学习计划

### Phase 1（进行中）：PyTorch 基础 + 部署
- 项目 B：个人账单智能分析（Week 1–6）✅
- 项目 A：个人影视推荐引擎（Week 7–12）✅
- 项目 C：股票/基金自选池助手 🔄 进行中

### Phase 2（未开始）：Fine-tuning + 量化蒸馏
- 平台：本机 MPS + Vast.ai / Lambda Labs（按需租 GPU，约 $1.5/hr A100）
- SFT / LoRA / QLoRA fine-tune LLM（7B 量级，如 Mistral/Llama）
- 模型量化：INT8/INT4，bitsandbytes
- ONNX export + TensorRT 推理加速
- 知识蒸馏：teacher-student，loss 设计

### Phase 3（未开始）：对齐 + 多模态
- 平台：Vast.ai / Lambda Labs + Google Cloud TPU（试用）
- DPO / RLHF：偏好对齐，从 DPO 切入（比 RLHF 实现简单）
- 图像模型：Stable Diffusion fine-tune（DreamBooth / LoRA）+ ControlNet
- 视频模型：CogVideoX fine-tune（小规模）+ 推理实验
- TPU 对比实验：同一任务在 GPU vs TPU 上的速度和成本对比

### Phase 4（未开始）：大规模训练 + 工程化
- 平台：视需要开通 AWS（新账号）或 GCP，Claude 届时提供完整 setup 指南
- Continuous pre-training：domain adaptation，数据 pipeline
- 分布式训练：多 GPU / 多节点
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

## 项目 C：股票/基金自选池助手（进行中）🔄

### 定位
聚焦在**排序逻辑**，不做价格预测（容易掉坑）。输入持仓偏好 → 模型打分排序 → 推送每日 watch list。

### 目标技术点
- 时序特征工程
- Learning-to-rank（LambdaRank / ListNet）
- 定时推理 pipeline
- 数据源：yfinance（免费）

### 当前进度
- 尚未开始

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
- SageMaker endpoint 内部：TorchServe + inference.py，不是 Streamlit
- batch size 影响：越大梯度越平滑、更新频率越低；小 batch 的噪声有正则化效果

### 推荐系统专项
- 矩阵分解（MF）：user/item embedding 点积预测评分，压缩至隐空间
- Bias 设计：global_bias 承担全局均值，user/item bias 学习偏离量，职责分离
- 显式反馈 vs 隐式反馈：显式用 RMSE，隐式用 HR@K / NDCG@K
- 负采样：正负比例 1:4（NCF 原论文经验值）
- BCE loss 随机基准：loss = ln(2) ≈ 0.693，低于此值说明模型在学习
- Two-Tower：user/item 各自独立建模；item tower 输出离线建 Faiss 索引，天然支持大规模召回
- 冷启动：user 属性特征弥补稀疏 embedding；item 内容特征弥补新物品无交互历史
- Faiss IndexFlatIP + L2 归一化 = 余弦相似度

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

**当前进度**：项目 C 股票/基金自选池助手，尚未开始。

**已完成**：
- 项目 B 账单分析工具（PyTorch 分类器 + Autoencoder 异常检测 + Streamlit + SageMaker 部署 + ONNX 导出）
- 项目 A 影视推荐引擎（Matrix Factorization → NCF → Two-Tower → Faiss → FastAPI → ONNX → Streamlit，Week 7–12 全部完成）

**项目 C 定位**：
- 聚焦排序逻辑，不做价格预测
- 输入持仓偏好 → 模型打分排序 → 推送每日 watch list
- 技术点：时序特征工程、Learning-to-rank（LambdaRank/ListNet）、定时推理 pipeline
- 数据源：yfinance（免费）

**环境**：MacBook Apple Silicon，conda，mlenv 环境
- faiss 必须用 conda-forge 版：`conda install -c conda-forge faiss-cpu`
- KMP_DUPLICATE_LIB_OK=TRUE 已永久写入 conda activate.d

**注意事项**：
- 暂不使用 AWS infrastructure
- 教学风格：每次只给一个 cell 的代码，跑完看结果再继续，遇到问题用数据验证而不是猜测

完整学习进度文档见附件 `ml_learning_progress_v8.md`。
