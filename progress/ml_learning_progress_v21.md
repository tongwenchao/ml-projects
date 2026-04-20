# ML Hands-on Learning Progress

## 学习者背景
- 有 ML/AI 理论背景，做过一些项目
- 无 hands-on 工程经验
- 机器：MacBook Apple Silicon M2，16GB RAM，conda 已安装
  - 本地可用于：INT4量化推理（7B以内）、ONNX导出验证、代码调试、student模型训练
  - 本地不适用：7B FP16推理、任何规模的 LoRA fine-tune
- AWS 账号已有（region: us-west-2），暂时不用 AWS infrastructure 继续学习
- 后续需要强大 GPU 时再决定用 AWS 还是 GCP，到时候 Claude 会把 setup 作为学习计划的一部分

---

## 总体学习计划

### Phase 1（已完成）：PyTorch 基础 + 部署
- 项目 B：个人账单智能分析（Week 1–6）✅
- 项目 A：个人影视推荐引擎（Week 7–12）✅
- 项目 C：股票/基金自选池助手（Week 13–18）✅

### Phase 2（进行中）：Fine-tuning + 量化蒸馏
- 平台分工：
  - 本地 M2：推理验证、ONNX导出、代码调试、student模型训练
  - Google Colab：中等规模实验、逻辑验证（有会话时长限制）
  - Vast.ai / Lambda Labs：正式 fine-tune 训练（SSH + tmux，按小时付费）
  - 费用参考：Vast.ai RTX 4090 ~$0.3–0.5/hr，A100 ~$1.5–2.5/hr；Lambda Labs A100 80GB ~$1.25/hr；Colab Pro $10/月
  - 预计总费用：$45–70
- 项目 D：本地 LLM 推理引擎（Week 19–21）✅ 已完成
- 项目 E：LoRA Fine-tune（Week 22–25）✅ 已完成
- 项目 F：模型量化实验（Week 26–28）✅ 已完成
- 项目 G：知识蒸馏（Week 29–32）
- 项目 H1：股票新闻摘要（Week 33–38）
- 项目 H2：知识库问答 + RAG（Week 39–43）

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

## Phase 2 详细计划

### 项目 D：本地 LLM 推理引擎（Week 19–21，本地 M2）✅

**目标**：在本地 M2 跑通 7B 模型推理，理解量化的实际效果，建立对 LLM 推理的直觉。

| 周 | 内容 | 平台 | 状态 |
|---|---|---|---|
| W19 | llama.cpp 安装 + Llama-3.2-3B / Mistral-7B INT4 推理，测速（tokens/sec） | 本地 M2 | ✅ |
| W20 | Apple MLX 框架推理对比（专为 M 系列优化），llama.cpp vs MLX 横向对比 | 本地 M2 | ✅ |
| W21 | 用 mlx-lm 做 few-shot prompting，封装成本地 CLI 工具 | 本地 M2 | ✅ |

**技术点**：GGUF 格式、INT4/INT8 量化原理、KV cache、MLX vs llama.cpp 架构差异、tokens/sec 作为推理速度指标、few-shot multi-turn 格式。

---

### 项目 E：LoRA Fine-tune（Week 22–25）✅

**目标**：用 LoRA 在特定任务上 fine-tune 7B 模型，理解参数高效微调的原理和工程实现。

| 周 | 内容 | 平台 | 状态 |
|---|---|---|---|
| W22 | Colab 上跑通 LoRA 训练流程（小模型 + 小数据验证逻辑） | Colab | ✅ |
| W23 | Vast.ai 上 QLoRA fine-tune Mistral-7B（金融情感分类数据集） | Vast.ai RTX 3090 | ✅ |
| W24 | 评估：confusion matrix + 人工案例分析，base model zero-shot 对比 | Colab | ✅ |
| W25 | 合并 adapter 权重，本地 NF4 推理验证 | Vast.ai RTX 3090 + 本地 M2 | ✅ |

**技术点**：LoRA 数学原理（低秩分解 W = W₀ + BA）、QLoRA（NF4量化+LoRA）、bitsandbytes、PEFT 库、Hugging Face Trainer、rank / alpha 超参数含义。

---

### 项目 F：模型量化实验（Week 26–28，本地 M2 + Colab）

**目标**：动手做 INT8 / INT4 量化，用数据理解精度和速度的 tradeoff。

| 周 | 内容 | 平台 | 状态 |
|---|---|---|---|
| W26 | bitsandbytes INT8 量化，perplexity 测量（量化损失量化） | Colab | ✅ |
| W27 | NF4 INT4 量化 + 显存解剖实验（GPTQ 环境不兼容跳过） | Colab | ✅ |
| W28 | ONNX 导出 + onnxruntime 推理，和 PyTorch 对比速度 | 本地 M2 | ✅ |

**技术点**：PTQ（训练后量化）vs QAT（量化感知训练）、对称/非对称量化、GPTQ 算法原理、perplexity 作为评估指标、量化粒度（per-tensor vs per-channel）。

---

### 项目 G：知识蒸馏（Week 29–32，Colab + 本地）

**目标**：用大模型（teacher）蒸馏出小模型（student），理解 loss 设计和温度参数。

| 周 | 内容 | 平台 |
|---|---|---|
| W29 | 蒸馏理论 + 用 GPT-2 large 蒸馏 GPT-2 small（验证流程） | Colab |
| W30 | 设计蒸馏 loss：soft label KL 散度 + hard label CE，实验不同温度 T | Colab |
| W31 | 中间层蒸馏（hidden state alignment），对比只蒸馏输出 vs 蒸馏中间层 | Colab |
| W32 | student 模型本地 M2 推理，和 teacher INT4 量化版做速度/精度对比 | 本地 M2 |

**技术点**：Hinton 蒸馏 loss `L = α·CE(hard) + (1-α)·KL(soft)`、温度参数 T 的作用（T大→软标签更平滑）、task-specific vs general 蒸馏、hidden state alignment loss。

---

### 项目 H1：股票新闻摘要（Week 33–38，Vast.ai + 本地）

**目标**：把 fine-tune + 量化 + 蒸馏串成完整链路，产出一个可本地运行的金融摘要模型。

| 周 | 内容 | 平台 |
|---|---|---|
| W33 | 数据准备：Financial PhraseBank + 爬取近期财经新闻，构造摘要数据集 | 本地 M2 |
| W34 | 数据清洗、格式化为 instruction-following 格式，EDA | 本地 M2 |
| W35 | Vast.ai 上 QLoRA fine-tune Mistral-7B（摘要任务） | Vast.ai A100 |
| W36 | GPTQ INT4 量化 fine-tuned 模型，ROUGE 评估 | Colab |
| W37 | 知识蒸馏：fine-tuned 7B 作 teacher，蒸馏出 1–3B student | Vast.ai RTX 4090 |
| W38 | student 本地推理 + Streamlit UI 封装，收尾 | 本地 M2 |

**技术点**：instruction-following 数据格式（prompt template）、ROUGE-1/2/L 评估摘要质量、摘要任务特有的 length penalty、seq2seq vs decoder-only 的摘要方式差异。

---

### 项目 H2：知识库问答 + RAG（Week 39–43，本地 M2 + Colab）

**目标**：在 H1 fine-tuned 模型基础上新增 RAG 检索层，理解 LLM 工程落地最核心的范式。

| 周 | 内容 | 平台 |
|---|---|---|
| W39 | RAG 原理 + 向量数据库选型（ChromaDB），构建本地知识库 | 本地 M2 |
| W40 | Embedding 模型选型（sentence-transformers），文档分块策略实验 | 本地 M2 |
| W41 | 检索层实现：语义检索 + BM25 混合检索，召回质量评估 | 本地 M2 |
| W42 | 生成层：用 H1 student 模型做生成端，上下文注入 prompt 设计 | 本地 M2 |
| W43 | Streamlit UI（支持上传 PDF/文档），端到端测试，收尾 | 本地 M2 |

**技术点**：RAG vs fine-tune 的适用场景对比、文档分块策略（固定长度 vs 语义分块）、向量检索 + BM25 混合检索、上下文窗口管理、幻觉（hallucination）缓解策略。

**与项目 C 的联系**：RAG 检索层本质是语义版 Faiss 召回——项目 C 用 Faiss 召回股票，H2 用 Faiss 召回文档片段，核心是同一套向量检索逻辑。

---

## Phase 2 汇总

| 项目 | 周数 | 主要平台 | 核心技术 |
|------|------|---------|---------|
| D：本地推理引擎 | W19–21（3周） | 本地 M2 | llama.cpp、MLX、GGUF、KV cache、few-shot |
| E：LoRA Fine-tune | W22–25（4周） | Colab + Vast.ai | LoRA、QLoRA、PEFT、bitsandbytes |
| F：量化实验 | W26–28（3周） | 本地 M2 + Colab | INT8/INT4、GPTQ、perplexity |
| G：知识蒸馏 | W29–32（4周） | Colab + 本地 | soft label KL、温度T、中间层蒸馏 |
| H1：股票新闻摘要 | W33–38（6周） | Vast.ai + 本地 | 完整链路：fine-tune→量化→蒸馏 |
| H2：知识库问答+RAG | W39–43（5周） | 本地 M2 | RAG、ChromaDB、混合检索 |
| **合计** | **25周** | | |

**Vast.ai 预计费用**：
- 项目 E：实际花费 ~$1（两次 RTX 3090，合计约 2–3 小时）
- 项目 H1：~$20–30（A100，约15小时 + RTX 4090，约20小时）
- **总计：$30–45**（加 Colab Pro $10/月，整体 $45–70）

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

### 最终指标
- val acc: 0.98–0.99
- PyTorch vs ONNX 最大误差：0.000005
- 推理速度：PyTorch 0.021ms → ONNX Runtime 0.007ms，加速 3.21x

### 关键踩坑
1. **近零方差维度导致标准化爆炸**：BERT proj 输出 16 维里有 11 维几乎全为 0，StandardScaler 除以极小 std 后产生天文数字。解决：用 VarianceThreshold(1e-6) 过滤，保留 8 维。
2. **ONNX 导出必须在 eval() 模式**：train 模式下 Dropout 是随机的，导出的图不确定。
3. **SageMaker 内部是 TorchServe**：不是 Streamlit，inference.py 的入口函数签名必须符合 TorchServe 规范。

### SageMaker 部署（已验证，暂不继续用 AWS）
- IAM Role：SageMakerExecutionRole（AmazonSageMakerFullAccess + AmazonS3FullAccess）
- 实例：ml.t2.medium（~$0.05/hr）
- 删除命令：`aws sagemaker delete-endpoint --endpoint-name bill-analyzer --region us-west-2`

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

### 最终评估结果

| 模型 | HR@10 | NDCG@10 | 参数量 | 备注 |
|------|-------|---------|--------|------|
| 随机基准 | 0.100 | — | — | |
| NCF | 0.404 | 0.208 | 631k | 纯 ID，无内容特征 |
| Two-Tower | 0.410 | 0.219 | 323k | user属性+genre，参数更少 |

### 关键踩坑
1. **clamp 梯度陷阱**：值落在边界时梯度为0，初始化必须让预测落在合理范围内。
2. **Data leakage**：推荐系统必须按时间戳切分，不能随机切分。
3. **两套 index 并存是静默 bug 的来源**：全程只维护一套 movie2idx。
4. **dying ReLU**：`nn.init.normal_(std=0.01)` + 多层 ReLU → 输出全零，loss 卡 0.693。修复：kaiming_uniform_，tower 最后一层不加 ReLU。
5. **评估随机性**：随机负采样导致 HR@10 抖动 ±0.02，跨模型对比必须用固定评估集。
6. **faiss-cpu pip 版在 Apple Silicon 上 segfault**：必须用 `conda install -c conda-forge faiss-cpu`。
7. **MPS 上 Faiss 会 crash**：全程用 CPU。
8. **OpenMP 冲突**：torch + faiss 并存报 OMP Error #15。永久修复：写入 `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`。
9. **NCF 定义必须和训练时完全一致**：层名（`out` 不是 `output_layer`）。
10. **保存模型必须带 hparams**：`torch.save({"model_state": ..., "hparams": {...}}, path)`。
11. **ONNX 只优化热路径**：user tower 走 ONNX，item tower 保持 PyTorch。

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
│   ├── week13.ipynb                 # 特征工程
│   ├── week14.ipynb                 # RankNet
│   ├── week15.ipynb                 # ListNet
│   ├── week16.ipynb                 # 扩展数据集（20只股票）
│   ├── week17.ipynb                 # ONNX 导出
│   └── week18.ipynb                 # APScheduler + Streamlit
└── src/
    ├── ranker.py                    # 推理封装
    ├── scheduler.py                 # APScheduler 定时任务
    └── app.py                       # Streamlit UI
```

### 最终评估结果（v2，val集）

| 模型 | NDCG@5 | val_acc | 备注 |
|------|--------|---------|------|
| 随机基准 | 0.200 | 0.500 | |
| RankNet | 0.540 | 0.490 | pairwise 边界对噪声大 |
| **ListNet** | **0.603** | — | listwise 对整体排序更鲁棒 |

### ONNX 导出结果
- PyTorch vs ONNX 最大误差：0.00000012
- 推理速度：PyTorch 0.027ms → ONNX Runtime 0.005ms，加速 **5.49x**

### 关键踩坑
1. **rolling(52) ≠ 52周**：日线数据 rolling(52) 是52个交易日，约2.5个月，不是1年。
2. **标准化对 MLP 至关重要**：加 StandardScaler 后 val_acc 0.56→0.66。
3. **ListNet 是梯度累积而非真正 batch**：每天是独立列表，"batch"是多天 loss 求和后一起 backward。
4. **数据量决定算法选择**：数据量小→RankNet 更稳；数据量大→ListNet 更好。
5. **pairwise 边界对噪声**：20只股票时 RankNet val_acc 跌破0.5。
6. **LTR 模型不依赖股票 ID**：输入是特征向量，天然没有冷启动问题。
7. **推理和训练形式不同**：训练输入 (fi, fj) 对，推理时每只股票单独打分再排序。

---

## 项目 D：本地 LLM 推理引擎（已完成）✅

### 项目目录
```
~/ml-projects/llm-inference/
├── models/
│   ├── llama-3.2-3b-instruct-q4_k_m.gguf           # 1.9GB（llama.cpp 用）
│   ├── mistral-7b-instruct-q4_k_m.gguf              # 4.1GB（llama.cpp 用）
│   └── ~/.cache/huggingface/                         # MLX 模型缓存（HF 自动管理）
│       ├── mlx-community/Llama-3.2-3B-Instruct-4bit  # 1.82GB
│       └── mlx-community/Mistral-7B-Instruct-v0.3-4bit # 4.08GB
├── notebooks/
│   └── week19_benchmark.md                           # 推理速度基准数据
└── cli/
    └── llm_cli.py                                    # CLI 工具（Week 21 产出）
```

### 推理速度基准

**llama.cpp（Llama-3.2-3B Q4_K_M，ngl 99 vs 0）**

| ngl | Prompt t/s | Generation t/s |
|-----|-----------|----------------|
| 99（全GPU） | 275.6 | 38.5 |
| 0（纯CPU） | 33.6 | 27.1 |

**llama.cpp vs MLX 横向对比**

| 框架 | 模型 | Prompt t/s | Generation t/s | Peak Memory |
|------|------|-----------|----------------|-------------|
| llama.cpp | Llama-3.2-3B Q4_K_M | 275.6 | 38.5 | — |
| MLX | Llama-3.2-3B 4bit | 137.6 | 40.9 | 1.98 GB |
| llama.cpp | Mistral-7B Q4_K_M | 66.3 | 17.2 | — |
| MLX | Mistral-7B 4bit | 40.9 | 20.7 | 4.18 GB |

**关键结论**：Generation 持平（瓶颈是内存带宽）；Prefill llama.cpp 快 2x（手写 Metal kernel）；对话场景两者都行，长 prompt 场景 llama.cpp 更优。

### CLI 工具
```
python llm_cli.py "你的问题"
python llm_cli.py --task sentiment "The earnings beat expectations"
python llm_cli.py --task translate "今天天气很好"
python llm_cli.py --task summarize "..."
python llm_cli.py --model mistral "Explain KV cache"
```
支持 task：chat / sentiment / translate / summarize；支持 model：llama / mistral。Few-shot 例子用 multi-turn messages 格式注入。

---

## 项目 E：LoRA Fine-tune（已完成）✅

### 项目目录
```
~/ml-projects/lora-finetune/
├── mistral-qlora-sentiment-adapter/     # Vast.ai 训练产出，scp 到本地
│   ├── adapter_config.json              # LoRA 配置（r=16，target=q_proj+v_proj，task=SEQ_CLS）
│   └── adapter_model.safetensors        # LoRA A/B 矩阵 + score 分类头（26MB）
├── mistral-7b-sentiment-merged/         # Vast.ai 合并产出，scp 到本地
│   ├── config.json                      # 含 quantization_config（NF4）
│   ├── model.safetensors                # 合并后 NF4 权重（3.86GB）
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── mistral-7b-sentiment-gguf/
    └── mistral-7b-merged-q4km.gguf      # GGUF Q4_K_M 量化（4.1GB，CausalLM，无分类头，不可用于分类推理）
```

### Week 22：Colab 跑通 LoRA 训练流程 ✅

**环境**：Google Colab，Tesla T4 14.6GB，transformers 5.0.0，peft 0.18.1

**实验**：GPT-2（124M）+ SST-2 情感分类，5000条训练，val_acc **0.884**，adapter **1.2MB**（原模型 548MB 的 0.2%）

**LoRA 配置**：r=8，lora_alpha=16，target_modules=["c_attn"]，可训练参数 296,448（0.24%）

**关键踩坑**：
1. 500条数据不够收敛（val_acc 0.57，严重偏向 label=1）→ 扩到 5000 条后 0.884
2. `from_pretrained` 后 requires_grad 默认全 False，续训需 `enable_adapter_layers()`
3. 手动恢复 requires_grad 容易漏 score 层（score 不含 `lora_` 前缀）

### Week 23：Vast.ai QLoRA fine-tune Mistral-7B ✅

**环境**：Vast.ai RTX 3090 24GB，transformers 5.5.4，peft 0.19.1，bitsandbytes 0.49.2，trl 1.2.0

**数据集**：`zeroshot/twitter-financial-news-sentiment`，金融推文三分类（Bearish/Bullish/Neutral），训练集 9543 条，验证集 2388 条，类别不均衡（Neutral 占 64.7%）

**QLoRA 配置**：NF4量化，r=16，lora_alpha=32，target_modules=["q_proj", "v_proj"]，显存 3.9GB（原 BF16 需 14GB）

**训练结果**（3 epoch，batch=16，lr=2e-4）：

| Epoch | Train Loss | Val Loss | Val Acc |
|-------|-----------|---------|---------|
| 1 | 0.328 | 0.291 | 0.885 |
| 2 | 0.213 | 0.271 | 0.913 |
| 3 | 0.068 | 0.429 | **0.915** |

Per-label accuracy：Bearish 0.879，Bullish 0.859，Neutral 0.940

**关键踩坑**：
1. `financial_phrasebank` 在 datasets 4.x 不兼容 → 换用 `zeroshot/twitter-financial-news-sentiment`
2. 整体 acc 65% 基准（全猜 Neutral），必须看 per-label acc
3. val_loss 最优（epoch 2）和 accuracy 最优（epoch 3）可能不同，`metric_for_best_model` 配置决定取哪个

### Week 24：评估与 Base Model 对比 ✅

**环境**：Google Colab，pip 不指定版本（transformers 5.5.4，peft 0.19.1，bitsandbytes 0.49.2，accelerate 1.13.0）

**评估结果**（2388条 validation set，复现 Week 23）：

| | Overall | Bearish | Bullish | Neutral |
|---|---|---|---|---|
| val_acc | 0.9146 | 0.8790 (n=347) | 0.8589 (n=475) | 0.9393 (n=1566) |

**Confusion Matrix**：
```
预测 →      Bearish  Bullish  Neutral
真实 Bearish   305       5      37
真实 Bullish     5     408      62
真实 Neutral    47      48    1471
```

**错误模式分析**：
- 主要误差：情感类 → Neutral（Bearish 错误 88% 是漏判为 Neutral，Bullish 错误 91% 是漏判为 Neutral）
- Bearish ↔ Bullish 几乎不混淆（只有 10 条）→ 情感方向判断准，困难在于"有没有情感"
- 根本原因：模型对**隐性情感**识别弱（需要领域知识推理的负面信号、事实陈述型正面）

**Base Model Zero-shot 对比**：

| Text | Base Model | Fine-tuned | 正确答案 |
|------|-----------|-----------|---------|
| $AAPL hits all-time high... | Bullish | Bullish | Bullish ✅ |
| Company reports massive losses... | Bearish | Bearish | Bearish ✅ |
| Federal Reserve holds steady | Bullish ❌ | Neutral | Neutral |
| No, The Fed Won't Save The Market | Bullish ❌ | Neutral | Bearish |
| $KTOS awarded $39M contract | Bullish | Neutral | Bullish |

Base model 有明显 **Bullish bias**（遇到不确定文本就猜 Bullish），fine-tune 消除了这个 bias，对模糊文本保守地判 Neutral。

### Week 25：合并权重 + 本地推理验证 ✅

**流程**：Vast.ai RTX 3090 上用 `AutoModelForSequenceClassification` + `merge_and_unload()` 合并 adapter，scp 到本地，本地用 bitsandbytes MPS 加载推理。

**关键发现**：
- `AutoModelForSequenceClassification` 的 `merge_and_unload()` 保存结果仍是 **NF4 格式**（非 FP16），和 `AutoModelForCausalLM` 行为不同；合并后文件 **3.86GB**（非预期的 14GB）
- `AutoModelForCausalLM` 合并结果是 FP16（14.5GB），可转 GGUF，但**丢失分类头**，无法用于分类推理
- bitsandbytes 0.49.2 已支持 MPS（slow implementation），可在 M2 本地加载 NF4 模型
- 加载已量化的 safetensors 时，传同样的 `BitsAndBytesConfig` 不会重复量化，bitsandbytes 会自动检测并直接复用已有量化格式
- GGUF Q4_K_M 格式只适用于 CausalLM 生成模型，llama.cpp 不支持 SeqCLS 分类头，**分类任务不能走 llama.cpp 推理路径**

**本地推理验证**（MPS，NF4，transformers + bitsandbytes）：

| Text | 本地结果 | 正确答案 |
|------|---------|---------|
| Federal Reserve holds interest rates steady | Neutral | Neutral ✅ |
| Company reports massive losses this quarter | Bearish | Bearish ✅ |
| $AAPL hits all-time high after earnings beat | Bullish | Bullish ✅ |
| No, The Fed Won't Save The Market | Neutral | Bearish ❌ |
| $KTOS awarded $39M contract | Neutral | Bullish ❌ |

结果与 Week 24 Colab 评估完全一致，本地推理验证通过。

**关键踩坑**：
1. **Jupyter kernel 和系统 Python 路径不同**：Vast.ai 实例 notebook kernel 在 `/venv/main/bin/python`，pip 需用 `!/venv/main/bin/pip install`
2. **hf CLI 已改名**：新版 `huggingface-cli` 已废弃，改用 `hf` 命令
3. **llama.cpp binary 在 `build/bin/` 下**：不在根目录，路径为 `~/llama.cpp/build/bin/llama-quantize`
4. **本地 M2 不能直接合并**：FP16 Mistral-7B 需 14GB，M2 16GB 硬空闲仅 ~2GB，macOS memory_pressure 显示的 80% free 包含 inactive/compressed 页面，实际可用远低于此
5. **SeqCLS merge 结果无法用 llama.cpp 推理**：必须用 transformers + bitsandbytes 走分类头路径

---

## 项目 F：模型量化实验（已完成）✅

### Week 26：bitsandbytes INT8 量化 + Perplexity 测量 ✅

**环境**：Google Colab，Tesla T4 15.6GB，torch 2.10.0+cu128，transformers 5.0.0，bitsandbytes 0.49.2

**实验设置**：GPT-2（124M），WikiText-2 测试集，前 50 个 chunk（51,200 tokens），per-chunk 计算平均 NLL 再求 exp

**实验结果**：

| | 显存 | PPL | 耗时 |
|---|---|---|---|
| FP32 | 0.51 GB | 32.35 | 4.6s |
| INT8 | 0.27 GB | 32.50 | 4.9s |
| **差值** | **-47%** | **+0.15** | **+0.3s** |

**关键结论**：INT8 量化显存减半，精度损失 < 0.5%，T4 上速度持平（小模型量化 overhead 盖过 INT8 tensor core 收益，7B+ 才能看到明显提速）。

**关键踩坑**：
- transformers 5.0 不能直接传 `load_in_8bit=True` 给 `from_pretrained`，必须用 `BitsAndBytesConfig(load_in_8bit=True)` 再传 `quantization_config=`

### Week 27：NF4 INT4 量化 + 显存解剖实验 ✅

**环境**：Google Colab，Tesla T4 15.6GB，torch 2.10.0+cu128，transformers 5.5.4，bitsandbytes 0.49.2，gptqmodel 6.0.3

**实验结果**（同 session 内对比，FP32 baseline 一致）：

| | 显存 | PPL |
|---|---|---|
| FP32 | 0.476 GB | 27.70 |
| NF4 INT4 | 0.697 GB | 29.90 |
| **差值** | **+46.5%** | **+2.20（7.9%）** |

Week 26 独立实验（不同 session，不可混用）：FP32 PPL 32.35 → INT8 PPL 32.50，显存 -47%

**显存反增的真正原因（逐层排查得出）**：

最初假设是 scale factor overhead 导致小模型显存反增，但数据否定了这个假设——scale overhead 是固定比例，对大小模型影响相同。

真正原因通过逐层解剖找到：
- embedding 层（wte + lm_head）未被 bitsandbytes 量化，保持 FP32
- GPT-2 的 wte 和 lm_head 原本 weight tying（共享权重），量化后 tie 断开，变成两份独立 FP32 拷贝：**294 MB**
- transformer 层量化权重（uint8）只有 **40 MB**，压缩非常成功
- GPT-2 embedding 占总参数 **31%**（50257×768 / 124M），比例异常高
- 7B 模型 embedding 占比仅 ~1%，INT4 才能看到整体显存大幅下降

**GPTQ 环境问题**：gptqmodel 6.0.3 与 Colab torch 2.10 / CUDA 12.8 不兼容（需 torch >= 2.11），GPTQ 实验跳过，算法原理已掌握。

**关键踩坑**：
- Colab 新 session 包依赖混乱时，Restart session 清进程缓存；彻底清环境需 Disconnect and delete runtime
- Colab 存储三层：内存（restart 丢）/ 本地磁盘（disconnect 丢，pip 包和模型缓存在此）/ Google Drive（永久）

### Week 28：ONNX 导出 + onnxruntime 推理 ✅

**环境**：本地 M2，torch 2.11.0，onnx 1.21.0，onnxruntime 1.24.4，transformers 5.4.0

**实验设置**：GPT-2（124M），导出单步 forward（不带 KV cache），greedy decoding 生成 50 个 token，3 次平均

**导出过程**：
- torch 2.11 新版 ONNX exporter 用 `torch.export` 做 tracing，不支持 `DynamicCache` 类型输出，需用 wrapper 屏蔽
- 用 `GPT2LogitsOnly(nn.Module)` wrapper 调用 `model(input_ids, use_cache=False)`，只返回 logits tensor
- opset 14 自动升级到 18（torch 2.11 exporter 只有 opset 18 实现，LayerNormalization 在 opset 14 之前不存在）
- 大权重自动存为 external data：`gpt2_no_cache.onnx`（1.3 MB 图结构）+ `gpt2_no_cache.onnx.data`（474.7 MB 权重）

**数值验证**：PT vs ORT 最大绝对误差 0.000366，平均误差 0.000103，通过（误差来源：opset 18 LayerNorm 实现差异）

**KV cache 验证实验**（patch model.forward 记录每步 input_len）：
- PyTorch `.generate()` 默认开启 KV cache：prefill 步处理 4 tokens，之后每步 input_len=1
- `DynamicCache` 是 in-place 更新对象，`out.past_key_values` 和传入的 `past` 是同一对象
- cache tensor shape：`(batch=1, n_heads=12, seq_len, head_dim=64)`，seq_len 每步 +1

**速度对比结果**：

| | 平均耗时 | tokens/sec | 备注 |
|---|---|---|---|
| PyTorch `.generate()` | 0.908s | 55.1 t/s | 有 KV cache，每步算 1 token |
| ONNX（无 cache） | 1.416s | 35.3 t/s | 无 cache，每步重算全序列 |
| **比值** | | **0.64x** | ONNX 慢不是 runtime 问题 |

**核心结论**：ONNX 版慢 36% 的根本原因是算法层面——无 cache 每步计算量 O(n²)，而非 onnxruntime 本身性能差。框架选择带来的差距远小于算法设计带来的差距。

**带 KV cache 的 ONNX 导出探索**（未完成，了解原理）：
- 需把 cache 显式作为输入输出传递（25 输入 + 25 输出），不能依赖 in-place 更新
- `DynamicCache` 不支持下标访问，需通过 `past.layers[i].keys / .values` 取 tensor
- torch 2.11 `DynamicCache` 为 in-place 更新，legacy tuple 格式兼容性待验证
- 工程复杂度高，本周以理解原理为主，不强求跑通

**文件位置**：`~/ml-projects/quantization-exp/gpt2_no_cache.onnx` + `gpt2_no_cache.onnx.data`

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
- ONNX dynamic_axes：让指定维度（如 batch size、股票池大小）在推理时可变
- SageMaker endpoint 内部：TorchServe + inference.py，不是 Streamlit
- 梯度累积：多步 loss 累加后统一 backward，等效增大 batch size，不等于真正的 batched tensor 计算
- Early stopping：val loss 连续 N 步不改善则停止，配合保存 best_state 使用

### LLM 推理专项（Week 19–20）
- **GGUF 格式**：llama.cpp 自定义格式，把权重、tokenizer、元数据打包成单文件，便于分发和量化
- **Q4_K_M**：INT4 量化变体，K 表示按 block 量化（更精确），M 表示中等 block size，精度/速度平衡最佳；量化时读取模型结构，对敏感层（ffn_down、attn_v）用 Q6_K，大矩阵用 Q4_K，极小层保持 F32
- **-ngl（n_gpu_layers）**：控制 offload 到 Metal GPU 的层数，M2 上始终用 99（全部 offload）
- **Prefill vs Generation**：prefill（处理输入）是并行矩阵乘，GPU 优势大；generation（逐 token 生成）受内存带宽限制，GPU 优势相对小
- **unified memory**：M2 CPU/GPU 共享同一物理内存，无需 CPU→GPU 数据拷贝，这是 Apple Silicon 推理效率高的核心原因
- **tokens/sec**：LLM 推理速度标准指标；人类阅读 ~5–8 t/s，>15 t/s 即流畅对话体验
- **MLX**：Apple 专为 M 系列设计的机器学习框架，Python-native，lazy evaluation，直接操作 unified memory
- **MLX vs llama.cpp 架构差异**：llama.cpp 是 C++ + 手写 Metal kernel（prefill 快）；MLX 是 Python + 通用 Metal kernel（generation 持平，prefill 慢）
- **mlx-community**：HuggingFace 上的 MLX 模型转换社区，提供现成的 `.safetensors` 格式 4bit 模型，无需自己转换
- **测速注意**：prompt 需 40+ tokens 才能得到有代表性的 Prompt t/s，短 prompt 数据会严重偏低

### LLM Prompting 专项（Week 21）
- **Few-shot prompting**：在 prompt 里提供输入输出例子，让模型从 context 中理解任务格式，无需 fine-tune
- **Multi-turn messages 格式是 few-shot 的正确写法**：每个例子用 user/assistant 交替表达，模型会模仿 assistant 的回答风格和长度；把例子堆在单条 user 消息里会导致模型续写
- **apply_chat_template**：将 messages 列表转换为模型特定的对话格式字符串（含特殊 token），不同模型格式不同，必须用 tokenizer 自带的方法
- **few-shot 效果**：① 格式锁定（大小写、标点等跟例子一致）② 边界案例有参照（模糊输入有了可类比的例子）

### LoRA / PEFT 专项（Week 22–25）
- **LoRA 数学原理**：在原始权重 W₀ 旁边加两个低秩矩阵 A（d×r）和 B（r×k），正向传播计算 W₀x + BAx，训练时只更新 A 和 B，W₀ 冻结。r 远小于 d/k，参数量从 d×k 降到 r×(d+k)
- **rank r 的含义**：低秩矩阵的秩，控制可训练参数量和表达能力的权衡；r=8 是常用起点
- **lora_alpha**：缩放系数，实际学习率效果是 `lr × alpha/r`；alpha=16, r=8 → 缩放2x
- **target_modules**：指定在哪些层插入 LoRA；attention 的 QKV 矩阵（`c_attn`、`q_proj`、`v_proj`）是最常见选择
- **get_peft_model()**：训练前调用，插入随机初始化的 LoRA 层，自动设 requires_grad=True
- **PeftModel.from_pretrained()**：加载已训练的 adapter，默认推理模式（requires_grad=False）；续训需调用 `enable_adapter_layers()`
- **adapter 文件结构**：`adapter_config.json`（配置）+ `adapter_model.safetensors`（权重），文件名硬编码，`save_pretrained` / `from_pretrained` 成对使用
- **adapter 体积**：只保存 A/B 矩阵，GPT-2 124M 的 adapter 仅 1.2MB（原模型 548MB 的 0.2%）
- **QLoRA**：在 LoRA 基础上把 base model 用 NF4 量化加载（4bit），大幅降低显存需求；Mistral-7B BF16 需 14GB，NF4 量化后只需 3.9GB
- **NF4（NormalFloat4）**：专为正态分布权重设计的 4bit 量化，比 INT4 精度更高；大模型权重分布接近正态，NF4 是最优匹配
- **double quantization**：连量化的 scale factor 也再量化一次，额外节省 ~0.4GB 显存
- **`bnb_4bit_compute_dtype`**：NF4 权重做矩阵乘法前临时反量化的目标格式（FP16 或 BF16），不影响存储格式
- **LayerNorm 不量化的原因**：LayerNorm 参数对数值稳定性极敏感（涉及除法），量化误差会逐层累积放大；其参数量极少（<0.1%），保持 BF16 对显存影响可忽略
- **训练中 dtype 分工**：存储 NF4 → 计算 FP16（compute_dtype）→ loss/梯度 BF16（bf16=True）→ optimizer 更新 FP32（防 underflow）
- **为什么只选 Q+V**：Q 控制 attend 哪里，V 控制取什么内容，覆盖 attention 两个独立变化方向；K 与 Q 高度相关，加 K 收益边际递减但参数量增加 1/3
- **Vast.ai 使用模式**：按小时付费，训练完即销毁实例，不需要 conda 环境隔离；唯一要做的是训练结束后把 adapter 文件 scp 到本地
- **modules_to_save vs LoRA 层的加载差异**：modules_to_save 的层保存完整权重，加载后通过 ModulesToSaveWrapper 旁路 base model 同名层；LoRA 层保存 A/B 矩阵，forward 时做加法
- **评估不只是看 accuracy**：confusion matrix 看错在哪，人工案例看为什么错，base model 对比看 fine-tune 带来了什么；数据不均衡时 per-label accuracy 是必须的
- **merge_and_unload() 的 task type 差异**：CausalLM 合并结果是 FP16（可转 GGUF）；SeqCLS 合并结果仍是 NF4（保留量化格式），两者行为不同
- **SeqCLS 不能走 llama.cpp**：llama.cpp 只支持生成式模型，分类头（score 层）在 GGUF 转换时丢失；SeqCLS 模型必须用 transformers + bitsandbytes 加载推理
- **bitsandbytes MPS 支持**：0.49.2 版本已支持 MPS（slow implementation），可在 M2 本地加载 NF4 模型做推理；加载已量化的 safetensors 时传同样的 BitsAndBytesConfig 不会重复量化

### ONNX 导出专项（Week 28）
- **生成模型 ONNX 导出难点**：GPT-2 forward 默认返回 `(logits, DynamicCache)`，torch 2.11 新版 exporter 不认识 `DynamicCache` 类型，必须用 wrapper 只返回 logits tensor；用 `use_cache=False` 让 forward 不产生 cache 输出
- **external data 格式**：大模型权重自动分离成 `.onnx`（图结构）+ `.onnx.data`（权重），两个文件必须在同一目录，加载时 onnxruntime 自动找到 `.data` 文件
- **opset 版本自动升级**：torch 2.11 exporter 只有 opset 18 实现，指定 opset 14 会自动升级；`LayerNormalization` 在 opset 17 才引入，无法降回 14
- **KV cache 验证**：patch `model.forward` 记录每步 input_len，可直接观察 cache 是否生效；有 cache 时每步 input_len=1，无 cache 时每步递增
- **DynamicCache 是 in-place 更新**：`out.past_key_values` 和传入的 `past` 是同一对象，调用 `model()` 后 `past` 内容已被修改；cache tensor 通过 `past.layers[i].keys / .values` 访问
- **cache tensor shape**：`(batch, n_heads, seq_len, head_dim)`，GPT-2 为 `(1, 12, seq_len, 64)`，seq_len 每生成一个 token +1
- **无 cache ONNX 的性能陷阱**：无 KV cache 时每步需重算所有历史 token，计算量 O(n²)；这是算法问题，不是 onnxruntime 本身性能差；框架选择带来的差距远小于算法设计带来的差距
- **带 cache 的 ONNX 导出**：需把 cache 显式作为输入输出（GPT-2 需 25 输入 + 25 输出），prefill 步用 shape=(1,12,0,64) 的空 tensor 占位；工程复杂，实际部署一般用 HuggingFace Optimum 库处理

### 模型量化专项（Week 26–27）
- **perplexity（PPL）**：语言模型精度指标，`PPL = exp(平均 NLL)`，越低越好；WikiText-2 上 GPT-2 FP32 基准约 29–33（取决于 chunk 数量）
- **INT8 量化实测**：GPT-2 显存 0.51GB→0.27GB（节省47%），PPL 32.35→32.50（+0.15，<0.5%损失）
- **weight vs 激活值**：weight 是固定参数，加载时量化一次存好；激活值（activation）是推理时每层动态产生的中间结果，每次推理现场量化
- **INT8 计算流**：weight 加载时量化成 INT8 存储；激活值推理时量化成 INT8；INT8×INT8 → INT32（硬件 tensor core，防溢出）→ × scale_W × scale_x → FP32 输出
- **为什么累加结果是 INT32**：INT8×INT8 最大值 127×127=16,129，累加 768 次后达 ~12M，超出 INT8 范围，必须用 INT32 承接
- **scale**：float32 标量，per-channel 粒度（每个输出 channel 一个），absmax 量化：`scale = max(|x|) / 127`；weight 的 scale 离线算好存着，激活值的 scale 推理时实时算
- **per-tensor vs per-channel vs per-group**：粒度越细精度越好，bitsandbytes INT8 默认 per-channel，GPTQ 用 per-group（W27 会涉及）
- **NF4 vs INT8 的本质区别**：NF4 只是压缩存储，计算前完整反量化回 FP16（计算是 FP16 的）；INT8 是计算本身也降精度，利用硬件整数计算单元（真正的 INT8 计算）
- **bitsandbytes INT8 的 outlier 处理**：权重和激活值中少量离群值单独用 FP16 计算，其余 99%+ 用 INT8 矩阵乘，最后合并结果（LLM.int8() 算法）
- **量化在哪里发生**：bitsandbytes `load_in_8bit` 是加载时实时量化（每次加载都算一遍）；GGUF 是离线预先量化好再分发（加载即用）；GPTQ 是用校准数据离线量化（精度最好，W27 涉及）
- **小模型量化速度不提升的原因**：量化 overhead（实时算激活值 scale）盖过 INT8 tensor core 收益；7B+ 才能明显看到提速
- **transformers 5.0 写法**：`load_in_8bit` 不能直接传给 `from_pretrained`，必须通过 `BitsAndBytesConfig(load_in_8bit=True)` 再传 `quantization_config=`
- **`dtype=torch.float16` 的作用**：指定激活值基础精度为 FP16，消除 bitsandbytes 的 FP32→FP16 转换 warning；大模型加载时必须指定，否则 FP32 weight 撑爆显存
- **RTN（Round-To-Nearest）**：最简单的量化方式，直接把权重映射到最近的量化值；bitsandbytes NF4 使用此方式，实现简单但精度相对较差
- **GPTQ**：用校准数据计算 Hessian 矩阵（二阶梯度信息），量化某权重时把误差补偿到同行其他权重，逐层最小化输出误差；同等 bit 数下精度远优于 RTN，接近 INT8 水平
- **weight tying**：GPT-2 的 wte（输入 embedding）和 lm_head（输出投影）共享同一权重矩阵，节省参数；bitsandbytes 量化后 tie 断开，变成两份独立拷贝，导致 embedding 显存翻倍
- **量化不一定节省显存**：embedding 等大层若不被量化，其开销可以抵消甚至超过 transformer 层的压缩收益；GPT-2 embedding 占总参数 31%，是显存反增的根本原因
- **embedding 占比与模型大小的关系**：GPT-2（124M）embedding 占 31%；7B 模型 embedding 占比 ~1%，量化收益才能充分体现；量化主要压缩 transformer 层的 Linear 权重

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
- **MPS**：小模型推理用 CPU 更稳定；bitsandbytes NF4 模型用 MPS 可以（0.49.2+）
- **OpenMP 冲突**：永久方案：写入 `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`
- **Jupyter crash 定位**：terminal 里 `python -c` 复现，segfault 会明确打印
- **M2 16GB 内存限制**：7B INT4 可推理（~4–5GB），7B FP16 不行（需14GB+），LoRA fine-tune 不行（训练态3–4倍显存）；`memory_pressure` 显示的 free% 包含 inactive/compressed，实际硬空闲看 `Pages free × 16KB`
- **llama.cpp Metal**：`recommendedMaxWorkingSetSize = 12.4GB`，始终用 `-ngl 99` 全 GPU offload

### Colab / Vast.ai 环境问题专项
- **pip 版本兼容**：不手动指定版本，直接 `pip install transformers peft bitsandbytes datasets accelerate trl`，让 pip 自动解决；手动锁定版本容易引入新的不兼容
- **OOM 恢复**：显存未完全释放时重试会 OOM，必须 Runtime → Restart session 后重跑
- **模型加载后 pad_token_id**：`model.config.pad_token_id = tokenizer.pad_token_id` 在加载后手动设置
- **BF16 tensor → numpy**：需先 `.float()` 转换，numpy 不支持 BF16
- **Vast.ai notebook kernel 路径**：在 `/venv/main/bin/python`，pip 需用 `!/venv/main/bin/pip install`
- **hf CLI 已改名**：`huggingface-cli` 废弃，改用 `hf download`
- **llama.cpp 编译后 binary 路径**：在 `build/bin/` 下，不在根目录

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

**当前进度**：Phase 2 项目 G，Week 29，尚未开始。

**已完成**：
- 项目 B 账单分析工具（PyTorch 分类器 + Autoencoder 异常检测 + Streamlit + SageMaker 部署 + ONNX 导出）
- 项目 A 影视推荐引擎（Matrix Factorization → NCF → Two-Tower → Faiss → FastAPI → ONNX → Streamlit，Week 7–12）
- 项目 C 股票/基金自选池助手（时序特征工程 + RankNet + ListNet + APScheduler + ONNX + Streamlit，Week 13–18）
- 项目 D 本地 LLM 推理引擎（llama.cpp + MLX 推理对比 + few-shot prompting + CLI 工具，Week 19–21）
- 项目 E LoRA Fine-tune（Week 22–25）：GPT-2 LoRA → Mistral-7B QLoRA 金融情感分类（val_acc 0.9146）→ 合并权重 → 本地 MPS NF4 推理验证
- 项目 F Week 26：bitsandbytes INT8 量化 + perplexity 测量（GPT-2，WikiText-2，FP32 PPL 32.35 → INT8 PPL 32.50，显存节省 47%）
- 项目 F Week 27：NF4 INT4 量化 + 显存解剖（FP32 PPL 27.70 → NF4 PPL 29.90，显存反增 46.5%；原因：embedding 未量化 + weight tying 断开）
- 项目 F Week 28：ONNX 导出 + onnxruntime 推理（GPT-2，无 cache 版 35.3 t/s vs PyTorch 有 cache 55.1 t/s；验证了 KV cache 对生成速度的关键作用）

**环境**：MacBook Apple Silicon M2 16GB，conda，mlenv 环境
- faiss 必须用 conda-forge 版：`conda install -c conda-forge faiss-cpu`
- KMP_DUPLICATE_LIB_OK=TRUE 已永久写入 conda activate.d
- M2 16GB：7B INT4 可推理，7B FP16 / 任何 LoRA fine-tune 不可行
- llama.cpp via Homebrew b8680，模型在 `~/ml-projects/llm-inference/models/`
- mlx-lm 0.31.2 已安装
- bitsandbytes 0.49.2 支持 MPS，可本地加载 NF4 模型
- 合并后模型在本地：`~/ml-projects/lora-finetune/mistral-7b-sentiment-merged/`（NF4，3.86GB）

**注意事项**：
- 暂不使用 AWS infrastructure
- 教学风格：每次只给一个 cell 的代码，跑完看结果再继续，遇到问题用数据验证而不是猜测

完整学习进度文档见附件 `ml_learning_progress_v21.md`。
