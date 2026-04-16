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
- 项目 E：LoRA Fine-tune（Week 22–25）⬅ 进行中（Week 22 已完成）
- 项目 F：模型量化实验（Week 26–28）
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

### 项目 E：LoRA Fine-tune（Week 22–25，Colab + Vast.ai）

**目标**：用 LoRA 在特定任务上 fine-tune 7B 模型，理解参数高效微调的原理和工程实现。

| 周 | 内容 | 平台 | 状态 |
|---|---|---|---|
| W22 | Colab 上跑通 LoRA 训练流程（小模型 + 小数据验证逻辑） | Colab | ✅ |
| W23 | Vast.ai 上 QLoRA fine-tune Mistral-7B（金融问答数据集） | Vast.ai RTX 4090 | |
| W24 | 评估：ROUGE / 人工对比，理解 fine-tune 前后差异 | Colab / 本地 | |
| W25 | 导出合并权重，本地 INT4 推理验证 | 本地 M2 | |

**技术点**：LoRA 数学原理（低秩分解 W = W₀ + BA）、QLoRA（NF4量化+LoRA）、bitsandbytes、PEFT 库、Hugging Face Trainer、rank / alpha 超参数含义。

---

### 项目 F：模型量化实验（Week 26–28，本地 M2 + Colab）

**目标**：动手做 INT8 / INT4 量化，用数据理解精度和速度的 tradeoff。

| 周 | 内容 | 平台 |
|---|---|---|
| W26 | bitsandbytes INT8 量化，perplexity 测量（量化损失量化） | Colab |
| W27 | GPTQ INT4 量化，和 llama.cpp GGUF 做横向对比 | Colab |
| W28 | ONNX 导出 + onnxruntime 推理，和 PyTorch 对比速度 | 本地 M2 |

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
- 项目 E：~$10–15（RTX 4090，约20–30小时）
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

### 关键踩坑
1. **近零方差维度导致标准化爆炸**：BERT proj 输出 16 维里有 11 维几乎全为 0，StandardScaler 除以极小 std 后产生天文数字。解决：用 VarianceThreshold(1e-6) 过滤，保留 8 维。
2. **ONNX 导出必须在 eval() 模式**：train 模式下 Dropout 是随机的，导出的图不确定。
3. **SageMaker 内部是 TorchServe**：不是 Streamlit，inference.py 的入口函数签名必须符合 TorchServe 规范。

### 最终指标
- val acc: 0.98–0.99
- PyTorch vs ONNX 最大误差：0.000005
- 推理速度：PyTorch 0.021ms → ONNX Runtime 0.007ms，加速 3.21x

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

### Week 19：llama.cpp 推理基准 ✅

**环境**：llama.cpp via Homebrew b8680，Apple M2 16GB，Metal backend（MTLGPUFamilyApple8）

**GPU vs CPU 对比（Llama-3.2-3B Q4_K_M，-n 100）**

| ngl | Prompt t/s | Generation t/s |
|-----|-----------|----------------|
| 99（全GPU） | 275.6 | 38.5 |
| 0（纯CPU） | 33.6 | 27.1 |

**模型对比（ngl 99）**

| 模型 | 大小 | Prompt t/s | Generation t/s |
|------|------|-----------|----------------|
| Llama-3.2-3B Q4_K_M | 1.9GB | 275.6 | 38.5 |
| Mistral-7B Q4_K_M | 4.1GB | 66.3 | 17.2 |

**关键结论**：
- 始终用 `-ngl 99`：Prompt eval 快 8x，Generation 快 1.4x
- Generation 速度与参数量近似成反比（3B/7B ≈ 38.5/17.2 ≈ 2.2x）
- prefill 瓶颈是算力（GPU 并行优势大），generation 瓶颈是内存带宽（unified memory CPU/GPU 共享，差距小）
- 7B INT4 在 M2 16GB 可用，17 t/s 远超人类阅读速度（~5–8 t/s）
- `recommendedMaxWorkingSetSize = 12.4GB`，7B INT4（~4.1GB）+ KV cache 余量充足

**注意事项**：
- 此版本 llama-cli 默认进入交互模式，`-p` 只是预填第一条消息，看完数字用 `/exit` 退出
- `--no-interactive` 此版本不支持

### Week 20：Apple MLX 推理 + llama.cpp vs MLX 横向对比 ✅

**环境**：mlx-lm 0.31.2，Apple M2 16GB，Metal GPU (Device(gpu, 0))

**llama.cpp vs MLX 横向对比（Llama-3.2-3B Q4/4bit，长 prompt ~66 tokens，生成 100 tokens）**

| 框架 | Prompt t/s | Generation t/s | Peak Memory |
|------|-----------|----------------|-------------|
| llama.cpp Q4_K_M | 275.6 | 38.5 | — |
| MLX 4bit | 137.6 | 40.9 | 1.98 GB |

**llama.cpp vs MLX 横向对比（Mistral-7B Q4/4bit，长 prompt ~40 tokens，生成 100 tokens）**

| 框架 | Prompt t/s | Generation t/s | Peak Memory |
|------|-----------|----------------|-------------|
| llama.cpp Q4_K_M | 66.3 | 17.2 | — |
| MLX 4bit | 40.9 | 20.7 | 4.18 GB |

**关键结论**：
- **Generation 持平**：两者都用 Metal GPU，generation 瓶颈是内存带宽（unified memory 决定上限），软件优化空间小
- **Prefill llama.cpp 更快（2x）**：prefill 是算力密集型（大矩阵乘），llama.cpp 有针对 Metal 手写的 INT4 矩阵乘法 kernel；MLX 走通用 Metal kernel
- **7B 的 prefill 差距比 3B 小**（1.6x vs 2x）：模型越大，kernel 优化的相对优势在收窄
- **选型建议**：对话场景（generation 为主）用哪个都行；长文档 / 长 prompt 场景 llama.cpp prefill 优势明显

**MLX 使用注意**：
- 模型格式是 `.safetensors`，不是 GGUF，从 `mlx-community` HuggingFace 组织下载
- 首次运行自动下载并缓存到 `~/.cache/huggingface/`，后续加载直接用缓存
- 短 prompt（<20 tokens）的 Prompt t/s 数据不可信，需用 40+ tokens 的 prompt 测速

### Week 21：Few-shot Prompting + 本地 CLI 工具 ✅

**核心发现**：few-shot 例子必须用 **multi-turn messages 格式**，不能堆在单条 user 消息里。

| 写法 | 问题 | 结果 |
|------|------|------|
| 例子全堆在一条 user 消息 | 模型把 prompt 当"列表续写"任务 | 输出多个标签（positive / negative / neutral...） |
| user/assistant 交替 multi-turn | 模型知道 assistant 说完就停 | 输出单个正确标签，max_tokens=200 也不续写 |

**CLI 工具**：`~/ml-projects/llm-inference/cli/llm_cli.py`

```
# 用法
python llm_cli.py "你的问题"
python llm_cli.py --task sentiment "The earnings beat expectations"
python llm_cli.py --task translate "今天天气很好"
python llm_cli.py --task summarize "..."
python llm_cli.py --model mistral "Explain KV cache"
```

支持 task：chat / sentiment / translate / summarize
支持 model：llama（Llama-3.2-3B）/ mistral（Mistral-7B）
Few-shot 例子用 multi-turn messages 格式注入，输出干净，不续写。

---

## 项目 E：LoRA Fine-tune（进行中）

### Week 22：Colab 跑通 LoRA 训练流程 ✅

**环境**：Google Colab，Tesla T4 14.6GB，PyTorch 2.10.0+cu128，transformers 5.0.0，peft 0.18.1

**实验设置**：GPT-2（124M）+ SST-2 情感分类，5000条训练数据，验证集872条

**LoRA 配置**：
```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn"],
)
```

**参数量对比**：

| | 参数量 |
|---|---|
| GPT-2 原始总参数 | 124,441,344（124.4M） |
| LoRA 后总参数 | 124,737,792（124.7M） |
| **可训练参数** | **296,448（0.24%）** |

**LoRA 数学验证**（以第0层为例）：
- 原始 `c_attn.weight`：768×2304 = 1,769,472（Q/K/V 拼在一起）
- `lora_A`：768×8 = 6,144
- `lora_B`：8×2304 = 18,432
- 12层 × 24,576 + score层 1,536 = 296,448 ✅

**训练结果**（5 epoch，batch=32，lr=2e-4）：

| Epoch | Loss | Val Acc |
|-------|------|---------|
| 1 | 0.6320 | 0.845 |
| 2 | 0.3905 | 0.870 |
| 3 | 0.3453 | 0.869 |
| 4 | 0.3362 | **0.884** |
| 5 | 0.3020 | 0.842 |

Best val_acc = **0.884**（Epoch 4），Epoch 5 轻微过拟合。

**保存的文件**（`/content/gpt2-lora-sst2/`）：

| 文件 | 大小 | 说明 |
|------|------|------|
| adapter_model.safetensors | 1161.1 KB | A/B 矩阵权重 |
| adapter_config.json | 0.9 KB | LoRA 配置（r、alpha、target_modules 等） |
| README.md | 5.0 KB | PEFT 自动生成 |

原始模型 548MB → LoRA adapter **1.2MB**，压缩到 0.2%。

**关键概念**：

`get_peft_model()` vs `PeftModel.from_pretrained()` 的区别：

| | `get_peft_model()` | `PeftModel.from_pretrained()` |
|---|---|---|
| 用途 | 训练前，插入随机初始化的 LoRA 层 | 推理/续训时，插入已训练好的 LoRA 权重 |
| requires_grad | 自动设为 True | 默认 False（推理模式） |
| 使用时机 | fine-tune 开始前 | fine-tune 结束后 |

**`from_pretrained` 后如何恢复训练**：
```python
# 方法一：手动遍历（注意不要漏 score 层）
for name, param in model.named_parameters():
    if "lora_" in name or "score" in name:
        param.requires_grad = True

# 方法二：推荐写法
model.enable_adapter_layers()
```

**`PeftModel.from_pretrained` 如何知道读哪个文件**：固定读 `adapter_config.json`（含 `peft_type: "LORA"`）和 `adapter_model.safetensors`，文件名是硬编码约定，不靠猜测。

**关键踩坑**：
1. **500条数据 + 3 epoch 不够收敛**：val_acc 0.57，模型严重偏向预测 label=1（预测分布 15:85，真实分布 48:52）。扩到 5000 条 + 5 epoch 后 val_acc 0.884。
2. **GPT-2 分类用最后一个 token 的 hidden state**：句子长度不一时分类信号弱，数据量少时尤其明显。
3. **`from_pretrained` 后 requires_grad 默认全为 False**：要继续训练必须手动恢复，用 `enable_adapter_layers()` 最安全。
4. **手动恢复 requires_grad 容易漏 score 层**：`score.weight` 不含 `lora_` 前缀，只过滤 `lora_` 会少 1,536 个参数（296,448 → 294,912）。

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
- **Q4_K_M**：INT4 量化变体，K 表示按 block 量化（更精确），M 表示中等 block size，精度/速度平衡最佳
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

### LoRA / PEFT 专项（Week 22）
- **LoRA 数学原理**：在原始权重 W₀ 旁边加两个低秩矩阵 A（d×r）和 B（r×k），正向传播计算 W₀x + BAx，训练时只更新 A 和 B，W₀ 冻结。r 远小于 d/k，参数量从 d×k 降到 r×(d+k)
- **rank r 的含义**：低秩矩阵的秩，控制可训练参数量和表达能力的权衡；r=8 是常用起点
- **lora_alpha**：缩放系数，实际学习率效果是 `lr × alpha/r`；alpha=16, r=8 → 缩放2x
- **target_modules**：指定在哪些层插入 LoRA；attention 的 QKV 矩阵（`c_attn`、`q_proj`、`v_proj`）是最常见选择
- **get_peft_model()**：训练前调用，插入随机初始化的 LoRA 层，自动设 requires_grad=True
- **PeftModel.from_pretrained()**：加载已训练的 adapter，默认推理模式（requires_grad=False）；续训需调用 `enable_adapter_layers()`
- **adapter 文件结构**：`adapter_config.json`（配置）+ `adapter_model.safetensors`（权重），文件名硬编码，`save_pretrained` / `from_pretrained` 成对使用
- **adapter 体积**：只保存 A/B 矩阵，GPT-2 124M 的 adapter 仅 1.2MB（原模型 548MB 的 0.2%）
- **QLoRA**：在 LoRA 基础上把 base model 用 NF4 量化加载（4bit），大幅降低显存需求，Week 23 正式使用

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
- **M2 16GB 内存限制**：7B INT4 可推理（~4–5GB），7B FP16 不行（需14GB+），LoRA fine-tune 不行（训练态3–4倍显存）
- **llama.cpp Metal**：`recommendedMaxWorkingSetSize = 12.4GB`，始终用 `-ngl 99` 全 GPU offload

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

**当前进度**：Phase 2 项目 E，Week 23，尚未开始。

**已完成**：
- 项目 B 账单分析工具（PyTorch 分类器 + Autoencoder 异常检测 + Streamlit + SageMaker 部署 + ONNX 导出）
- 项目 A 影视推荐引擎（Matrix Factorization → NCF → Two-Tower → Faiss → FastAPI → ONNX → Streamlit，Week 7–12）
- 项目 C 股票/基金自选池助手（时序特征工程 + RankNet + ListNet + APScheduler + ONNX + Streamlit，Week 13–18）
- 项目 D 本地 LLM 推理引擎（llama.cpp + MLX 推理对比 + few-shot prompting + CLI 工具，Week 19–21）
- 项目 E Week 22：Colab 跑通 LoRA 训练流程（GPT-2 + SST-2，val_acc 0.884，0.24% 参数可训练，adapter 1.2MB）

**环境**：MacBook Apple Silicon M2 16GB，conda，mlenv 环境
- faiss 必须用 conda-forge 版：`conda install -c conda-forge faiss-cpu`
- KMP_DUPLICATE_LIB_OK=TRUE 已永久写入 conda activate.d
- M2 16GB：7B INT4 可推理，7B FP16 / 任何 LoRA fine-tune 不可行
- llama.cpp via Homebrew b8680，模型在 `~/ml-projects/llm-inference/models/`
- mlx-lm 0.31.2 已安装

**注意事项**：
- 暂不使用 AWS infrastructure
- 教学风格：每次只给一个 cell 的代码，跑完看结果再继续，遇到问题用数据验证而不是猜测

完整学习进度文档见附件 `ml_learning_progress_v15.md`。
