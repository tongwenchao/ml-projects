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
- 项目 A：个人影视推荐引擎（Week 7–12）🔄 进行中
- 项目 C：股票/基金自选池助手（Phase 1 结束后选做）

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

### 关键踩坑记录

1. **近零方差维度导致标准化爆炸**：BERT proj 输出 16 维里有 11 维几乎全为 0，StandardScaler 除以极小 std 后产生天文数字（Conoco 误差 32.9 → 修复后 0.07）。解决：用 VarianceThreshold(1e-6) 过滤，保留 8 维。

2. **模型迭代导致 pipeline 不一致**：手工 Embedding 升级到 BERT 后，Autoencoder 的输入维度从 11 变 19，必须重训 Autoencoder，不能直接 pooling 对齐维度。

3. **布尔 mask 当整数索引**：`idx.numpy()[mikes_mask.values]` 把布尔值 True/False 转成 0/1 当 index 用，取到了错误的误差值。正确做法是先取原始行号，再在 idx 里反查位置。

4. **SageMaker SDK 版本**：sagemaker>=3.0 的 API 不兼容，需要 `pip install "sagemaker<3.0"`。

5. **optimizer.zero_grad() 漏写**：梯度累加导致 loss 爆炸，PyTorch 默认累加而不是覆盖梯度。

6. **Autoencoder 需要标准化，分类器不需要**：分类器的 Linear 层权重可以自动补偿特征尺度差异；Autoencoder 的 MSE loss 直接受特征值域影响，尺度大的维度主导 loss。

### 环境配置
```bash
conda create -n mlenv python=3.11 -y
conda activate mlenv
pip install torch torchvision
pip install jupyter notebook pandas scikit-learn matplotlib
pip install transformers
pip install streamlit
pip install onnx onnxruntime onnxscript
pip install awscli boto3 "sagemaker<3.0"
```

### ONNX 导出结果
- PyTorch vs ONNX 最大误差：0.000005（正常）
- 推理速度：PyTorch 0.021ms → ONNX Runtime 0.007ms，加速 3.21x

### SageMaker 部署（已验证，暂不继续用 AWS）
- IAM Role：SageMakerExecutionRole（AmazonSageMakerFullAccess + AmazonS3FullAccess）
- 实例：ml.t2.medium（~$0.05/hr）
- 项目 A 部署时改用 ONNX 格式（待做）
- 删除命令：`aws sagemaker delete-endpoint --endpoint-name bill-analyzer --region us-west-2`

### GitHub
- 仓库：https://github.com/tongwenchao/bill-analyzer（Private）
- 不提交：.pth / .pkl / .npy / .csv / .DS_Store

---

## 项目 A：影视推荐引擎（进行中）🔄

### 当前进度
- Week 8 开始
- Week 7 已完成：数据探索 + Matrix Factorization，最佳 val RMSE 0.8934，权重保存在 `../data/mf_best.pth`

### 项目目录
```
~/ml-projects/recommender/
├── data/
│   ├── ml-1m/
│   │   ├── movies.dat    # 3883 部电影（3706 有评分，177 无评分）
│   │   ├── ratings.dat   # 1,000,209 条评分
│   │   └── users.dat     # 6040 个用户
│   └── mf_best.pth       # MF 最佳模型权重（epoch 8）
└── notebooks/
    ├── week7.ipynb        # 数据探索 + Matrix Factorization（已完成）
    └── week8.ipynb        # 当前工作文件
```

### 数据格式
```python
# ratings: user_id, movie_id, rating(1-5), timestamp
# movies:  movie_id, title, genres（多标签，| 分隔）
# users:   user_id, gender, age, occupation, zip
# 分隔符：:: ，读取时用 sep="::", engine="python"

ratings = pd.read_csv("../data/ml-1m/ratings.dat", sep="::",
    names=["user_id","movie_id","rating","timestamp"], engine="python")
movies = pd.read_csv("../data/ml-1m/movies.dat", sep="::",
    names=["movie_id","title","genres"], engine="python", encoding="latin-1")
users = pd.read_csv("../data/ml-1m/users.dat", sep="::",
    names=["user_id","gender","age","occupation","zip"], engine="python")
```

### Week 7–12 计划

**Week 7**：协同过滤原理 ✅
- 数据探索：评分分布（4星最多34.9%）、矩阵稠密度（4.47%，95.53%为空）、热门电影 Top 20
- Matrix Factorization：user/item embedding 点积 + bias，global_bias 初始化为全局均值(3.59)
- 踩坑：clamp 在边界梯度为0（初始预测全为1.0，loss不动）；按时间戳切分防止 data leakage
- 训练结果：val RMSE 0.8934（epoch 8 early stop，weight_decay=1e-5）

**Week 8**：Neural Collaborative Filtering ← 当前位置
- 在 MF 基础上加 MLP 分支
- 负采样训练
- 评估指标：Hit Rate@10 / NDCG@10

**Week 9**：加入内容特征
- sentence-transformers 编码电影简介
- 拼接到 item embedding，冷启动处理
- 双塔模型（Two-Tower）最简版本

**Week 10**：部署
- Faiss 向量检索（本地）召回 → NCF 打分排序
- 暂不部署 SageMaker（已暂停 AWS 使用）
- 本地 serving：FastAPI 封装推理接口

**Week 11**：ONNX 优化
- NCF 模型导出 ONNX
- 推理延迟 benchmark
- Netron 可视化计算图

**Week 12**：整合收尾
- Streamlit UI：输入已看电影 → 实时推荐 10 部
- A/B 对比：MF vs NCF 推荐结果
- Phase 2 准备

### 核心概念（待学习）
- 协同过滤 vs 内容过滤
- 隐式反馈（看过=1，未看=0）vs 显式评分
- 负采样为什么必要
- Hit Rate@K / NDCG@K 的含义
- 召回（ANN/Faiss）→ 排序（NCF打分）两阶段架构
- 双塔模型（Two-Tower）基本结构

---

## 项目 C：股票/基金自选池助手（Phase 1 选做）

### 定位
Phase 1 结束后，如果时间允许再做。难度最高，时序数据 + learning-to-rank 组合，容易迷失在数据问题里。建议 Phase 1 基础扎稳后再回来。

### 目标
- 输入持仓偏好 → 模型打分排序 → 推送每日 watch list
- 聚焦在**排序逻辑**，不做价格预测（容易掉坑）

### 技术点
- 时序特征工程
- Learning-to-rank（LambdaRank / ListNet）
- 定时推理 pipeline
- 数据源：yfinance（免费）

---

## 重要概念总结（已掌握）

### PyTorch 核心
- `requires_grad=True`：追踪该变量用于自动求导
- `loss.backward()`：反向传播，沿计算图求导
- `optimizer.zero_grad()`：清空梯度（必须在 backward 前调用）
- `optimizer.step()`：用梯度更新参数
- `model.train()` / `model.eval()`：切换训练/推理模式（影响 Dropout、BatchNorm）
- `torch.no_grad()`：推理时关闭梯度计算，省内存
- `torch.save(model.state_dict())` / `model.load_state_dict(torch.load(...))`：保存/加载权重
- `register_buffer`：注册不需要训练的固定张量，保存加载时自动处理

### 模型设计
- `nn.Embedding`：离散 ID → 连续向量，本质是一个可训练的查找表
- `nn.Linear` + `nn.ReLU`：基本 MLP 构建块
- `nn.Dropout`：训练时随机丢弃神经元，缓解 overfit
- `nn.CrossEntropyLoss`：分类任务 loss
- `F.mse_loss`：回归/重建任务 loss

### 工程概念
- Overfitting：train loss 降、val loss 升，模型记住了噪声
- 标准化（StandardScaler）vs 归一化（MinMaxScaler）：前者对异常值更鲁棒
- 迁移学习：冻结预训练权重，只训练后面的 head
- ONNX：模型格式标准化，脱离 PyTorch 依赖，推理更快
- SageMaker endpoint 内部：TorchServe + inference.py，不是 Streamlit
- SageMaker 调用方式：boto3 / 直接 HTTP + AWS Signature V4 / SageMaker SDK
- IAM Role 最小权限原则：只给需要的权限，不用 S3FullAccess 时用 inline policy

### 推荐系统专项（Week 7 新增）
- 矩阵分解（MF）：user/item embedding 点积预测评分，压缩至隐空间（k=64）
- Bias 设计：global_bias 承担全局均值，user/item bias 学习偏离量，职责分离
- `clamp` 梯度陷阱：值落在边界时梯度为0，初始化必须让预测落在合理范围内
- Data leakage：推荐系统必须按时间戳切分，随机切分会让模型"偷看"未来评分
- 稠密度 4.47%：推荐系统本质是补全稀疏矩阵
- Early stopping + weight_decay：共同对抗 overfitting
- 标准化 ≠ 归一化（常被混用，实际不同）
- 布尔 mask ≠ 整数索引（numpy 里两者行为不同）
- `@st.cache_resource`（缓存对象，如模型）vs `@st.cache_data`（缓存数据，如推理结果）
- 分类器不需要标准化（权重自动适应）vs Autoencoder 必须标准化（MSE loss 受尺度影响）
- Autoencoder 检测的是"模式罕见性"，不是"金额大小"

---

## 新 Chat 继续的说明

把以下内容贴给新的 Claude 即可继续：

---

我正在系统学习 ML hands-on 技能，请扮演我的老师，按照以下进度继续。

**当前进度**：项目 A 影视推荐引擎 Week 8，Week 7 已完成。MF 模型已训练，最佳 val RMSE 0.8934，权重保存在 `../data/mf_best.pth`。下一步是 Neural Collaborative Filtering，在 MF 基础上加 MLP 分支，引入负采样，评估指标换成 Hit Rate@10 / NDCG@10。

**已完成**：
- 项目 B 账单分析工具（PyTorch 分类器 + Autoencoder 异常检测 + Streamlit + SageMaker 部署 + ONNX 导出）
- 项目 A Week 7：数据探索 + Matrix Factorization（val RMSE 0.8934）

**环境**：MacBook Apple Silicon，conda，mlenv 环境，PyTorch 已装

**当前文件**：`~/ml-projects/recommender/notebooks/week8.ipynb`

**数据读取代码**（已跑通）：
```python
import pandas as pd

ratings = pd.read_csv("../data/ml-1m/ratings.dat", sep="::",
    names=["user_id","movie_id","rating","timestamp"], engine="python")
movies = pd.read_csv("../data/ml-1m/movies.dat", sep="::",
    names=["movie_id","title","genres"], engine="python", encoding="latin-1")
users = pd.read_csv("../data/ml-1m/users.dat", sep="::",
    names=["user_id","gender","age","occupation","zip"], engine="python")
# ratings: 1,000,209 条，movies: 3883 部，users: 6040 个
```

**注意事项**：
- 暂不使用 AWS infrastructure，本地或 Vast.ai 跑训练
- 项目 C（股票助手）Phase 1 结束后选做
- 教学风格：每次只给一个 cell 的代码，跑完看结果再继续，遇到问题用数据验证而不是猜测

完整学习进度文档见附件 `ml_learning_progress_v3.md`。