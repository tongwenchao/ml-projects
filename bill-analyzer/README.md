# Bill Analyzer

个人账单分类与异常检测工具，基于 PyTorch + HuggingFace + Streamlit。

## 功能

- 自动将交易按类别分类（餐饮、购物、出行、居家、娱乐）
- 检测异常交易（大额、罕见商家）
- Streamlit Web UI，上传 CSV 即可使用

## 模型架构

- **分类器**：BERT embedding（bert-base-uncased）+ MLP，准确率 98%
- **异常检测**：Autoencoder，基于重建误差，无监督

## 项目结构
```
bill-analyzer/
├── data/              # 数据和模型文件（不提交）
├── notebooks/         # 训练过程
│   ├── week1.ipynb    # PyTorch 基础 + 分类器
│   ├── week2.ipynb    # Autoencoder 异常检测
│   └── week2b.ipynb   # BERT embedding
└── src/
    ├── predict.py     # 推理逻辑
    └── app.py         # Streamlit UI
```

## 快速开始

**1. 安装依赖**
```bash
conda create -n mlenv python=3.11 -y
conda activate mlenv
pip install torch torchvision transformers streamlit pandas scikit-learn
```

**2. 准备模型文件**

训练过程见 `notebooks/`，训练完后 `data/` 目录需要包含：
```
bill_classifier_bert.pth
bill_autoencoder_v2.pth
bert_emb_matrix.pth
scaler_v2.pkl
keep_dims.npy
desc_encoder.pkl
label_encoder.pkl
```

**3. 启动 UI**
```bash
cd src
streamlit run app.py
```

## 数据格式

CSV 需包含以下列：

| 列名 | 说明 |
|------|------|
| Description | 商家名称 |
| Amount | 金额 |
| Date | 日期 |
| Transaction Type | debit / credit |

## 已知局限

- 训练数据只有 617 条，未知商家用模糊匹配或默认值处理
- 异常检测对训练集中出现过的模式不敏感
- 模型文件不包含在仓库中，需要自行训练

## ONNX 导出
```bash
pip install onnx onnxruntime onnxscript
cd src
python export_onnx.py
```

导出后模型保存在 `data/bill_classifier.onnx`，推理速度比 PyTorch 快约 3x。

## SageMaker 部署
```bash
pip install "sagemaker<3.0"
export SAGEMAKER_ROLE_ARN="arn:aws:iam::你的账号ID:role/SageMakerExecutionRole"
cd src
python deploy.py       # 部署，约 5–8 分钟
python test_endpoint.py  # 测试
# 测试完立刻删除 endpoint
aws sagemaker delete-endpoint --endpoint-name bill-analyzer --region us-west-2
```
