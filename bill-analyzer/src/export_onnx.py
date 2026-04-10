import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import onnx
import onnxruntime as ort
import time

# ── 模型定义 ──────────────────────────────────────────────
class BillClassifierBERT(nn.Module):
    def __init__(self, bert_emb_matrix, n_num_features, n_classes):
        super().__init__()
        bert_dim = bert_emb_matrix.shape[1]
        self.register_buffer("bert_emb", bert_emb_matrix)
        self.proj = nn.Sequential(nn.Linear(bert_dim, 16), nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(16 + n_num_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x_desc, x_num):
        emb = self.proj(self.bert_emb[x_desc])
        return self.mlp(torch.cat([emb, x_num], dim=1))


# ── 加载模型 ──────────────────────────────────────────────
MODEL_DIR = "../data"

bert_emb = torch.load(f"{MODEL_DIR}/bert_emb_matrix.pth", weights_only=True)
clf = BillClassifierBERT(bert_emb, n_num_features=3, n_classes=5)
clf.load_state_dict(
    torch.load(f"{MODEL_DIR}/bill_classifier_bert.pth", weights_only=True)
)
clf.eval()

# ── 导出 ONNX ─────────────────────────────────────────────
# ONNX export 需要一个示例输入，用来追踪计算图
dummy_desc = torch.tensor([0], dtype=torch.long)       # 商家 ID
dummy_num  = torch.tensor([[3.5, 1.0, 0.0]], dtype=torch.float32)  # 数值特征

output_path = f"{MODEL_DIR}/bill_classifier.onnx"

torch.onnx.export(
    clf,
    (dummy_desc, dummy_num),           # 示例输入
    output_path,
    input_names=["desc_id", "x_num"],  # 输入节点名
    output_names=["logits"],           # 输出节点名
    dynamic_axes={                     # 支持 batch size 动态变化
        "desc_id": {0: "batch_size"},
        "x_num":   {0: "batch_size"},
        "logits":  {0: "batch_size"},
    },
    opset_version=17,
)
print(f"ONNX 模型已导出：{output_path}")

# ── 验证 ONNX 模型结构 ────────────────────────────────────
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX 模型结构验证通过")

# ── 对比推理结果 ──────────────────────────────────────────
# PyTorch 推理
with torch.no_grad():
    pt_output = clf(dummy_desc, dummy_num).numpy()

# ONNX Runtime 推理
sess = ort.InferenceSession(output_path)
ort_output = sess.run(
    ["logits"],
    {
        "desc_id": dummy_desc.numpy(),
        "x_num":   dummy_num.numpy(),
    }
)[0]

print(f"\nPyTorch 输出: {pt_output.round(4)}")
print(f"ONNX 输出:    {ort_output.round(4)}")
print(f"最大误差:     {abs(pt_output - ort_output).max():.6f}")

# ── 推理速度对比 ──────────────────────────────────────────
N = 1000

start = time.time()
with torch.no_grad():
    for _ in range(N):
        clf(dummy_desc, dummy_num)
pt_time = (time.time() - start) / N * 1000

start = time.time()
for _ in range(N):
    sess.run(["logits"], {
        "desc_id": dummy_desc.numpy(),
        "x_num":   dummy_num.numpy(),
    })
ort_time = (time.time() - start) / N * 1000

print(f"\n推理速度对比（单条，均值 {N} 次）:")
print(f"  PyTorch:      {pt_time:.3f} ms")
print(f"  ONNX Runtime: {ort_time:.3f} ms")
print(f"  加速比:       {pt_time/ort_time:.2f}x")