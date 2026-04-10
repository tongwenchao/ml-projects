import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
import math
import numpy as np


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


class BillAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        return F.mse_loss(self.forward(x), x, reduction="none").mean(dim=1)


class BillPredictor:
    def __init__(self, model_dir="../data"):
        self.model_dir = model_dir
        self._load()

    def _load(self):
        with open(f"{self.model_dir}/desc_encoder.pkl", "rb") as f:
            self.desc_enc = pickle.load(f)
        with open(f"{self.model_dir}/label_encoder.pkl", "rb") as f:
            self.label_enc = pickle.load(f)
        with open(f"{self.model_dir}/scaler_v2.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # 加载维度过滤索引
        self.keep_dims = np.load(f"{self.model_dir}/keep_dims.npy")

        bert_emb = torch.load(f"{self.model_dir}/bert_emb_matrix.pth",
                               weights_only=True)
        self.clf = BillClassifierBERT(bert_emb, n_num_features=3, n_classes=5)
        self.clf.load_state_dict(
            torch.load(f"{self.model_dir}/bill_classifier_bert.pth",
                       weights_only=True)
        )
        self.clf.eval()

        self.ae = BillAutoencoder(input_dim=8, latent_dim=4)
        self.ae.load_state_dict(
            torch.load(f"{self.model_dir}/bill_autoencoder_v2.pth",
                       weights_only=True)
        )
        self.ae.eval()

        self.threshold = 0.50
        print("模型加载完成")
        print(f"  已知商家: {len(self.desc_enc.classes_)} 个")
        print(f"  消费类别: {list(self.label_enc.classes_)}")
        print(f"  有效特征维度: {len(self.keep_dims)}")

    def _get_desc_id(self, description):
        known = list(self.desc_enc.classes_)
        if description in known:
            return self.desc_enc.transform([description])[0]
        desc_lower = description.lower()
        for k in known:
            if k.lower() in desc_lower or desc_lower in k.lower():
                print(f"  未知商家 '{description}' → 匹配到 '{k}'")
                return self.desc_enc.transform([k])[0]
        print(f"  未知商家 '{description}' → 使用默认值 'Grocery Store'")
        return self.desc_enc.transform(["Grocery Store"])[0]

    def _preprocess(self, description, amount, date_str):
        desc_id = self._get_desc_id(description)
        date = pd.to_datetime(date_str)
        x_desc = torch.tensor([desc_id], dtype=torch.long)
        x_num = torch.tensor([[
            math.log1p(amount),
            float(date.dayofweek),
            float(date.dayofweek >= 5)
        ]], dtype=torch.float32)
        return x_desc, x_num

    def predict(self, description, amount, date_str):
        x_desc, x_num = self._preprocess(description, amount, date_str)

        with torch.no_grad():
            # 分类
            logits = self.clf(x_desc, x_num)
            probs = torch.softmax(logits, dim=1).squeeze()
            pred_id = probs.argmax().item()
            confidence = probs[pred_id].item()

            # 异常检测：过滤维度 + 标准化
            emb_proj = self.clf.proj(self.clf.bert_emb[x_desc])
            x_combined = torch.cat([emb_proj, x_num], dim=1)      # [1, 19]
            x_filtered = x_combined[:, self.keep_dims]             # [1, 8]
            x_scaled = torch.tensor(
                self.scaler.transform(x_filtered.numpy()),
                dtype=torch.float32
            )
            error = self.ae.reconstruction_error(x_scaled).item()

        return {
            "description": description,
            "amount": f"${amount:.2f}",
            "date": date_str,
            "category": self.label_enc.inverse_transform([pred_id])[0],
            "confidence": f"{confidence:.1%}",
            "anomaly_score": round(error, 4),
            "is_anomaly": error > self.threshold,
        }

    def predict_batch(self, transactions):
        return [self.predict(t["description"], t["amount"], t["date"])
                for t in transactions]

    def predict_csv(self, input_path, output_path):
        df = pd.read_csv(input_path)
        df = df[df["Transaction Type"] == "debit"].copy()
        df = df[~df["Category"].isin(["Credit Card Payment", "Paycheck"])].copy()
        print(f"共 {len(df)} 条交易，推理中...")

        transactions = df[["Description", "Amount", "Date"]].rename(columns={
            "Description": "description",
            "Amount": "amount",
            "Date": "date"
        }).to_dict("records")

        results = self.predict_batch(transactions)
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_path, index=False)
        print(f"结果已保存到 {output_path}")

        anomalies = df_out[df_out["is_anomaly"] == True]
        print(f"\n发现 {len(anomalies)} 条异常交易:")
        print(anomalies[["description", "amount", "category",
                          "anomaly_score"]].to_string())
        return df_out


if __name__ == "__main__":
    predictor = BillPredictor(model_dir="../data")

    test_cases = [
        {"description": "Starbucks",           "amount": 6.50,   "date": "2024-01-15"},
        {"description": "Amazon",              "amount": 89.99,  "date": "2024-01-16"},
        {"description": "Mike's Construction", "amount": 8500.0, "date": "2024-01-17"},
        {"description": "Spotify",             "amount": 10.69,  "date": "2024-01-18"},
        {"description": "Conoco",              "amount": 33.67,  "date": "2024-01-19"},
        {"description": "Sushi Palace",        "amount": 120.0,  "date": "2024-01-20"},
    ]

    print("\n" + "─" * 65)
    print(f"{'商家':<25} {'金额':>8} {'类别':>6} {'置信度':>7} {'异常分':>8} {'异常?':>5}")
    print("─" * 65)

    for r in predictor.predict_batch(test_cases):
        flag = "⚠️ " if r["is_anomaly"] else "  "
        print(f"{r['description']:<25} {r['amount']:>8} "
              f"{r['category']:>6} {r['confidence']:>7} "
              f"{r['anomaly_score']:>8} {flag}")

    print("\n─── 全量账单推理 ───")
    predictor.predict_csv(
        input_path="../data/personal_transactions.csv",
        output_path="../data/predictions.csv"
    )