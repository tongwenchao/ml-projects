import os
import json
import torch
import pickle
import numpy as np
import math
import pandas as pd
from predict import BillPredictor

# SageMaker 启动时调用，加载模型
def model_fn(model_dir):
    predictor = BillPredictor(model_dir=model_dir)
    return predictor

# 处理输入数据
def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        return data
    raise ValueError(f"不支持的 content type: {content_type}")

# 推理
def predict_fn(data, predictor):
    if isinstance(data, list):
        return predictor.predict_batch(data)
    else:
        return predictor.predict(
            data["description"],
            data["amount"],
            data["date"]
        )

# 处理输出
def output_fn(prediction, accept="application/json"):
    return json.dumps(prediction, ensure_ascii=False), accept