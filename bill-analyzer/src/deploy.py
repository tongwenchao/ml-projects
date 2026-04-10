import boto3
import os
import sagemaker
from sagemaker.pytorch import PyTorchModel

# 配置
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
S3_MODEL = os.environ.get("S3_MODEL_PATH", "s3://bill-analyzer-models-maple/model.tar.gz")
REGION   = os.environ.get("AWS_REGION", "us-west-2")

session = sagemaker.Session(boto_session=boto3.Session(region_name=REGION))

model = PyTorchModel(
    model_data=S3_MODEL,
    role=ROLE_ARN,
    framework_version="2.1",
    py_version="py310",
    entry_point="inference.py",
    source_dir=".",           # 把整个 src/ 目录打包，包含 predict.py
    sagemaker_session=session,
)

print("部署中，大约需要 5–8 分钟...")
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",   # 最便宜的实例，约 $0.05/hr
    endpoint_name="bill-analyzer",
)

print(f"Endpoint 部署成功：{predictor.endpoint_name}")