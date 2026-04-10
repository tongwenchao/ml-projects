import boto3
import json

REGION        = "us-west-2"
ENDPOINT_NAME = "bill-analyzer"

client = boto3.client("sagemaker-runtime", region_name=REGION)

# 测试单条
single = {
    "description": "Starbucks",
    "amount": 6.50,
    "date": "2024-01-15"
}

response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(single)
)

result = json.loads(response["Body"].read().decode())
print("单条推理结果:")
print(json.dumps(result, ensure_ascii=False, indent=2))

# 测试批量
batch = [
    {"description": "Starbucks",           "amount": 6.50,   "date": "2024-01-15"},
    {"description": "Mike's Construction", "amount": 8500.0, "date": "2024-01-17"},
    {"description": "Spotify",             "amount": 10.69,  "date": "2024-01-18"},
]

response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(batch)
)

results = json.loads(response["Body"].read().decode())
print("\n批量推理结果:")
for r in results:
    flag = "⚠️" if r["is_anomaly"] else "  "
    print(f"{flag} {r['description']:<25} {r['amount']:>8} {r['category']:>6} {r['anomaly_score']:>8}")