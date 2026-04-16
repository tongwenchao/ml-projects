#!/usr/bin/env python3
"""
本地 LLM CLI 工具
用法：
  python llm_cli.py "你的问题"
  python llm_cli.py --task sentiment "The stock dropped 8%"
  python llm_cli.py --task translate "今天天气很好"
  python llm_cli.py --model mistral "Explain KV cache briefly"
"""

import argparse
import time
from mlx_lm import load, generate

MODELS = {
    "llama":   "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

TASKS = {
    "chat": {
        "system": "You are a helpful assistant. Be concise.",
        "few_shots": [],
        "template": "{input}",
    },
    "sentiment": {
        "system": "Classify sentiment as positive, negative, or neutral. Only output the label.",
        "few_shots": [
            ("The company reported record profits this quarter.", "positive"),
            ("The product recall affected thousands of customers.", "negative"),
            ("The meeting has been rescheduled to Thursday.", "neutral"),
        ],
        "template": "Sentence: {input}\nLabel:",
    },
    "translate": {
        "system": "Translate the following Chinese sentence to English. Only output the translation.",
        "few_shots": [
            ("今天很热", "It's very hot today."),
            ("我喜欢吃苹果", "I like eating apples."),
            ("市场波动很大", "The market is very volatile."),
        ],
        "template": "Chinese: {input}\nEnglish:",
    },
    "summarize": {
        "system": "Summarize the following text in one sentence.",
        "few_shots": [],
        "template": "Text: {input}\nSummary:",
    },
}

def build_messages(task_name: str, user_input: str) -> list:
    task = TASKS[task_name]
    messages = []

    # 第一条 user 消息带 system 指令 + 第一个 few-shot 问题
    # 如果没有 few-shot，直接是真正的问题
    few_shots = task["few_shots"]

    if not few_shots:
        content = task["system"] + "\n\n" + task["template"].format(input=user_input) if task["system"] else task["template"].format(input=user_input)
        messages.append({"role": "user", "content": content})
    else:
        # 第一个例子：system 指令拼在第一条 user 消息里
        first_q = task["template"].format(input=few_shots[0][0])
        messages.append({"role": "user",      "content": task["system"] + "\n\n" + first_q})
        messages.append({"role": "assistant", "content": few_shots[0][1]})

        # 剩余例子
        for src, tgt in few_shots[1:]:
            messages.append({"role": "user",      "content": task["template"].format(input=src)})
            messages.append({"role": "assistant", "content": tgt})

        # 真正的问题
        messages.append({"role": "user", "content": task["template"].format(input=user_input)})

    return messages

def run(model_key: str, task_name: str, user_input: str, max_tokens: int = 200):
    model_id = MODELS[model_key]
    print(f"[模型] {model_key} ({model_id})")
    print(f"[任务] {task_name}")
    print(f"[输入] {user_input}")
    print("─" * 50)

    t0 = time.time()
    model, tokenizer = load(model_id)
    print(f"[加载] {time.time() - t0:.1f}s")

    messages = build_messages(task_name, user_input)
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    t1 = time.time()
    output = generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False)
    print(f"[输出] {output.strip()}")
    print(f"[耗时] 生成 {time.time() - t1:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="本地 LLM CLI")
    parser.add_argument("input", help="输入文本")
    parser.add_argument("--task", choices=list(TASKS.keys()), default="chat")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="llama")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()
    run(args.model, args.task, args.input, args.max_tokens)

if __name__ == "__main__":
    main()
