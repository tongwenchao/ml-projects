# Week 19 · llama.cpp 推理基准

## 环境
- llama.cpp via Homebrew, version b8680
- Apple M2 16GB, Metal backend

## GPU vs CPU 对比（Llama-3.2-3B Q4_K_M）
| ngl | Prompt t/s | Generation t/s |
|-----|-----------|----------------|
| 99 (GPU) | 275.6 | 38.5 |
| 0 (CPU)  | 33.6  | 27.1 |

## 模型对比（ngl 99）
| 模型 | 大小 | Prompt t/s | Generation t/s |
|------|------|-----------|----------------|
| Llama-3.2-3B Q4_K_M | 1.9GB | 275.6 | 38.5 |
| Mistral-7B Q4_K_M   | 4.1GB | 66.3  | 17.2 |

## 结论
- 始终用 -ngl 99，Generation 快 1.4x，Prompt eval 快 8x
- Generation 速度与参数量近似成反比
- 7B INT4 在 M2 16GB 可用，17 t/s 流畅
