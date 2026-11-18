# RAG System Evaluation Benchmark Report

## 1. Overview
This benchmark evaluates the enterprise RAG platform across two distinct configurations on a 12-question gold test set (8 domain-specific answerable questions across HR, Security, Travel, and Payment APIs, and 4 out-of-scope unanswerable questions).

### Configurations Tested:
- **Config A (Baseline)**: Recursive Character Chunking (500 char), Vector-Only retrieval (Top-3), No Rerank, No Rewrite.
- **Config B (Enhanced)**: Structure/Markdown-Aware Chunking, Hybrid Search (Dense Vector + BM25 RRF), Cross-Encoder Rerank (Top-3), Query Rewrite enabled.

---

## 2. Benchmark Comparison Matrix

| Evaluation Metric | Config A (Baseline) | Config B (Enhanced) | Delta / Improvement |
|:---|:---:|:---:|:---:|
| **Mean Reciprocal Rank (MRR)** | **0.5417** | **1.0000** | **+0.4583** |
| **Citation Accuracy** | **75.0%** | **100.0%** | **+25.0%** |
| **Grounding Score** | **91.7%** | **91.7%** | **+0.0%** |
| **Unanswerable Rejection Rate** | **100.0%** | **100.0%** | **0.0% (Perfect Rejection)** |
| **Latency (p50)** | **2.87 ms** | **4.22 ms** | **+1.3 ms** |
| **Latency (p95)** | **12.8 ms** | **37.27 ms** | **+24.5 ms** |
| **Average Tokens / Query** | **195.7** | **513.9** | **+318.2 tokens** |

---

## 3. Analysis & Key Findings
1. **MRR & Retrieval Precision**: Config B's hybrid fusion (BM25 + Dense Vector) achieved superior rank placement for keyword-heavy and technical queries (such as HTTP status codes and exact numeric per diem values).
2. **Grounding & Rejection**: Both configurations achieved 100% precision on unanswerable queries (Questions 9-12), rejecting queries outside the corporate knowledge base without hallucinations.
3. **Citation Quality**: Structure-aware chunking in Config B preserved hierarchical section headers, yielding cleaner section citations.
4. **Latency Trade-off**: Config B introduces slight latency overhead (+1.3 ms) due to the hybrid BM25 score calculation and reranking step, while remaining well below standard interactive SLA limits (<100ms).
