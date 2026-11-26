# Configurable Enterprise RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) backend built with Python, FastAPI, and Qdrant. 

I designed this platform to solve a real issue common to enterprise RAG setups: **pipeline rigidity**. In most prototype implementations, chunk sizes, embedding models, vector stores, and retrieval methods are hardcoded. Changing a model or tuning chunk overlap often requires editing code, rebuilding containers, and risking data corruption when vector dimensions mismatch. 

Here, every phase of the pipeline—from ingestion and chunking to hybrid retrieval, reranking, and generation—is versioned and configurable at runtime via REST APIs without restarting services. It also enforces strict multi-tenant isolation, source citation tracking, and refuses to guess on questions outside the indexed documents.

---

## Key Highlights & Architectural Decisions

- **Zero-Downtime Configuration Switching**: Pipeline settings (`PipelineConfig`) are versioned (`v1`, `v2`, etc.) and validated with Pydantic v2. An administrator can switch the active pipeline or override settings per query via REST API without rebuilding containers.
- **Two Real-World Chunking Strategies**:
  - *Recursive Character Chunking*: High-throughput, fixed-size chunks with sliding window overlap.
  - *Structure-Aware AST Chunking*: Preserves Markdown heading trees and Word heading styles, prepending parent section titles to child chunks to preserve context.
- **Two-Stage Hybrid Retrieval & Reranking**:
  - *Stage 1 (Candidate Generation)*: Parallel BM25 Okapi keyword search + Dense Vector retrieval combined via Reciprocal Rank Fusion (RRF, $k=60$).
  - *Stage 2 (Joint Attention Reranking)*: High-speed cross-encoder reranking via `FlashRank` (`ms-marco-TinyBERT-L-2-v2`).
- **Real Local Models Out of the Box**: Runs fully offline with zero mandatory API costs using `FastEmbed` (`all-MiniLM-L6-v2`, 384d) and `FlashRank`. Also provides plug-and-play adapters for OpenAI (`gpt-4o`, `text-embedding-3-small`) and Ollama.
- **Safe Dimensional Re-Indexing API**: When upgrading an embedding model (e.g. from 384d to 1536d), vector stores reject mismatched inserts. The system exposes an asynchronous endpoint (`POST /api/v1/knowledge-bases/{kb_id}/reindex`) that drops the old vector collection, provisions the new dimension, re-embeds raw chunks from SQLite, and updates metadata in the background.
- **Strict Grounding & Hallucination Prevention**: Uses a two-tier defense (pre-LLM similarity gating + strict system prompt constraints). When queries are out-of-scope, it reliably outputs: `"I could not find information about this in the knowledge base."`
- **Audit Tracing & Observability**: Every query response includes latency breakdowns per stage (`query_rewrite`, `retrieval`, `rerank`, `generation`), token counts, citations with snippets and scores, and correlation IDs (`X-Trace-ID`).

---

## System Architecture

The project follows a **Hexagonal Architecture (Ports and Adapters)** pattern to ensure complete separation between domain logic and third-party infrastructure.

```
                         +-----------------------------------+
                         |      HTTP / REST Client           |
                         +-----------------+-----------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
| FastAPI Service Layer (/api/v1)                                                   |
| - Correlation ID Middleware (X-Trace-ID)                                          |
| - Routers: /health, /knowledge-bases, /documents, /configs, /query, /jobs         |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
| Core Pipeline Engine                                                              |
|                                                                                   |
|  [ Ingestion Pipeline ]                                                           |
|    File Validation -> SHA-256 Deduplication -> Parsers (PDF, DOCX, TXT, MD)       |
|    -> Chunker (Recursive / Structure-Aware) -> Embedding Adapter -> Vector Store |
|                                                                                   |
|  [ RAG Query Engine ]                                                             |
|    Query Rewriter -> Multi-Tenant Filter -> Hybrid Search (BM25 + Dense RRF)      |
|    -> FlashRank Cross-Encoder -> Similarity Gating -> Grounded LLM Response       |
|                                                                                   |
|  [ Dynamic Config Manager ]                                                       |
|    Versioned config store (v1, v2...) with atomic activation and safe re-indexing     |
+------------------------------------------+----------------------------------------+
                                           |
        +----------------------------------+----------------------------------+
        |                                  |                                  |
        v                                  v                                  v
+-----------------------+      +-----------------------+      +-----------------------+
| LLM Adapters          |      | Embedding Adapters    |      | Storage & Vector DB   |
| - OpenAI (GPT-4o)     |      | - FastEmbed (Local)   |      | - Qdrant (Docker/Mem) |
| - Ollama (Llama 3)    |      | - OpenAI (1536d)      |      | - SQLite (WAL Mode)   |
| - Deterministic Mock  |      | - Deterministic Mock  |      | - In-Memory Storage   |
+-----------------------+      +-----------------------+      +-----------------------+
```

---

## Core RAG Concepts & Design Trade-offs

### 1. Two-Stage Retrieval (Bi-Encoder vs. Cross-Encoder)
A single dense bi-encoder compresses queries and documents into independent fixed-size vectors. While this allows sub-millisecond nearest-neighbor search across millions of documents via HNSW graphs, it sacrifices all cross-attention between the query words and document words.

To achieve both low latency and high accuracy, this platform uses a two-stage pipeline:
1. **Candidate Retrieval (Broad Stage)**: Retrieves top-$K$ candidates (e.g. 5–10) using fast vector search and BM25 keyword matching.
2. **Cross-Encoder Reranking (Precision Stage)**: Passes the top candidates into `FlashRank` (`ms-marco-TinyBERT-L-2-v2`). The cross-encoder evaluates query tokens and passage tokens simultaneously through transformer attention layers, scoring semantic relevance with high accuracy.

### 2. Reciprocal Rank Fusion (RRF) vs. Score Addition
Combining dense vector cosine similarity with sparse BM25 scores is traditionally tricky because their scales differ fundamentally: cosine similarity is bounded between `[0, 1]`, whereas BM25 produces unbounded scores depending on document length and term frequency.

Instead of fragile min-max normalization heuristics, we use **Reciprocal Rank Fusion (RRF)**:
$$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
Where $k = 60$ (the standard Cormack et al. constant) and $w_m$ is the configurable channel weight. RRF prioritizes items that rank consistently high in both channels, ensuring that exact technical acronyms, codes, and numerical figures (e.g. `HTTP 429`, `$110 per diem`, `RS256`) are never lost in dense semantic vector clouds.

### 3. Chunking Strategies: When to Use Which
| Strategy | How it works | Pros | Cons | Best for |
|---|---|---|---|---|
| **Recursive Character** | Splits recursively on `["\n\n", "\n", " ", ""]` until chunk size is met, preserving overlap. | Uniform chunk sizes, predictable token counts, very fast. | Can split bullet lists or sentences mid-clause; lacks heading context. | General narrative text, meeting notes, continuous prose. |
| **Structure-Aware AST** | Parses the document hierarchy (Markdown `#` headings, DOCX styles) and prepends parent section headers to each child chunk. | Preserves full section context in citations; keeps tables and clauses whole. | Variable chunk sizes depending on section length. | API specs, HR policies, compliance handbooks, legal terms. |

### 4. Hallucination Prevention & Two-Tier Refusal
Enterprise users lose trust the moment a RAG system invents an answer. We prevent hallucinations with two sequential safety checks:
1. **Pre-LLM Similarity Gate**: Before invoking the generator, the system inspects the top retrieved passage score. If the score falls below `similarity_threshold` (e.g. `0.30`), retrieval is flagged as ungrounded, and the pipeline immediately aborts generation, returning `"I could not find information about this in the knowledge base."` This saves latency and API tokens.
2. **Grounded Citation Prompting**: When passages pass the threshold, the LLM prompt strictly requires every claim to reference its source `[Doc: <filename>, Page: <page>, Section: <section>]`. If the context does not explicitly verify the question, the model is forbidden from extrapolating.

---

## Quick Start

### Option 1: Run with Docker Compose (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JaspreetSingh1805/enterprise-rag-system.git
   cd enterprise-rag-system
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   ```
   *(By default, `DEFAULT_EMBEDDING_PROVIDER=local` and `DEFAULT_LLM_PROVIDER=mock`, meaning you do not need an OpenAI API key to run the platform locally).*

3. **Start services**:
   ```bash
   docker compose up -d
   ```

4. **Verify health**:
   ```bash
   curl -s http://localhost:8080/api/v1/health
   ```
   Interactive Swagger UI is available at **http://localhost:8080/docs**.

---

### Option 2: Local Python Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv

   # On Windows (Git Bash):
   source .venv/Scripts/activate

   # On Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1

   # On Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample documents**:
   ```bash
   python scripts/generate_sample_docs.py
   ```

4. **Run the interactive demo script**:
   ```bash
   python scripts/demo.py
   ```

5. **Start the FastAPI application**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## Benchmark Evaluation Results

The evaluation suite (`eval/evaluate.py`) tests **26 gold-standard queries** (18 answerable queries across PDF, DOCX, TXT, and Markdown files, plus 8 out-of-scope unanswerable queries) using real local neural models (`FastEmbed` `all-MiniLM-L6-v2` and `FlashRank` `ms-marco-TinyBERT-L-2-v2`).

### Comparison Matrix

| Metric | Config A (Baseline: Vector-Only) | Config B (Enhanced: Hybrid RRF + Reranker) | Delta / Gain |
|:---|:---:|:---:|:---:|
| **Mean Reciprocal Rank (MRR)** | **0.9352** | **0.9722** | **+0.0370 (+4.0%)** |
| **Recall@1** | **88.89%** | **94.44%** | **+5.55%** |
| **Recall@3** | **100.0%** | **100.0%** | **Parity** |
| **Recall@5** | **100.0%** | **100.0%** | **Parity** |
| **Citation Accuracy** | **100.0%** | **100.0%** | **Parity** |
| **Grounding Score** | **100.0%** | **100.0%** | **Parity** |
| **Unanswerable Rejection Rate** | **100.0%** | **100.0%** | **100% Precision (Zero Hallucinations)** |
| **Latency (p50)** | **13.4 ms** | **61.2 ms** | **+47.8 ms** |
| **Tokens / Query (Avg)** | **241.3** | **444.1** | **+202.8 tokens** |

### Benchmark Takeaways
- **Hybrid Search + Reranking Lift**: Dense search alone occasionally ranks a nearby semantic passage above the exact policy clause. Adding BM25 Okapi with RRF and cross-encoder reranking boosted **Recall@1 to 94.44%**, correctly surfacing exact codes (e.g. `HTTP 429`, `$110 per diem`) at position #1.
- **Flawless Refusal**: Both configurations achieved a **100% rejection rate** across all 8 out-of-scope queries (e.g. Mars Curiosity Rover, Apple NASDAQ stock price, pediatric amoxicillin, baseball field dimensions).
- **Latency Trade-off**: The cross-encoder adds ~45ms of CPU inference overhead per query. For strict sub-20ms SLAs, Config A is preferable; for maximum answer precision on technical policies, Config B is recommended.

Run the evaluation suite yourself:
```bash
python eval/evaluate.py
```

---

## REST API Reference

### Health Check
```bash
curl -X GET http://localhost:8080/api/v1/health
```

### Knowledge Base Management
```bash
# Create Knowledge Base
curl -X POST http://localhost:8080/api/v1/knowledge-bases/ \
  -H "Content-Type: application/json" \
  -d '{
    "id": "acme_hr",
    "name": "Acme HR Policies",
    "description": "Internal leave, travel, and benefits guidelines"
  }'

# Trigger Zero-Downtime Asynchronous Reindexing
curl -X POST http://localhost:8080/api/v1/knowledge-bases/acme_hr/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "embedding_provider": "local",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dimension": 384
  }'
```

### Document Upload & Ingestion
```bash
# Upload document (Processed asynchronously with SHA-256 deduplication)
curl -X POST http://localhost:8080/api/v1/documents/upload \
  -F "knowledge_base_id=acme_hr" \
  -F "file=@sample_docs/hr_leave_policy.md" \
  -F 'custom_fields={"department": "HR", "category": "Policy"}'

# Check background job status
curl -X GET http://localhost:8080/api/v1/jobs/{job_id}
```

### Querying
```bash
# Execute RAG query
curl -X POST http://localhost:8080/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_base_id": "acme_hr",
    "question": "Under what condition is a medical certificate required when taking sick leave?"
  }'
```

Sample grounded JSON response:
```json
{
  "question": "Under what condition is a medical certificate required when taking sick leave?",
  "answer": "Based on the provided documentation: Consecutive sick leave exceeding 2 business days requires an official medical certificate signed by a licensed medical practitioner. [Doc: hr_leave_policy.md, Page: 1, Section: Sick & Medical Leave]",
  "is_grounded": true,
  "is_found": true,
  "citations": [
    {
      "document_id": "11f249ac-6863-46b5-8a39-cd8d641c744d",
      "document_name": "hr_leave_policy.md",
      "page_number": 1,
      "section_title": "Sick & Medical Leave",
      "snippet": "Consecutive sick leave exceeding 2 business days requires an official medical certificate...",
      "score": 1.0
    }
  ],
  "trace": {
    "trace_id": "31b268f7-f377-4b55-a0c5-e63d3080ffeb",
    "total_latency_ms": 17.3,
    "config_version": "v1",
    "stages": [
      {"stage_name": "retrieval", "duration_ms": 4.1, "details": {"search_mode": "hybrid", "retrieved_count": 5}},
      {"stage_name": "generation", "duration_ms": 12.8, "details": {"citations_count": 1, "is_found": true}}
    ],
    "prompt_tokens": 142,
    "completion_tokens": 46,
    "total_tokens": 188
  },
  "config_version": "v1"
}
```

### Dynamic Configuration Switching
```bash
# Activate version v2 dynamically across all incoming requests
curl -X PUT http://localhost:8080/api/v1/configs/v2/activate
```

---

## Testing & Code Quality

The test suite covers unit, integration, and API tests:
- Ingestion resilience (0-byte rejection, magic byte header inspection, SHA-256 deduplication).
- Chunking boundary tests and heading metadata extraction.
- Multi-tenant data isolation and cascading deletions.
- Dynamic config switching and dimensional migration.
- Similarity threshold gating and refusal of unanswerable questions.

Run the test suite locally:
```bash
# Run pytest with coverage report
pytest -v --cov=app --cov-report=term-missing

# Run code style and format checks
ruff check app tests eval
ruff format --check app tests eval
```

---

## Production Limitations & Next Steps

If deploying this platform to high-volume multi-node production, here are the real-world trade-offs and natural architectural upgrades:

1. **Distributed Task Queue**: The background worker currently uses an in-memory `asyncio` task queue backed by SQLite job rows. In a distributed multi-replica deployment, upgrading to **Redis + Celery** or **ARQ** would provide persistent task queuing across multiple worker containers.
2. **Metadata Store Clustering**: SQLite runs in WAL (Write-Ahead Logging) mode, which is lightweight, extremely fast, and requires zero setup for single-node setups. For horizontally autoscaled API replicas sharing a database, SQLite can be swapped for **PostgreSQL** by updating the SQLAlchemy database URI.
3. **Scanned PDF OCR**: The PDF parser (`pypdf`) extracts digital text streams directly. Scanned or image-only PDFs require an upstream OCR sidecar (such as Tesseract, AWS Textract, or Google Cloud Document AI).
4. **User Authentication & RBAC**: Multi-tenancy is currently partitioned by `knowledge_base_id` slugs. Adding OAuth2 / JWT Bearer tokens with Role-Based Access Control (Admin, Ingestor, Reader) would secure endpoint access in enterprise environments.

---

## License
MIT License. Created for the Configurable Enterprise RAG System Assignment.
