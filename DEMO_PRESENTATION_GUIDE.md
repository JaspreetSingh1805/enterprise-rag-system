# Enterprise Configurable RAG Platform — Interview Demo & Presentation Guide

> **Target Audience**: Technical Interviewer / Hiring Manager / Solution Architect  
> **Presentation Duration**: 10 – 15 Minutes  
> **Project Goal**: A production-grade, containerized RAG platform that dynamically configures every pipeline stage (chunking, embeddings, vector store, search, reranking, LLM) without code changes or server restarts, enforces strict multi-tenant isolation, cites sources, and safely rejects unanswerable questions.

---

## 1. The 30-Second Elevator Pitch

> *"I built a production-grade, configurable enterprise RAG platform designed for rapidly changing company documents—such as HR policies, security guidelines, employee handbooks, and API technical manuals.*  
>  
> *Unlike rigid prototype RAG scripts, every stage in this system is modular, containerized, and dynamically configurable by an administrator via versioned REST APIs without code changes or container rebuilds. It features multi-format ingestion (PDF, DOCX, TXT, MD) with background jobs and SHA-256 deduplication, multi-tenant knowledge base isolation, two distinct chunking strategies with documented trade-offs, provider adapters for LLMs and embeddings, hybrid vector + BM25 search with Reciprocal Rank Fusion, cross-encoder reranking, and a strict grounding gate that prevents hallucinations and explicitly says **'I could not find information about this in the knowledge base'** when data is missing.*  
>  
> *It boots in a single Docker Compose command, passes 35 automated tests in CI/CD (79% coverage), and includes a 26-question gold standard benchmark with real FastEmbed embeddings and FlashRank cross-encoder reranking, driving MRR from 0.9352 to 0.9722 and Recall@1 from 88.9% to 94.4%."*

---

## 2. Core Concepts Explained in Plain English

When the interviewer asks conceptual questions, use these clear, real-world analogies:

### Concept 1: Hexagonal Architecture (Ports & Adapters)
- **Plain English**: Think of the system as a laptop with USB-C ports. The core RAG logic is the laptop. The LLMs (OpenAI, Ollama, Mock), Embeddings (MiniLM, OpenAI, Mock), and Vector Stores (Qdrant, In-Memory) are USB accessories.
- **Why it matters**: We can switch from OpenAI to an offline local model or switch from Qdrant to in-memory testing without rewriting a single line of business logic.

### Concept 2: Chunking Strategies & Trade-offs
- **Strategy A: Recursive Character Chunking**:
  - *Analogy*: Slicing a loaf of bread using a ruler every 500 millimeters, sliding back 50mm for overlap.
  - *Trade-off*: Extremely fast, predictable token size, but might split a bullet list or sentence mid-thought.
- **Strategy B: Structure / Markdown-Aware Chunking**:
  - *Analogy*: Slicing a book along its chapters, sections, and paragraphs, keeping the chapter title stamped on every slice.
  - *Trade-off*: Preserves semantic meaning and hierarchical context in citations, but chunks have variable lengths.

### Concept 3: Hybrid Search & Reciprocal Rank Fusion (RRF)
- **Plain English**:
  - **Vector search** understands *concepts* (e.g., "vacation" matches "annual leave"), but can miss exact acronyms or numbers.
  - **Keyword search (BM25)** matches *exact words* (e.g., "HTTP 429", "$110 per diem", "BitLocker"), but doesn't understand synonyms.
  - **Reciprocal Rank Fusion (RRF)** combines both into a single ranked list using formula:
    $$\text{Score}(d) = \sum \frac{1}{60 + \text{rank}(d)}$$
  - *Result*: You get the semantic intuition of vectors plus the laser precision of keywords.

### Concept 4: Grounding & The "Not Found" Rejection Gate
- **Plain English**: Traditional chatbots hallucinate when they don't know the answer because they are forced to generate text. Our platform has a **two-tier grounding firewall**:
  1. **Similarity Gate**: If the top retrieved chunk score is below the threshold (e.g., 0.35), the LLM is not even called. The system immediately returns: *"I could not find information about this in the knowledge base."*
  2. **Citation Verification**: Every fact must cite `[Doc: <name>, Page: <page>, Section: <section>]`. If the context lacks the facts, the model is strictly forbidden from guessing.

### Concept 5: Safe Re-Indexing on Model Change
- **Plain English**: If an admin upgrades the embedding model from a 384-dimensional local model to a 1536-dimensional OpenAI model, existing vector points cannot be queried because vector dimensions must match.
- **Our Solution**: The platform detects the dimension mismatch, safely recreates the vector index, and automatically re-embeds all existing documents from stored raw chunks in SQLite with zero data loss.

---

## 3. The 10-Minute Live Demo Script (Step-by-Step)

Follow this exact script during your interview. You can run commands in **Git Bash** or show them live in **Swagger UI** (`http://localhost:8080/docs`).

### Act 1: The Setup & Health Check (Minute 0 – 2)

**What to Say**:
> *"The platform is completely containerized. It starts with one command: `docker compose up -d`. Let me show you the live running containers and system health."*

**What to Run**:
```bash
docker compose ps
```
**Show on Screen**:
- `rag-api` is `healthy` on port `8080`
- `rag-qdrant` is `healthy` on port `6333`

**What to Run**:
```bash
curl -s http://localhost:8080/api/v1/health
```
**Show on Screen**:
```json
{
  "status": "healthy",
  "environment": "production",
  "active_config_version": "v1",
  "vector_store_type": "qdrant",
  "vector_store_status": "connected"
}
```

> **Pro-Tip**: Open **`http://localhost:8080/docs`** in your browser. Show the interviewer the clean, standardized OpenAPI documentation with endpoints for Knowledge Bases, Documents, Configs, Query, Jobs, and Health.

---

### Act 2: Multi-Tenant Knowledge Base & Async Ingestion (Minute 2 – 5)

**What to Say**:
> *"Next, I'll demonstrate multi-tenancy and multi-format document ingestion. Each Knowledge Base is completely isolated. When documents are uploaded, background jobs process them asynchronously with SHA-256 deduplication."*

**1. Create Knowledge Base `acme_corp`**:
```bash
curl -s -X POST http://localhost:8080/api/v1/knowledge-bases/ \
  -H "Content-Type: application/json" \
  -d '{"id": "acme_corp", "name": "Acme Corporation KB", "description": "Official company policies"}'
```

**2. Ingest Sample Documents (Markdown, PDF, DOCX, TXT)**:
```bash
# Upload Markdown policy (HR Leave Policy)
curl -s -X POST http://localhost:8080/api/v1/documents/upload \
  -F "knowledge_base_id=acme_corp" \
  -F "file=@sample_docs/hr_leave_policy.md" \
  -F 'custom_fields={"department": "HR", "category": "Policy"}'

# Upload PDF guide (Payment Gateway Workflow)
curl -s -X POST http://localhost:8080/api/v1/documents/upload \
  -F "knowledge_base_id=acme_corp" \
  -F "file=@sample_docs/product_workflow_api.pdf" \
  -F 'custom_fields={"department": "Engineering", "category": "API"}'

# Upload DOCX manual (Employee Handbook)
curl -s -X POST http://localhost:8080/api/v1/documents/upload \
  -F "knowledge_base_id=acme_corp" \
  -F "file=@sample_docs/employee_handbook.docx" \
  -F 'custom_fields={"department": "Operations", "category": "Handbook"}'
```

**3. Show Document List & Deduplication**:
```bash
curl -s "http://localhost:8080/api/v1/documents/?knowledge_base_id=acme_corp"
```
**Point out to Interviewer**:
- Notice the extracted page counts, section counts, content hashes, and custom fields.
- If you upload the same file again, the SHA-256 deduplication engine detects it and prevents duplicate pollution.

---

### Act 3: Grounded Querying & The "Not Found" Test (Minute 5 – 8)

**What to Say**:
> *"Now let's test query execution. I will show two queries: first, an answerable policy question requiring citations; second, an unanswerable question to prove the system refuses to hallucinate."*

#### Query 1: Answerable Question (HR Policy)
```bash
curl -s -X POST http://localhost:8080/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"knowledge_base_id": "acme_corp", "question": "Under what condition is a medical certificate required when taking sick leave?"}'
```

**Highlight to Interviewer**:
1. **The Answer**: *"Consecutive sick leave exceeding 2 business days requires an official medical certificate signed by a licensed medical practitioner."*
2. **The Citation**: Exact document citation `[Doc: hr_leave_policy.md, Page: 1, Section: Sick & Medical Leave]`.
3. **Audit Trail**: Look at `trace.stages` showing exact milliseconds spent in retrieval and generation, along with token counts and active config version (`v1`).

---

#### Query 2: Unanswerable Question (The Anti-Hallucination Test)
**What to Say**:
> *"Now, let's ask about something not in the company documents, like the Mars Rover landing weight."*

```bash
curl -s -X POST http://localhost:8080/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"knowledge_base_id": "acme_corp", "question": "What was the total landing payload weight of the NASA Curiosity Mars rover mission?"}'
```

**Highlight to Interviewer**:
- Answer: **`"I could not find information about this in the knowledge base."`**
- `is_grounded: true`, `is_found: false`.
- Zero hallucinations, zero false citations.

---

### Act 4: Dynamic Configuration Activation (Minute 8 – 10)

**What to Say**:
> *"One of the core requirements of this assignment is that every pipeline stage must be configurable and activated by an admin without code changes or restarts. Let's create and activate Config `v2` on the fly."*

**1. Create Configuration `v2` (Switching to Hybrid Search + Reranking + Structure Chunking)**:
```bash
curl -s -X POST http://localhost:8080/api/v1/configs/ \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v2",
    "description": "Production v2 with Hybrid Search and Reranking",
    "chunking": {"strategy": "structure", "chunk_size": 500, "chunk_overlap": 50},
    "retrieval": {"search_mode": "hybrid", "top_k": 5, "similarity_threshold": 0.30},
    "rerank": {"enabled": true, "top_n": 3},
    "llm": {"query_rewrite_enabled": true}
  }'
```

**2. Activate `v2` Dynamically**:
```bash
curl -s -X PUT http://localhost:8080/api/v1/configs/v2/activate
```

**3. Verify Active Version**:
```bash
curl -s http://localhost:8080/api/v1/health
```
**Show on Screen**: `active_config_version` is now `"v2"`. Zero server downtime.

---

### Act 5: Evaluation Benchmark & CI/CD (Minute 10 – 12)

**What to Say**:
> *"To rigorously prove retrieval quality, I built an automated evaluation suite testing 26 realistic questions (18 domain-specific answerable questions across PDF, DOCX, TXT, MD, plus 8 out-of-scope unanswerables) across Config A (Baseline Vector) and Config B (Enhanced Hybrid + Reranking) using real local FastEmbed embeddings and FlashRank cross-encoder reranking."*

**What to Run**:
- In **Git Bash**:
  ```bash
  ./.venv/Scripts/python.exe eval/evaluate.py
  ```
- Or in **PowerShell**:
  ```powershell
  .\.venv\Scripts\python.exe eval/evaluate.py
  ```

**Show the Interviewer the Comparison Matrix**:

| Evaluation Metric | Config A (Baseline: Vector-Only) | Config B (Enhanced: Hybrid + Rerank) | Gain / Improvement |
|:---|:---:|:---:|:---:|
| **Mean Reciprocal Rank (MRR)** | **0.9352** | **0.9722** | **+0.0370 (+4.0%)** |
| **Recall@1** | **88.9%** | **94.4%** | **+5.5%** |
| **Recall@3** | **100.0%** | **100.0%** | **+0.0%** |
| **Recall@5** | **100.0%** | **100.0%** | **+0.0%** |
| **Citation Accuracy** | **100.0%** | **100.0%** | **+0.0%** |
| **Grounding Score** | **100.0%** | **100.0%** | **+0.0%** |
| **Unanswerable Rejection Rate** | **100.0%** | **100.0%** | **100% Precision (Zero Hallucinations)** |
| **Latency (p50)** | **9.47 ms** | **38.89 ms** | **+29.42 ms** |
| **Latency (p95)** | **12.50 ms** | **57.61 ms** | **+45.11 ms** |
| **Average Tokens / Query** | **241.3** | **444.1** | **+202.8 tokens** |

**Key Talking Point**:
> *"Config B achieved an MRR of 0.9722 and Recall@1 of 94.4% because the hybrid BM25 + dense vector search combined with FlashRank cross-encoder reranking correctly prioritized exact alphanumeric keywords (e.g., HTTP 429, $110 per diem, RS256) while maintaining hierarchical section context. Both configurations achieved 100% unanswerable rejection accuracy on out-of-scope queries."*

---

## 4. Top 10 Tough Interview Questions & Winning Answers

### Q1: "Why did you choose Qdrant over Chroma or pgvector?"
> **Answer**:  
> *"Qdrant provides three decisive architectural advantages for enterprise RAG:  
> 1. Native payload filtering, allowing us to filter by `knowledge_base_id` and custom metadata directly at the HNSW graph traversal level with zero post-filtering overhead.  
> 2. Dual operation modes: it runs as a distributed cluster in Docker production AND supports an embedded in-memory mode (`QdrantClient(':memory:')`), which allows our 34 pytest unit tests to run in under 3 seconds in CI without needing an external database.  
> 3. Production resilience: snapshotting, distributed collections, and gRPC/REST APIs."*

---

### Q2: "What happens if someone uploads a 50MB corrupted PDF or a blank file?"
> **Answer**:  
> *"The ingestion pipeline has defensive validation before parsing:  
> 1. `validate_file_bytes` checks for 0-byte or whitespace-only files, raising `EmptyFileError` (HTTP 400).  
> 2. Magic header inspection: PDF files must start with `%PDF-` and DOCX files must start with `PK\x03\x04`. If someone renames a JPEG to `.pdf`, `MalformedFileError` is caught with a clear error reason.  
> 3. All ingestion runs in an async `BackgroundJobQueue`. If an individual page fails to parse, the job marks status as `failed` with the error trace, preventing the API server from crashing."*

---

### Q3: "How is multi-tenant isolation enforced?"
> **Answer**:  
> *"Multi-tenancy is enforced at both the storage and retrieval layers:  
> 1. In SQLite, all documents and chunks have a foreign key to `knowledge_base_id`.  
> 2. In Qdrant, collections are scoped per KB, and all query filters strictly inject `knowledge_base_id == X`.  
> 3. We have an automated test in `tests/test_kb_isolation.py` that ingests Tenant A's document and Tenant B's document, and verifies that querying Tenant B's knowledge base never leaks Tenant A's data."*

---

### Q4: "How does the system switch configurations dynamically without restarting?"
> **Answer**:  
> *"Configurations are versioned Pydantic v2 domain models stored in our persistent metadata store. When an admin calls `PUT /api/v1/configs/{version}/activate`, the `DynamicConfigManager` atomically updates the active pointer. All subsequent calls to `get_active_config()` immediately read the new version. Additionally, queries can pass optional `config_overrides` in the JSON request body to A/B test pipeline parameters on a single query without changing the global default."*

---

### Q5: "What is Reciprocal Rank Fusion (RRF) and why is it better than simple score addition?"
> **Answer**:  
> *"Dense vector cosine similarity and BM25 scores have completely different distributions: cosine scores live between 0.0 and 1.0, while BM25 scores are unbounded positive numbers. Simply adding them is fragile and requires manual score normalization heuristics.  
> Reciprocal Rank Fusion (RRF) bypasses score calibration by focusing on **ranks** rather than raw values:  
> $$\text{RRF Score}(d) = \sum_{m \in \{\text{vector}, \text{bm25}\}} \frac{\text{weight}_m}{60 + \text{rank}_m(d)}$$  
> Documents that rank near the top in both dense and sparse retrieval receive the highest boost, regardless of arbitrary model score scales."*

---

### Q6: "How do you detect hallucinations and prevent unanswerable questions from answering?"
> **Answer**:  
> *"We use a two-layer defense:  
> 1. **Similarity Gate**: Before calling the LLM, we check the maximum similarity score of retrieved chunks. If `max_score < similarity_threshold`, the query is flagged as unanswerable immediately, saving LLM tokens and latency.  
> 2. **Strict System Prompt & Extraction**: The system prompt enforces strict grounding: 'Base your answer ONLY on provided context. If context is insufficient, state: I could not find information about this in the knowledge base.'  
> In our 26-question gold standard benchmark, all 8 out-of-scope unanswerable questions achieved a 100% rejection rate with zero false claims."*

---

### Q7: "What happens when an admin switches from a 384-dim embedding model to a 1536-dim model?"
> **Answer**:  
> *"Vector stores fail with dimension mismatch errors if you attempt to insert or search vectors with differing dimensions.  
> We expose an operational endpoint `POST /api/v1/knowledge-bases/{kb_id}/reindex` which accepts an optional target embedding model and dimension. It submits a `knowledge_base_reindex` job to our asynchronous `BackgroundJobQueue`. The worker safely drops the old vector collection in Qdrant, creates a new collection with the target dimension, re-embeds all existing documents from raw chunks stored in SQLite, and updates the knowledge base metadata with live progress tracking (0% -> 25% -> 100%) and zero data loss."*

---

### Q8: "How does your CI/CD pipeline ensure reliability?"
> **Answer**:  
> *"Our GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push and pull request. It executes:  
> 1. Code quality & linting via `ruff`  
> 2. 35 unit, integration, and API tests via `pytest` with coverage report generation (79% coverage)  
> 3. The 26-question evaluation benchmark (`eval/evaluate.py`) using real local neural FastEmbed and FlashRank models  
> 4. A Docker container build and healthcheck verification step  
> A PR cannot be merged unless all tests, evaluations, and Docker packaging pass."*

---

### Q9: "Why did you implement both Recursive and Structure-Aware Chunking?"
> **Answer**:  
> *"Different document types require different chunking strategies:  
> - For continuous narrative prose, **Recursive Character Chunking** is optimal because it creates uniform, high-throughput chunks with sliding window overlap.  
> - For structured enterprise documents (like Markdown policies with `#` headings or DOCX files with hierarchical styles), recursive chunking often breaks tables or separates sub-clauses from their section context. **Structure-Aware Chunking** parses the heading tree and prefixes the parent heading (e.g. `[Sick Leave Policy]`) onto child chunks, which boosted our evaluation citation accuracy from 75% to 100%."*

---

### Q10: "If you had two more weeks on this project, what would you improve?"
> **Answer**:  
> *"I have four concrete production extensions in mind:  
> 1. **OCR Pipeline**: Integrate an OCR sidecar (like Tesseract or AWS Textract) to extract text from scanned image PDFs.  
> 2. **Distributed Queue**: Upgrade the in-memory background worker to Redis + Celery or ARQ for multi-worker container scaling.  
> 3. **Role-Based Access Control (RBAC)**: Add JWT authentication with distinct roles: Admin (can change configs), Editor (can ingest docs), and Reader (can only query).  
> 4. **Relational Backend**: Provide PostgreSQL support via SQLAlchemy alongside SQLite for horizontally scaled multi-replica deployments."*

---

## 5. Cheat Sheet: Useful Commands During the Demo

### For Git Bash (MINGW64):
```bash
# 1. Check Container Status
docker compose ps

# 2. Check Health & Active Config
curl -s http://localhost:8080/api/v1/health

# 3. Run Automated 34-Test Pytest Suite (via Docker or Virtualenv)
docker compose exec rag-api pytest
# OR
./.venv/Scripts/python.exe -m pytest

# 4. Run Evaluation Benchmark (Config A vs Config B)
./.venv/Scripts/python.exe eval/evaluate.py

# 5. Run the Interactive Demo Script
./.venv/Scripts/python.exe scripts/demo.py

# 6. Check Container Logs
docker logs rag-api
docker logs rag-qdrant
```

### For Windows PowerShell:
```powershell
# Run Tests in Container
docker compose exec rag-api pytest

# Run Tests locally
.\.venv\Scripts\python.exe -m pytest

# Run Evaluation Benchmark
.\.venv\Scripts\python.exe eval/evaluate.py

# Run Demo Script
.\.venv\Scripts\python.exe scripts/demo.py
```
