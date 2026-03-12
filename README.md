# RAG AI Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions grounded in your own documents. Embeddings are stored in PostgreSQL via pgvector; retrieval uses cosine similarity search.

## Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline:

```
User Question
      │
      ▼
FastAPI API Layer
      │
      ▼
Agent Controller
(LLM + Tool Selection)
      │
      ├───────────────┐
      │               │
      ▼               ▼
search_documents()   get_current_time()
      │
      ▼
Sentence Transformer
(Query Embedding)
      │
      ▼
PostgreSQL + pgvector
(Vector Similarity Search)
      │
      ▼
Relevant Document Chunks
      │
      ▼
LLM Reasoning
(Final Answer Generation)
      │
      ▼
Answer + Source Citations
```

The agent layer decides whether to call tools such as document retrieval or system utilities before generating the final response.

## Features

- Document chunking with configurable size and overlap
- Sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- HNSW index for fast approximate nearest-neighbor search
- Source-grounded prompting — LLM is restricted to retrieved context
- Returns answer + source citations (filename, chunk index)
- Retrieval logging to stdout

## Requirements

- Python 3.9+
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension
- An OpenAI API key (or swap in Anthropic — see `app.py`)

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd rag-ai-assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your API key
```

**.env.example**
```
OPENAI_API_KEY=your_key_here
```

## Database Setup

```sql
-- Run once in psql
CREATE DATABASE ragdb;
\c ragdb
CREATE EXTENSION vector;
```

Default connection: `postgres:postgres@localhost:5432/ragdb`
Override by setting `DATABASE_URL` in `.env` if needed.

## Usage

**Step 1 — Ingest documents**

Drop `.txt` files into `documents/`, then run:

```bash
python ingest.py
```

This chunks each file, generates embeddings, and upserts them into pgvector.

**Step 2 — Start the API**

```bash
uvicorn app:app --reload
```

**Step 3 — Ask a question**

```bash
curl "http://localhost:8000/ask?question=What+is+Kubernetes?"
```

Response format:
```json
{
  "answer": "Kubernetes is ...",
  "sources": [
    {"filename": "kubernetes.txt", "chunk_index": 2}
  ]
}
```

## Project Structure

```
rag-ai-assistant/
├── documents/          # Source .txt files for ingestion
├── app.py              # FastAPI query endpoint
├── ingest.py           # Document chunking + embedding pipeline
├── requirements.txt
├── .env                # API keys (not committed)
└── .env.example        # Template for environment variables
```

## Configuration

| Parameter | Location | Default |
|-----------|----------|---------|
| Chunk size (words) | `ingest.py` `chunk_text()` | 250 |
| Chunk overlap | `ingest.py` `chunk_text()` | 50 |
| Top-k retrieval | `app.py` `LIMIT` | 7 |
| Embedding model | both files | `all-MiniLM-L6-v2` |
| LLM | `app.py` | `gpt-4o-mini` |

## Demo

Example query:

http://localhost:8000/ask?question=How%20do%20Docker%20and%20Kubernetes%20work%20together?

Example response:

Docker is a platform for packaging applications into containers, while Kubernetes orchestrates and manages those containers at scale. Together, they enable developers to build containerized applications and run them reliably across distributed environments.

Sources:
- docker.txt (chunk 1)
- kubernetes.txt (chunk 0)