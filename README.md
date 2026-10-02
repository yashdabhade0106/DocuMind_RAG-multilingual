# DocuMind_RAG — Latency-Optimized Multilingual RAG Application

A lightweight, enterprise-grade Retrieval-Augmented Generation (RAG) system built from scratch in Python without third-party RAG orchestration frameworks (no LangChain, no LlamaIndex). 

DocuMind features a high-performance **FastAPI** backend with real-time SSE streaming, a modern **React + Vite + TypeScript** frontend with citations and per-stage metrics breakdown, and an optional **Streamlit** dashboard.

---

## 🌟 Key Features

- **Zero-Framework Hand-Rolled RAG**: Complete architectural control over document loading (`pypdf`, `python-docx`, `.txt`), custom chunking, `sentence-transformers` embeddings, and `ChromaDB` vector storage.
- **Persistent Multi-Document Registry**: Store multiple documents in a single shared ChromaDB collection with `doc_id` filtering. Upload, select, or delete documents seamlessly.
- **Latency & Cache Optimizations**:
  - Model and vector store singletons with startup pre-warming.
  - On-disk hash caching (`.cache/`) with `schema_version` validation to skip re-chunking and re-embedding.
  - Multi-threaded parallel PDF text extraction (`ThreadPoolExecutor`).
  - Thread-safe in-memory LRU query cache keyed by `(normalized_question, language, sorted(doc_ids))`.
  - Sentence-pipelined multilingual streaming flushing on sentence boundaries (including Devanagari danda `।`).
- **Resilient 3-Tier Multilingual Pipeline (English / Hindi / Marathi)**:
  - **Tier 1 (Primary)**: Dedicated Indic translation via **Sarvam AI REST API** (`api.sarvam.ai/translate`).
  - **Tier 2 (Fallback)**: LLM translation pass via Groq (`llama-3.1-8b-instant`) or local Ollama (`gemma:2b`).
  - **Tier 3 (Graceful Fallback)**: Returns original text with a clear `Translation Unavailable` notification banner.
- **Modern Full-Stack UI**:
  - React + TypeScript + Tailwind CSS with dark/light themes.
  - Real-time token-by-token streaming with typing cursor and generation cancellation.
  - Drag-and-drop document upload with animated progress tracking.
  - Interactive citation cards showing document name, page number, and vector relevance score.
  - Live latency diagnostics (Retrieval, Generation, Translation, TTFT).
  - Provider health monitoring (Groq, Sarvam AI, Ollama).

---

## 🏗️ Architecture

```
                       ┌────────────────────────────────────────┐
                       │     React + Vite + TypeScript UI       │
                       │   (SSE Streaming, Citations, Metrics)  │
                       └───────────────────┬────────────────────┘
                                           │ HTTP / SSE
                                           ▼
┌──────────────────────┐       ┌────────────────────────────────────────┐
│  Streamlit Dashboard │◄─────►│          FastAPI Backend API           │
│      (app.py)        │       │   (Lifespan Warmup, CORS, REST, SSE)   │
└──────────────────────┘       └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │        Decoupled RAG Core Engine       │
                               │           (rag_service.py)             │
                               └──────┬───────────────────────┬─────────┘
                                      │                       │
                ┌─────────────────────┴───────┐     ┌─────────┴─────────────┐
                ▼                             ▼     ▼                       ▼
      ┌──────────────────┐           ┌──────────────┐   ┌────────────────┐  ┌──────────────┐
      │     ChromaDB     │           │  Disk Cache  │   │      Groq      │  │  Sarvam AI   │
      │ (Shared Collect) │           │ (.cache/v2)  │   │ (Primary LLM)  │  │ (Indic API)  │
      └──────────────────┘           └──────────────┘   └────────────────┘  └──────────────┘
```

---

## 🚀 Quickstart

### 1. Clone & Set Up Python Environment

```bash
git clone https://github.com/AmeyKhodke/DocuMind_RAG.git
cd DocuMind_RAG

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and supply your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
HF_TOKEN=your_hf_token_here
```

> **Note**: Both Groq and Sarvam AI offer free tiers. If an API key is missing or rate-limited, DocuMind automatically falls back to local Ollama or heuristic retrieval scoring.

---

### 3. Run the Application

#### Option A: Full-Stack React + FastAPI App (Recommended)

1. **Start the FastAPI backend** (runs on port 8000):
   ```bash
   uvicorn backend.main:app --port 8000 --reload
   ```

2. **Start the React frontend** (in a separate terminal, runs on port 5173):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Open [http://localhost:5173](http://localhost:5173) in your browser.

#### Option B: Standalone Streamlit Dashboard

If you prefer the single-process Streamlit interface:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Provider connectivity status (Groq, Sarvam, Ollama) |
| `GET` | `/api/languages` | Supported languages and BCP-47 codes |
| `GET` | `/api/documents` | List uploaded documents with chunk and cache metadata |
| `POST` | `/api/documents` | Upload and process documents (`.pdf`, `.docx`, `.txt`) |
| `DELETE` | `/api/documents/{doc_id}` | Delete document and purge related cache entries |
| `POST` | `/api/chat/stream` | Server-Sent Events (SSE) streaming chat endpoint |

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 🧪 Testing & Benchmarks

### Automated Test Suite
Run the backend test suite covering endpoints, caching, document lifecycle, and fallback mechanisms:

```bash
pytest
```

### Latency Profiling
Run the standalone benchmarking tool:

```bash
python test_and_profile.py
```

Detailed performance statistics and architectural comparisons are documented in [PERFORMANCE.md](PERFORMANCE.md).

---

## 🌐 Adding New Languages

Adding support for an additional language is completely zero-code on the frontend:

1. Open `rag_service.py` and append your language to `LANGUAGE_CODES`:
   ```python
   LANGUAGE_CODES = {
       "English": "en-IN",
       "हिंदी (Hindi)": "hi-IN",
       "मराठी (Marathi)": "mr-IN",
       "ગુજરાતી (Gujarati)": "gu-IN",   # Example 4th language
   }
   ```
2. Both the FastAPI backend `/api/languages` endpoint and the Streamlit dropdown will automatically populate the new language.

---

## 📄 License
MIT License. Handcrafted for high-speed, local & cloud hybrid multilingual document intelligence.
