import io
import os
import sys
import time
import uuid
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Ensure root workspace directory is in python sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import rag_service
from backend.schemas import (
    DocumentResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
    ChatRequest,
    LanguageItem,
    LanguageListResponse,
    HealthProviderStatus,
    HealthResponse,
)

# Configure logging with request IDs and timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Req:%(request_id)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("docmind.api")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

logger.addFilter(RequestIdFilter())

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm embedding model and ChromaDB client during application startup
    logger.info("Initializing application lifespan: pre-warming embedding model & ChromaDB...", extra={"request_id": "STARTUP"})
    t0 = time.perf_counter()
    try:
        rag_service.get_embedding_model()
        rag_service.get_chroma_client()
        logger.info(f"Model and ChromaDB ready in {time.perf_counter() - t0:.2f}s", extra={"request_id": "STARTUP"})
    except Exception as e:
        logger.warning(f"Lifespan model pre-warm exception: {e}", extra={"request_id": "STARTUP"})
    yield
    logger.info("Application shutdown complete.", extra={"request_id": "SHUTDOWN"})

app = FastAPI(
    title="DocuMind RAG API",
    description="High-performance multilingual RAG backend with real-time SSE streaming",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
origins_env = os.getenv("CORS_ORIGINS", "*")
if origins_env == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True if "*" not in allow_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID & Latency Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = req_id
    t0 = time.perf_counter()

    logger.info(f"{request.method} {request.url.path}", extra={"request_id": req_id})
    try:
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = req_id
        logger.info(f"{request.method} {request.url.path} -> Status {response.status_code} ({elapsed_ms:.1f}ms)", extra={"request_id": req_id})
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc} ({elapsed_ms:.1f}ms)", extra={"request_id": req_id})
        raise

# Helper class for wrapping UploadFile bytes for rag_service.ingest_document
class IngestableFile:
    def __init__(self, filename: str, content: bytes):
        self.name = filename
        self._content = content
        self._io = io.BytesIO(content)

    def read(self, *args, **kwargs):
        return self._io.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._io.seek(*args, **kwargs)

@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """Health check reporting configuration and availability for LLM and translation providers."""
    providers = {}

    # Groq API check
    groq_key = rag_service.get_groq_api_key()
    if groq_key:
        providers["groq"] = HealthProviderStatus(status="ok", details="API Key configured (Tier 1 LLM)")
    else:
        providers["groq"] = HealthProviderStatus(status="not_configured", details="GROQ_API_KEY unset in environment")

    # Sarvam AI check
    sarvam_key = rag_service.get_sarvam_api_key()
    if sarvam_key:
        providers["sarvam"] = HealthProviderStatus(status="ok", details="API Key configured (Tier 1 Indic Translation)")
    else:
        providers["sarvam"] = HealthProviderStatus(status="not_configured", details="SARVAM_API_KEY unset in environment")

    # Ollama check (ping localhost:11434/api/tags)
    try:
        resp = rag_service.HTTP_SESSION.get("http://localhost:11434/api/tags", timeout=1.5)
        if resp.status_code == 200:
            providers["ollama"] = HealthProviderStatus(status="ok", details="Local Ollama service reachable (Tier 2 LLM)")
        else:
            providers["ollama"] = HealthProviderStatus(status="degraded", details=f"Ollama returned status {resp.status_code}")
    except Exception:
        providers["ollama"] = HealthProviderStatus(status="down", details="Local Ollama service unreachable on localhost:11434")

    # Overall system health
    all_down = all(p.status in ["down", "not_configured"] for p in providers.values())
    status = "degraded" if all_down else "healthy"

    return HealthResponse(status=status, providers=providers)

@app.get("/api/languages", response_model=LanguageListResponse)
def get_languages():
    """Derived dynamically from rag_service.LANGUAGE_CODES."""
    langs = [
        LanguageItem(name=name, code=code)
        for name, code in rag_service.LANGUAGE_CODES.items()
    ]
    return LanguageListResponse(languages=langs)

@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents():
    """Returns all currently registered documents with metadata and chunk counts."""
    docs = rag_service.list_registered_documents()
    formatted = [
        DocumentResponse(
            doc_id=d["doc_id"],
            file_name=d["file_name"],
            total_pages=d.get("total_pages", 1),
            chunk_count=d.get("chunk_count", 0),
            cached=True
        )
        for d in docs
    ]
    return DocumentListResponse(documents=formatted, total=len(formatted))

@app.post("/api/documents", response_model=list[DocumentResponse])
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Multi-file upload (PDF, DOCX, TXT).
    Returns doc_id, file_name, total_pages, chunk_count, and cached status badge.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}' for file '{file.filename}'. Allowed types: {list(ALLOWED_EXTENSIONS)}"
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is empty.")
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' exceeds maximum allowed size of {MAX_FILE_SIZE // (1024*1024)}MB."
            )

        wrapped = IngestableFile(file.filename, content)
        try:
            ingested = rag_service.ingest_document(wrapped)
            results.append(
                DocumentResponse(
                    doc_id=ingested["doc_id"],
                    file_name=ingested["file_name"],
                    total_pages=ingested["total_pages"],
                    chunk_count=ingested["chunk_count"],
                    cached=ingested.get("cached", False)
                )
            )
        except Exception as exc:
            logger.error(f"Failed to ingest document '{file.filename}': {exc}")
            raise HTTPException(status_code=500, detail=f"Document ingestion failed for '{file.filename}': {str(exc)}")

    return results

@app.delete("/api/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str):
    """Deletes a document's chunks from ChromaDB, removes registry entry, and invalidates query cache."""
    deleted = rag_service.delete_registered_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")
    return DeleteDocumentResponse(
        deleted=True,
        doc_id=doc_id,
        message=f"Document '{doc_id}' successfully removed from vector store and registry."
    )

@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint.
    Streams token events as they generate, then a final event with citations, metrics, and source.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def event_generator():
        try:
            answer_result = rag_service.answer_query(
                None,
                request.question,
                target_lang=request.language,
                stream=True,
                doc_ids=request.doc_ids
            )

            text_or_stream = answer_result[0]
            source = answer_result[1]
            citations = answer_result.citations
            metrics = answer_result.metrics

            # Stream tokens to client
            if hasattr(text_or_stream, "__iter__") and not isinstance(text_or_stream, str):
                for token in text_or_stream:
                    if token:
                        data = json.dumps({"type": "token", "content": token})
                        yield f"data: {data}\n\n"
            else:
                data = json.dumps({"type": "token", "content": str(text_or_stream)})
                yield f"data: {data}\n\n"

            # Final metadata event with citations, latency breakdown, and sources
            final_payload = {
                "type": "final",
                "citations": citations,
                "metrics": metrics,
                "source": source
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.error(f"Error in chat streaming generator: {exc}")
            err_data = json.dumps({"type": "error", "error": str(exc)})
            yield f"data: {err_data}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
