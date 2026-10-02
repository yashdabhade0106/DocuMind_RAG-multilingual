from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    doc_id: str = Field(..., description="Unique document ID (hash-derived or assigned)")
    file_name: str = Field(..., description="Original filename")
    total_pages: int = Field(..., description="Total pages or sections extracted")
    chunk_count: int = Field(..., description="Number of chunks generated and stored")
    cached: bool = Field(default=False, description="Whether this document was served from on-disk cache")


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class DeleteDocumentResponse(BaseModel):
    deleted: bool
    doc_id: str
    message: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question to answer")
    language: str = Field(default="English", description="Target answer language")
    doc_ids: list[str] | None = Field(default=None, description="Optional list of doc_ids to filter context")


class CitationItem(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    text: str
    score: float


class MetricsData(BaseModel):
    retrieval_ms: float
    generation_ms: float
    first_token_ms: float | None = None
    translation_ms: float
    total_ms: float
    llm_provider: str
    translation_tiers: list[str] = []
    target_lang: str


class ChatFinalEvent(BaseModel):
    citations: list[CitationItem]
    metrics: MetricsData
    source: str


class LanguageItem(BaseModel):
    name: str
    code: str


class LanguageListResponse(BaseModel):
    languages: list[LanguageItem]


class HealthProviderStatus(BaseModel):
    status: str  # "ok", "degraded", "down", "not_configured"
    details: str | None = None


class HealthResponse(BaseModel):
    status: str  # "healthy", "degraded"
    providers: dict[str, HealthProviderStatus]
