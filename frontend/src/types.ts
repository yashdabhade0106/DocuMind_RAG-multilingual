// Central API types derived from backend Pydantic schemas

export interface DocumentInfo {
  doc_id: string;
  file_name: string;
  total_pages: number;
  chunk_count: number;
  cached: boolean;
}

export interface LanguageItem {
  name: string;
  code: string;
}

export interface HealthProviderStatus {
  status: 'ok' | 'degraded' | 'down' | 'not_configured';
  details?: string;
}

export interface HealthResponse {
  status: string;
  providers: Record<string, HealthProviderStatus>;
}

export interface Citation {
  chunk_id: string;
  doc_id: string;
  doc_name: string;
  page: number;
  text: string;
  score: number;
}

export interface MetricsData {
  retrieval_ms: number;
  generation_ms: number;
  first_token_ms: number | null;
  translation_ms: number;
  total_ms: number;
  llm_provider: string;
  translation_tiers: string[];
  target_lang: string;
}

export interface ChatFinalEvent {
  citations: Citation[];
  metrics: MetricsData;
  source: string;
}

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  source?: string;
  citations?: Citation[];
  metrics?: MetricsData;
  streaming?: boolean;
  error?: boolean;
}

export interface SSETokenEvent {
  type: 'token';
  content: string;
}

export interface SSEFinalEvent extends ChatFinalEvent {
  type: 'final';
}

export interface SSEErrorEvent {
  type: 'error';
  error: string;
}

export type SSEEvent = SSETokenEvent | SSEFinalEvent | SSEErrorEvent;
