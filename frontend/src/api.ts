import type {
  DocumentInfo,
  LanguageItem,
  HealthResponse,
  SSEEvent,
} from './types';

const BASE = '/api';

export async function fetchLanguages(): Promise<LanguageItem[]> {
  const res = await fetch(`${BASE}/languages`);
  if (!res.ok) throw new Error('Failed to fetch languages');
  const data = await res.json();
  return data.languages as LanguageItem[];
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error('Failed to fetch health');
  return res.json();
}

export async function fetchDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${BASE}/documents`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  const data = await res.json();
  return data.documents as DocumentInfo[];
}

export async function uploadDocuments(files: File[]): Promise<DocumentInfo[]> {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  const res = await fetch(`${BASE}/documents`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail ?? 'Upload failed');
  }
  return res.json();
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${docId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete document');
}

/**
 * Stream a chat answer via SSE.
 * Calls onToken for each streamed token, onFinal when the final event arrives.
 * Returns an AbortController so the caller can cancel mid-stream.
 */
export function streamChat(
  question: string,
  language: string,
  docIds: string[] | null,
  onToken: (token: string) => void,
  onFinal: (event: SSEEvent & { type: 'final' }) => void,
  onError: (err: string) => void,
): AbortController {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, language, doc_ids: docIds }),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ detail: 'Stream failed' }));
        onError(err.detail ?? 'Stream failed');
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') return;
          try {
            const event = JSON.parse(raw) as SSEEvent;
            if (event.type === 'token') onToken(event.content);
            else if (event.type === 'final') onFinal(event);
            else if (event.type === 'error') onError(event.error);
          } catch {
            // ignore malformed lines
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      onError(err instanceof Error ? err.message : 'Connection error');
    }
  })();

  return controller;
}
