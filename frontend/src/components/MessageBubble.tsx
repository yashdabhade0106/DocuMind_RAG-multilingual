import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Copy, Check, ChevronDown, ChevronRight } from 'lucide-react';
import type { ChatMessage } from '../types';

interface Props {
  message: ChatMessage;
}

function sourceClass(source: string): string {
  const s = source.toLowerCase();
  if (s.includes('groq')) return 'groq';
  if (s.includes('ollama')) return 'ollama';
  if (s.includes('heuristic') || s.includes('scoring') || s.includes('fallback')) return 'heuristic';
  if (s.includes('sarvam')) return 'sarvam';
  return 'default';
}

function scoreColor(score: number): string {
  if (score >= 0.8) return '#10b981';
  if (score >= 0.5) return '#f59e0b';
  return '#9898b0';
}

export default function MessageBubble({ message }: Props) {
  const [copied, setCopied] = useState(false);
  const [showCitations, setShowCitations] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isTransUnavailable =
    message.source?.includes('Translation Unavailable') ||
    message.metrics?.translation_tiers?.some((t) => t.includes('Translation Unavailable'));

  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`} role="article" aria-label={`${isUser ? 'You' : 'DocuMind'} says`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? '👤' : '🤖'}
      </div>

      <div className="message-body">
        <div className="message-bubble">
          {message.streaming ? (
            <>
              <span
                style={{
                  fontFamily: "'Noto Sans Devanagari', 'Inter', sans-serif",
                  whiteSpace: 'pre-wrap',
                }}
              >
                {message.content}
              </span>
              <span className="streaming-cursor" aria-hidden="true" />
            </>
          ) : (
            <div style={{ fontFamily: "'Noto Sans Devanagari', 'Inter', sans-serif" }}>
              <ReactMarkdown>{message.content || '…'}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Translation unavailable warning */}
        {!isUser && isTransUnavailable && (
          <div className="trans-unavailable" role="alert">
            <span>⚠</span>
            <span>
              Translation unavailable — answer shown in English. Check Sarvam AI and LLM provider
              connectivity.
            </span>
          </div>
        )}

        {/* Source + Metrics row */}
        {!isUser && message.source && !message.streaming && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
            <span className={`source-tag ${sourceClass(message.source)}`}>
              {message.source.includes('Groq') && '⚡ '}
              {message.source.includes('Ollama') && '🦙 '}
              {(message.source.includes('Heuristic') || message.source.includes('Scoring')) &&
                '⚙️ '}
              {message.source.includes('Sarvam') && '🌐 '}
              {message.source}
            </span>

            {message.metrics && (
              <div className="metrics-row" aria-label="Performance metrics">
                <span className="metric-chip" title="Retrieval time">
                  🔍 {message.metrics.retrieval_ms.toFixed(0)}ms
                </span>
                <span className="metric-chip" title="Generation time">
                  ⚙ {message.metrics.generation_ms.toFixed(0)}ms
                </span>
                {message.metrics.translation_ms > 0 && (
                  <span className="metric-chip" title="Translation time">
                    🌐 {message.metrics.translation_ms.toFixed(0)}ms
                  </span>
                )}
                {message.metrics.first_token_ms != null && (
                  <span className="metric-chip" title="Time to first token">
                    ⚡ TTFT {message.metrics.first_token_ms.toFixed(0)}ms
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {/* Citations */}
        {!isUser && message.citations && message.citations.length > 0 && !message.streaming && (
          <div className="citations-section">
            <button
              id={`citations-toggle-${message.id}`}
              className="citations-toggle"
              aria-expanded={showCitations}
              aria-controls={`citations-list-${message.id}`}
              onClick={() => setShowCitations((v) => !v)}
            >
              {showCitations ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {message.citations.length} source
              {message.citations.length !== 1 ? 's' : ''} cited
            </button>

            {showCitations && (
              <div
                id={`citations-list-${message.id}`}
                className="citations-list"
                role="list"
                aria-label="Source citations"
              >
                {message.citations.map((c) => (
                  <div key={c.chunk_id} className="citation-card" role="listitem">
                    <div className="citation-header">
                      <span className="citation-doc" title={`Document: ${c.doc_name}`}>
                        📄 {c.doc_name}
                      </span>
                      <div className="citation-meta">
                        <span className="citation-page">p. {c.page}</span>
                        <span
                          className="citation-score"
                          style={{
                            color: scoreColor(c.score),
                            background: `${scoreColor(c.score)}18`,
                          }}
                        >
                          {(c.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <p className="citation-text">{c.text.slice(0, 280)}{c.text.length > 280 ? '…' : ''}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Copy button */}
        {!isUser && !message.streaming && (
          <div className="message-actions">
            <button
              id={`copy-msg-${message.id}`}
              className="copy-btn"
              onClick={handleCopy}
              aria-label="Copy response to clipboard"
              title="Copy"
            >
              {copied ? <Check size={11} /> : <Copy size={11} />}
              <span>{copied ? 'Copied!' : 'Copy'}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
