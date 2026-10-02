import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';
import { Send, Square, PlusCircle, Sun, Moon } from 'lucide-react';

import type { ChatMessage, DocumentInfo, HealthResponse, LanguageItem } from './types';
import {
  fetchLanguages,
  fetchHealth,
  fetchDocuments,
  deleteDocument,
  streamChat,
} from './api';
import HealthPanel from './components/HealthPanel';
import DropZone from './components/DropZone';
import DocumentList from './components/DocumentList';
import MessageBubble from './components/MessageBubble';

const SAMPLE_QUESTIONS = [
  'What is this document about?',
  'Summarize the key points briefly.',
  'What are the main findings?',
];

let msgCounter = 0;
const uid = () => `msg-${++msgCounter}-${Date.now()}`;

export default function App() {
  // Theme
  const [lightMode, setLightMode] = useState(false);

  // Backend state
  const [languages, setLanguages] = useState<LanguageItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [targetLang, setTargetLang] = useState('English');

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Abort controller for stop generation
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Initial data fetch
  useEffect(() => {
    fetchLanguages()
      .then(setLanguages)
      .catch(() => setLanguages([{ name: 'English', code: 'en-IN' }]));
    fetchHealth()
      .then(setHealth)
      .catch(() => {});
    fetchDocuments()
      .then(setDocuments)
      .catch(() => {});

    // Poll health every 30s
    const healthInterval = setInterval(() => {
      fetchHealth().then(setHealth).catch(() => {});
    }, 30_000);

    return () => clearInterval(healthInterval);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Apply theme
  useEffect(() => {
    document.documentElement.classList.toggle('light-mode', lightMode);
  }, [lightMode]);

  // Auto-resize textarea
  const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  };

  const handleUpload = useCallback((newDocs: DocumentInfo[]) => {
    setDocuments((prev) => {
      const existingIds = new Set(prev.map((d) => d.doc_id));
      const merged = [...prev];
      for (const d of newDocs) {
        if (existingIds.has(d.doc_id)) {
          const idx = merged.findIndex((x) => x.doc_id === d.doc_id);
          merged[idx] = d;
        } else {
          merged.push(d);
        }
      }
      return merged;
    });
    // Auto-select newly uploaded docs
    setSelectedDocIds((prev) => {
      const newIds = newDocs.map((d) => d.doc_id);
      return [...new Set([...prev, ...newIds])];
    });
  }, []);

  const handleDeleteDoc = useCallback(async (docId: string) => {
    await deleteDocument(docId).catch(() => {});
    setDocuments((prev) => prev.filter((d) => d.doc_id !== docId));
    setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
  }, []);

  const handleToggleSelect = useCallback((docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId],
    );
  }, []);

  const handleNewChat = () => {
    if (abortRef.current) abortRef.current.abort();
    setMessages([]);
    setInput('');
    setError(null);
    setStreaming(false);
  };

  const handleStopGeneration = () => {
    abortRef.current?.abort();
    setStreaming(false);
    // Finalize streaming message
    setMessages((prev) =>
      prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)),
    );
  };

  const sendMessage = useCallback(
    (question: string) => {
      if (!question.trim() || streaming) return;
      setError(null);

      const userMsg: ChatMessage = { id: uid(), role: 'user', content: question };
      const assistantMsgId = uid();
      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput('');
      setStreaming(true);
      if (textareaRef.current) textareaRef.current.style.height = 'auto';

      const docIds = selectedDocIds.length > 0 ? selectedDocIds : null;

      abortRef.current = streamChat(
        question,
        targetLang,
        docIds,
        // onToken
        (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, content: m.content + token } : m,
            ),
          );
        },
        // onFinal
        (ev) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? {
                    ...m,
                    streaming: false,
                    source: ev.source,
                    citations: ev.citations,
                    metrics: ev.metrics,
                  }
                : m,
            ),
          );
          setStreaming(false);
        },
        // onError
        (err) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, streaming: false, content: m.content || err, error: true }
                : m,
            ),
          );
          setError(err);
          setStreaming(false);
        },
      );
    },
    [streaming, selectedDocIds, targetLang],
  );

  const handleSend = () => sendMessage(input);
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={lightMode ? 'light-mode' : ''}>
      {/* Ambient background orbs */}
      <div className="bg-orb bg-orb-1" aria-hidden="true" />
      <div className="bg-orb bg-orb-2" aria-hidden="true" />

      <div className="layout">
        {/* ===== SIDEBAR ===== */}
        <aside className="sidebar" aria-label="Document management sidebar">
          {/* Logo */}
          <div className="sidebar-header">
            <div className="logo-row">
              <div className="logo-icon" aria-hidden="true">🧠</div>
              <span className="logo-name">DocuMind RAG</span>
            </div>
            <div className="logo-tagline">Multilingual · Hand-rolled · Zero-framework</div>
          </div>

          {/* Language selector */}
          <div className="sidebar-section">
            <div className="sidebar-label">Answer Language</div>
            <select
              id="language-selector"
              className="lang-select"
              value={targetLang}
              onChange={(e) => setTargetLang(e.target.value)}
              aria-label="Select answer language"
            >
              {languages.length > 0
                ? languages.map((l) => (
                    <option key={l.code} value={l.name}>
                      {l.name}
                    </option>
                  ))
                : <option value="English">English</option>}
            </select>
          </div>

          {/* Upload */}
          <div className="sidebar-section">
            <div className="sidebar-label">Upload Documents</div>
            <DropZone onUploaded={handleUpload} />
          </div>

          {/* Document list */}
          <div className="sidebar-section" style={{ flex: 1, overflow: 'hidden' }}>
            <div className="sidebar-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>Documents ({documents.length})</span>
              {selectedDocIds.length > 0 && (
                <span style={{ color: 'var(--accent-bright)', fontSize: '0.68rem' }}>
                  {selectedDocIds.length} selected
                </span>
              )}
            </div>
            <DocumentList
              documents={documents}
              onDelete={handleDeleteDoc}
              selectedIds={selectedDocIds}
              onToggleSelect={handleToggleSelect}
            />
          </div>

          {/* Provider health */}
          <div className="sidebar-section">
            <div className="sidebar-label">Provider Status</div>
            <HealthPanel health={health} />
          </div>

          {/* Footer */}
          <div className="sidebar-footer">
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
              v2.0 · No LangChain
            </span>
            <button
              id="theme-toggle"
              className="theme-btn"
              onClick={() => setLightMode((v) => !v)}
              aria-label={lightMode ? 'Switch to dark mode' : 'Switch to light mode'}
              title={lightMode ? 'Dark mode' : 'Light mode'}
            >
              {lightMode ? <Moon size={13} /> : <Sun size={13} />}
              <span>{lightMode ? 'Dark' : 'Light'}</span>
            </button>
          </div>
        </aside>

        {/* ===== MAIN CHAT AREA ===== */}
        <main className="chat-area" aria-label="Chat interface">
          {/* Top bar */}
          <div className="topbar">
            <div>
              <div className="topbar-title">
                {selectedDocIds.length > 0
                  ? `Chatting with ${selectedDocIds.length} document${selectedDocIds.length !== 1 ? 's' : ''}`
                  : 'DocuMind Chat'}
              </div>
              <div className="topbar-meta">
                {targetLang !== 'English' ? `Answering in ${targetLang}` : 'Answering in English'}
              </div>
            </div>
            <div className="topbar-actions">
              <button
                id="new-chat-btn"
                className="icon-btn"
                onClick={handleNewChat}
                aria-label="Start new chat"
                title="New chat"
              >
                <PlusCircle size={14} />
                New Chat
              </button>
            </div>
          </div>

          {/* Error banner */}
          {error && (
            <div className="error-banner" role="alert">
              <span>⚠</span>
              <span>{error}</span>
            </div>
          )}

          {/* Messages */}
          <div
            className="messages-container"
            id="messages-container"
            role="log"
            aria-label="Chat messages"
            aria-live="polite"
          >
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon" aria-hidden="true">🧠</div>
                <h1 className="empty-state-title">Ask your documents anything</h1>
                <p className="empty-state-desc">
                  Upload a PDF, DOCX, or TXT file to begin. DocuMind retrieves the most relevant
                  passages and generates a streaming answer — in English, Hindi, or Marathi.
                </p>
                {documents.length > 0 && (
                  <>
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 12 }}>
                      Try one of these to get started:
                    </p>
                    <div className="empty-pills">
                      {SAMPLE_QUESTIONS.map((q) => (
                        <button
                          key={q}
                          className="empty-pill"
                          onClick={() => sendMessage(q)}
                          aria-label={`Ask: ${q}`}
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            ) : (
              messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Loading indicator (when generating but no tokens yet) */}
          {streaming && messages.at(-1)?.content === '' && (
            <div style={{ padding: '0 24px 8px', display: 'flex', alignItems: 'center', gap: 10 }}>
              <div className="message-avatar" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', width: 28, height: 28, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem' }} aria-hidden="true">🤖</div>
              <div className="loading-dots" aria-label="Generating response">
                <div className="loading-dot" />
                <div className="loading-dot" />
                <div className="loading-dot" />
              </div>
            </div>
          )}

          {/* Input */}
          <div className="input-area">
            <div className="input-row">
              <textarea
                ref={textareaRef}
                id="chat-input"
                className="chat-input"
                placeholder={
                  documents.length === 0
                    ? 'Upload a document first…'
                    : `Ask something${targetLang !== 'English' ? ` (answer in ${targetLang})` : ''}…`
                }
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={streaming}
                aria-label="Type your question"
                aria-multiline="true"
              />

              {streaming ? (
                <button
                  id="stop-generation-btn"
                  className="stop-btn"
                  onClick={handleStopGeneration}
                  aria-label="Stop generation"
                  title="Stop generation"
                >
                  <Square size={15} />
                </button>
              ) : (
                <button
                  id="send-btn"
                  className="send-btn"
                  onClick={handleSend}
                  disabled={!input.trim() || streaming}
                  aria-label="Send message"
                  title="Send (Enter)"
                >
                  <Send size={15} />
                </button>
              )}
            </div>
            <div className="input-hint">
              Enter to send · Shift+Enter for new line
              {selectedDocIds.length === 0 && documents.length > 0
                ? ' · Click documents to filter context'
                : ''}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
