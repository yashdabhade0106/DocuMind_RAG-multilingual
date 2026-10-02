import { Trash2 } from 'lucide-react';
import type { DocumentInfo } from '../types';

interface Props {
  documents: DocumentInfo[];
  onDelete: (docId: string) => void;
  selectedIds: string[];
  onToggleSelect: (docId: string) => void;
}

const extIcon = (name: string) => {
  if (name.endsWith('.pdf')) return '📄';
  if (name.endsWith('.docx')) return '📝';
  return '📃';
};

export default function DocumentList({ documents, onDelete, selectedIds, onToggleSelect }: Props) {
  if (documents.length === 0) {
    return (
      <div
        style={{
          textAlign: 'center',
          padding: '12px 0',
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
        }}
      >
        No documents uploaded yet.
      </div>
    );
  }

  return (
    <div className="doc-list" role="list" aria-label="Uploaded documents">
      {documents.map((doc) => {
        const isSelected = selectedIds.includes(doc.doc_id);
        return (
          <div
            key={doc.doc_id}
            className="doc-item"
            role="listitem"
            style={{
              borderColor: isSelected ? 'var(--accent)' : undefined,
              background: isSelected ? 'var(--accent-glow)' : undefined,
            }}
            onClick={() => onToggleSelect(doc.doc_id)}
            title={`${doc.file_name} · ${doc.chunk_count} chunks · ${doc.total_pages} pages`}
          >
            <span className="doc-item-icon" aria-hidden="true">
              {extIcon(doc.file_name)}
            </span>
            <div className="doc-item-info">
              <div className="doc-item-name">{doc.file_name}</div>
              <div className="doc-item-meta">
                {doc.chunk_count} chunks · {doc.total_pages} page{doc.total_pages !== 1 ? 's' : ''}
              </div>
            </div>
            <div className="doc-item-badges">
              {doc.cached && (
                <span className="cached-badge" title="Served from disk cache">
                  ⚡ cached
                </span>
              )}
              <button
                id={`delete-doc-${doc.doc_id}`}
                className="doc-delete-btn"
                aria-label={`Delete ${doc.file_name}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(doc.doc_id);
                }}
                title="Remove document"
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
