import { useRef, useState } from 'react';
import type { DocumentInfo } from '../types';
import { uploadDocuments } from '../api';

interface Props {
  onUploaded: (docs: DocumentInfo[]) => void;
}

export default function DropZone({ onUploaded }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const ALLOWED = ['.pdf', '.docx', '.txt'];

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const arr = Array.from(files).filter((f) =>
      ALLOWED.some((ext) => f.name.toLowerCase().endsWith(ext)),
    );
    if (arr.length === 0) {
      setError('Only PDF, DOCX, and TXT files are supported.');
      return;
    }
    setError(null);
    setUploading(true);
    setProgress(10);

    // Fake progress ticks while upload is in-flight
    const ticker = setInterval(() => setProgress((p) => Math.min(p + 12, 85)), 600);

    try {
      const docs = await uploadDocuments(arr);
      clearInterval(ticker);
      setProgress(100);
      setTimeout(() => { setProgress(0); setUploading(false); }, 600);
      onUploaded(docs);
    } catch (err: unknown) {
      clearInterval(ticker);
      setUploading(false);
      setProgress(0);
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  return (
    <div>
      <div
        id="dropzone"
        className={`dropzone ${dragOver ? 'drag-over' : ''}`}
        role="button"
        aria-label="Upload documents by clicking or dragging files"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onDragEnter={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => !uploading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          id="file-input"
          multiple
          accept=".pdf,.docx,.txt"
          aria-label="File upload input"
          onChange={(e) => handleFiles(e.target.files)}
          style={{ display: 'none' }}
        />
        <div className="dropzone-icon">
          {uploading ? '⏳' : dragOver ? '📂' : '📁'}
        </div>
        <div className="dropzone-text">
          {uploading ? 'Uploading…' : dragOver ? 'Drop to upload' : 'Click or drag files here'}
        </div>
        <div className="dropzone-hint">PDF · DOCX · TXT · up to 50MB each</div>

        {uploading && (
          <div className="upload-progress">
            <div className="upload-progress-bar">
              <div className="upload-progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}
      </div>

      {error && (
        <div
          style={{
            fontSize: '0.72rem',
            color: 'var(--error)',
            marginTop: 6,
            display: 'flex',
            gap: 4,
            alignItems: 'center',
          }}
          role="alert"
        >
          ⚠ {error}
        </div>
      )}
    </div>
  );
}
