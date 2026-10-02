import { Wifi } from 'lucide-react';
import type { HealthResponse } from '../types';

interface Props {
  health: HealthResponse | null;
}

const icons: Record<string, string> = {
  groq: '⚡',
  sarvam: '🌐',
  ollama: '🦙',
};

const statusClass: Record<string, string> = {
  ok: 'badge-ok',
  degraded: 'badge-degraded',
  down: 'badge-down',
  not_configured: 'badge-not_configured',
};

const statusLabel: Record<string, string> = {
  ok: 'OK',
  degraded: 'DEGRADED',
  down: 'DOWN',
  not_configured: 'NOT SET',
};

export default function HealthPanel({ health }: Props) {
  if (!health) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--text-muted)', fontSize: '0.78rem' }}>
        <Wifi size={12} />
        Checking providers...
      </div>
    );
  }

  return (
    <div>
      {Object.entries(health.providers).map(([name, info]) => (
        <div key={name} className="provider-row">
          <span className="provider-name">
            <span>{icons[name] ?? '🔧'}</span>
            <span style={{ textTransform: 'capitalize' }}>{name}</span>
          </span>
          <span
            className={`provider-badge ${statusClass[info.status] ?? 'badge-not_configured'}`}
            title={info.details ?? ''}
          >
            {statusLabel[info.status] ?? info.status}
          </span>
        </div>
      ))}
    </div>
  );
}
