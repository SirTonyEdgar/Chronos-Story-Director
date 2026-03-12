import React, { useState, useEffect } from 'react';
import { CheckCircle2, XCircle, AlertTriangle, Info, X, AlertCircle } from 'lucide-react';

let _addToast = null;
let _showConfirm = null;

export function toast(message, type = 'info') {
  if (_addToast) {
    _addToast({ message, type, id: Date.now() + Math.random() });
  }
}

export function confirm(message, options = {}) {
  return new Promise((resolve) => {
    if (_showConfirm) {
      _showConfirm({ message, options, resolve });
    } else {
      resolve(window.confirm(message));
    }
  });
}

const TOAST_CONFIG = {
  success: {
    icon: <CheckCircle2 size={16} />,
    bg: '#0f2a1a', border: '#16a34a', color: '#4ade80', duration: 3500,
  },
  error: {
    icon: <XCircle size={16} />,
    bg: '#2a0a0a', border: '#dc2626', color: '#f87171', duration: 5000,
  },
  warning: {
    icon: <AlertTriangle size={16} />,
    bg: '#2a1a00', border: '#d97706', color: '#fbbf24', duration: 4000,
  },
  info: {
    icon: <Info size={16} />,
    bg: '#0a1a2a', border: '#3b82f6', color: '#60a5fa', duration: 3500,
  },
};

export function NotificationProvider() {
  const [toasts, setToasts] = useState([]);
  const [confirmState, setConfirmState] = useState(null);

  useEffect(() => {
    _addToast = ({ message, type, id }) => {
      const cfg = TOAST_CONFIG[type] || TOAST_CONFIG.info;
      setToasts(prev => [...prev, { message, type, id }]);
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== id));
      }, cfg.duration);
    };
    _showConfirm = (state) => setConfirmState(state);
    return () => { _addToast = null; _showConfirm = null; };
  }, []);

  const dismissToast = (id) => setToasts(prev => prev.filter(t => t.id !== id));

  const handleConfirmResult = (result) => {
    if (confirmState) confirmState.resolve(result);
    setConfirmState(null);
  };

  const { title, confirmLabel, danger } = confirmState?.options || {};

  return (
    <>
      <div style={styles.toastStack}>
        {toasts.map(t => {
          const cfg = TOAST_CONFIG[t.type] || TOAST_CONFIG.info;
          return (
            <div key={t.id} style={{ ...styles.toast, background: cfg.bg, border: `1px solid ${cfg.border}`, animation: 'chronos-slide-in 0.2s ease-out' }}>
              <span style={{ color: cfg.color, display: 'flex', flexShrink: 0 }}>{cfg.icon}</span>
              <span style={styles.toastMessage}>{t.message}</span>
              <button onClick={() => dismissToast(t.id)} style={{ ...styles.toastClose, color: cfg.color }}>
                <X size={13} />
              </button>
            </div>
          );
        })}
      </div>

      {confirmState && (
        <div style={styles.overlay}>
          <div style={styles.modal}>
            <div style={styles.modalIcon}>
              <AlertCircle size={22} color={danger ? '#ef4444' : '#f59e0b'} />
            </div>
            <div style={styles.modalBody}>
              {title && <h3 style={styles.modalTitle}>{title}</h3>}
              <p style={styles.modalMessage}>{confirmState.message}</p>
            </div>
            <div style={styles.modalActions}>
              <button onClick={() => handleConfirmResult(false)} style={styles.cancelBtn}>Cancel</button>
              <button onClick={() => handleConfirmResult(true)} style={{ ...styles.confirmBtn, background: danger ? '#dc2626' : '#3b82f6' }}>
                {confirmLabel || 'Confirm'}
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes chronos-slide-in {
          from { opacity: 0; transform: translateX(20px); }
          to   { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </>
  );
}

const styles = {
  toastStack: { position: 'fixed', bottom: '90px', right: '24px', zIndex: 9999, display: 'flex', flexDirection: 'column', gap: '8px', pointerEvents: 'none', alignItems: 'flex-end' },
  toast: { display: 'flex', alignItems: 'center', gap: '10px', padding: '11px 14px', borderRadius: '8px', fontSize: '13px', fontWeight: '500', color: '#e4e4e7', boxShadow: '0 4px 16px rgba(0,0,0,0.6)', pointerEvents: 'auto', maxWidth: '380px', minWidth: '240px' },
  toastMessage: { flex: 1, lineHeight: '1.4' },
  toastClose: { background: 'none', border: 'none', cursor: 'pointer', padding: '2px', display: 'flex', alignItems: 'center', opacity: 0.7, flexShrink: 0 },
  overlay: { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', zIndex: 10000, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(2px)' },
  modal: { background: '#111', border: '1px solid #2a2a2a', borderRadius: '10px', padding: '28px', width: '420px', maxWidth: '90vw', boxShadow: '0 20px 60px rgba(0,0,0,0.8)', display: 'flex', flexDirection: 'column', gap: '16px' },
  modalIcon: { display: 'flex' },
  modalBody: { display: 'flex', flexDirection: 'column', gap: '6px' },
  modalTitle: { margin: 0, fontSize: '16px', fontWeight: '700', color: '#fff' },
  modalMessage: { margin: 0, fontSize: '14px', color: '#a1a1aa', lineHeight: '1.6' },
  modalActions: { display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '4px' },
  cancelBtn: { padding: '9px 18px', background: 'transparent', border: '1px solid #3f3f46', color: '#a1a1aa', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', fontWeight: '600' },
  confirmBtn: { padding: '9px 18px', border: 'none', color: '#fff', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', fontWeight: '700' },
};