import React, { useState, useEffect } from 'react';
import { FileJson, AlertTriangle, CheckCircle2 } from 'lucide-react';

/**
 * JsonTab Component
 * Allows power users to directly edit the raw JSON state of the simulation.
 */
export default function JsonTab({ state, setState }) {
  const [text, setText] = useState("");
  const [error, setError] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  // Sync local text with global state on load (or external update)
  useEffect(() => {
    if (state) {
      setText(JSON.stringify(state, null, 4));
    }
  }, [state]);

  const handleApply = () => {
    try {
      const parsed = JSON.parse(text);
      setState(parsed); // Update global state
      setError(null);
      setSuccessMsg("State successfully overridden from JSON.");
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMsg(null), 3000);
    } catch (err) {
      setError(err.message);
      setSuccessMsg(null);
    }
  };

  return (
    <div style={styles.container}>
      
      {/* --- INFO HEADER --- */}
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <FileJson size={20} style={{ marginTop: '2px' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Raw State Editor</strong>
            <p style={{ margin: '4px 0 0', color: '#93c5fd', fontSize: '13px', lineHeight: '1.4' }}>
              Power User Mode. Directly manipulate the simulation state tree. 
              Click <b>"Apply Override"</b> to update the application memory (this turns the main Save button green).
            </p>
          </div>
        </div>
      </div>

      {/* --- TOOLBAR --- */}
      <div style={styles.toolbar}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <h3 style={styles.header}>JSON Source</h3>
          {/* Status Indicators */}
          {error && (
            <span style={{ color: '#ef4444', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <AlertTriangle size={14} /> Syntax Error: {error}
            </span>
          )}
          {successMsg && (
            <span style={{ color: '#22c55e', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <CheckCircle2 size={14} /> {successMsg}
            </span>
          )}
        </div>

        <button onClick={handleApply} style={styles.applyBtn}>
          Apply JSON Override
        </button>
      </div>
      
      {/* --- EDITOR AREA --- */}
      <textarea 
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          if (error) setError(null);
        }}
        style={styles.textArea}
        spellCheck="false"
      />
    </div>
  );
}

/**
 * Component Styles
 */
const styles = {
  container: { 
    padding: '10px',
    height: '100%',
    display: 'flex',
    flexDirection: 'column'
  },
  
  // Info Box
  infoBox: {
    background: 'rgba(59, 130, 246, 0.1)', 
    border: '1px solid rgba(59, 130, 246, 0.2)', 
    padding: '16px', 
    borderRadius: '8px', 
    marginBottom: '20px',
    flexShrink: 0
  },

  // Toolbar
  toolbar: {
    display: 'flex', 
    justifyContent: 'space-between', 
    alignItems: 'center', 
    marginBottom: '15px',
    flexShrink: 0
  },
  header: { 
    fontSize: '18px', 
    fontWeight: '700', 
    color: '#fff', 
    margin: 0
  },

  // Buttons
  applyBtn: { 
    background: '#ef4444',
    color: '#fff', 
    border: 'none', 
    padding: '8px 16px', 
    borderRadius: '6px', 
    cursor: 'pointer', 
    fontWeight: '700', 
    fontSize: '13px',
    transition: 'background 0.2s',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
  },

  // Editor
  textArea: { 
    width: '100%', 
    height: '600px',
    background: '#09090b', 
    color: '#22c55e',
    border: '1px solid #333', 
    borderRadius: '6px', 
    padding: '15px', 
    fontFamily: '"Fira Code", "Consolas", monospace',
    fontSize: '13px', 
    lineHeight: '1.5',
    resize: 'vertical',
    outline: 'none',
    boxSizing: 'border-box'
  }
};