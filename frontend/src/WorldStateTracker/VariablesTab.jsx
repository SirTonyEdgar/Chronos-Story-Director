import React, { useState } from 'react';
import { 
  Plus, Trash2, Globe, Activity, 
  AlertTriangle, ChevronRight, Search 
} from 'lucide-react';

/**
 * VariablesTab (Master-Detail Version)
 * Manages abstract global states and logic rules.
 */
export default function VariablesTab({ state, setState }) {
  const vars = state["World Variables"] || [];
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");

  // --- DERIVED DATA ---
  const activeVar = vars[selectedIdx] || {};

  const filteredVars = vars.map((v, i) => ({ ...v, originalIndex: i }))
    .filter(v => (v.Name || "").toLowerCase().includes(searchTerm.toLowerCase()));

  // --- HANDLERS ---

  const handleAdd = () => {
    const newVar = { Name: "New Variable", Value: "0", Mechanic: "Define logic..." };
    const newList = [...vars, newVar];
    setState({ ...state, "World Variables": newList });
    setSelectedIdx(newList.length - 1);
  };

  const updateActive = (field, val) => {
    if (selectedIdx === null || !vars[selectedIdx]) return;
    const updated = [...vars];
    updated[selectedIdx] = { ...updated[selectedIdx], [field]: val };
    setState({ ...state, "World Variables": updated });
  };

  const handleDelete = () => {
    if (!confirm("Delete this world variable?")) return;
    const updated = vars.filter((_, i) => i !== selectedIdx);
    setState({ ...state, "World Variables": updated });
    setSelectedIdx(0);
  };

  return (
    <div style={styles.container}>
      
      {/* HEADER */}
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <Globe size={20} style={{ marginTop: '2px', color: '#60a5fa' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Global World Mechanics</strong>
            <p style={{ margin: '4px 0 0', color: '#93c5fd', fontSize: '13px', lineHeight: '1.4' }}>
              Define abstract states that affect the entire story (e.g., "Economy", "Chaos Level"). 
              The AI uses the <b>Logic Rule</b> to determine consequences for characters.
            </p>
          </div>
        </div>
      </div>

      <div style={styles.layout}>
        
        {/* --- SIDEBAR (LIST) --- */}
        <div style={styles.sidebar}>
          <div style={styles.toolbar}>
            <div style={styles.searchWrapper}>
              <Search size={14} color="#777" />
              <input 
                placeholder="Search..." 
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                style={styles.searchInput}
              />
            </div>
            <button onClick={handleAdd} style={styles.addBtn}>
              <Plus size={16} />
            </button>
          </div>

          <div style={styles.list}>
            {filteredVars.length === 0 && (
              <div style={styles.emptyList}>No variables found.</div>
            )}
            {filteredVars.map((v) => (
              <div 
                key={v.originalIndex}
                onClick={() => setSelectedIdx(v.originalIndex)}
                style={{
                  ...styles.listItem,
                  background: selectedIdx === v.originalIndex ? '#3b82f6' : 'transparent',
                  color: selectedIdx === v.originalIndex ? '#fff' : '#a1a1aa'
                }}
              >
                <div style={{ flex: 1, overflow: 'hidden' }}>
                  <div style={{ fontWeight: '600', fontSize: '13px', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                    {v.Name || "Untitled"}
                  </div>
                  <div style={{ fontSize: '11px', opacity: 0.8, display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Activity size={10} /> {v.Value || "N/A"}
                  </div>
                </div>
                {selectedIdx === v.originalIndex && <ChevronRight size={14} />}
              </div>
            ))}
          </div>
        </div>

        {/* --- MAIN CONTENT (DETAIL) --- */}
        <div style={styles.main}>
          {vars.length > 0 && activeVar ? (
            <div style={styles.editor}>
              
              {/* Top Row: Name & Value */}
              <div style={styles.row}>
                <div style={{ flex: 2 }}>
                  <label style={styles.label}>VARIABLE NAME</label>
                  <input 
                    value={activeVar.Name || ""}
                    onChange={e => updateActive("Name", e.target.value)}
                    style={styles.input}
                    placeholder="e.g. Corruption Level"
                  />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={styles.label}>CURRENT VALUE / STATE</label>
                  <input 
                    value={activeVar.Value || ""}
                    onChange={e => updateActive("Value", e.target.value)}
                    style={{ ...styles.input, color: '#4ade80', fontWeight: 'bold' }}
                    placeholder="e.g. 75%"
                  />
                </div>
              </div>

              {/* Logic Rule */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <label style={styles.label}>LOGIC / CONSEQUENCE RULE (AI INSTRUCTION)</label>
                <textarea 
                  value={activeVar.Mechanic || ""}
                  onChange={e => updateActive("Mechanic", e.target.value)}
                  style={styles.textArea}
                  placeholder="Example: If Corruption > 50, guards will demand bribes. If > 80, open riots occur."
                />
              </div>

              {/* Footer Actions */}
              <div style={styles.footer}>
                <div style={styles.tip}>
                  <AlertTriangle size={14} color="#f59e0b" />
                  <span>This rule applies globally to all scenes.</span>
                </div>
                <button onClick={handleDelete} style={styles.deleteBtn}>
                  <Trash2 size={14} /> Delete
                </button>
              </div>

            </div>
          ) : (
            <div style={styles.emptyState}>Select or create a variable to define rules.</div>
          )}
        </div>

      </div>
    </div>
  );
}

// --- STYLES ---
const styles = {
  container: { padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' },
  infoBox: { background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)', padding: '16px', borderRadius: '8px', marginBottom: '20px' },
  
  layout: { display: 'flex', gap: '20px', flex: 1, minHeight: 0 },
  
  // Sidebar
  sidebar: { width: '240px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #27272a', paddingRight: '15px' },
  toolbar: { display: 'flex', gap: '10px', marginBottom: '15px' },
  searchWrapper: { display: 'flex', alignItems: 'center', background: '#09090b', border: '1px solid #333', borderRadius: '6px', padding: '8px', flex: 1 },
  searchInput: { background: 'transparent', border: 'none', color: '#fff', outline: 'none', fontSize: '12px', width: '100%' },
  addBtn: { background: '#2563eb', color: '#fff', border: 'none', padding: '8px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' },
  
  list: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' },
  listItem: { padding: '10px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', transition: 'all 0.2s' },
  emptyList: { color: '#555', fontSize: '13px', fontStyle: 'italic', textAlign: 'center', marginTop: '20px' },

  // Main Editor
  main: { flex: 1, display: 'flex', flexDirection: 'column', background: '#131315', borderRadius: '8px', border: '1px solid #27272a', overflow: 'hidden' },
  editor: { padding: '30px', display: 'flex', flexDirection: 'column', gap: '25px', height: '100%', boxSizing: 'border-box' },
  emptyState: { margin: 'auto', color: '#555', fontSize: '14px' },

  row: { display: 'flex', gap: '20px' },
  label: { fontSize: '11px', color: '#71717a', fontWeight: '700', display: 'block', marginBottom: '8px', letterSpacing: '0.5px' },
  input: { width: '100%', padding: '12px', background: '#09090b', border: '1px solid #333', color: '#fff', borderRadius: '6px', outline: 'none', fontSize: '14px', boxSizing: 'border-box' },
  textArea: { flex: 1, width: '100%', padding: '15px', background: '#09090b', border: '1px solid #333', color: '#e4e4e7', borderRadius: '6px', resize: 'none', fontSize: '14px', lineHeight: '1.6', boxSizing: 'border-box', fontFamily: 'monospace' },

  footer: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 'auto', paddingTop: '20px', borderTop: '1px solid #27272a' },
  tip: { display: 'flex', gap: '8px', alignItems: 'center', fontSize: '12px', color: '#71717a' },
  deleteBtn: { background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }
};