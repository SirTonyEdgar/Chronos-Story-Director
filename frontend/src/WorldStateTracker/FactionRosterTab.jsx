import React, { useState } from 'react';
import { Plus, Trash2, Search, Network } from 'lucide-react';
import { TimelineDropdown } from '../components/SharedComponents';
import { confirm } from '../components/Notifications';

const ROLES = ['Main POV', 'Supporting', 'Antagonist', 'Neutral'];
const STATUSES = ['Active', 'Compromised', 'Dissolved', 'Unknown'];

const getRoleColor = (role) => {
  switch (role) {
    case 'Main POV': return '#ffd700';
    case 'Supporting': return '#22c55e';
    case 'Antagonist': return '#ef4444';
    default: return '#52525b';
  }
};

const getStatusColor = (status) => {
  switch (status) {
    case 'Active': return '#22c55e';
    case 'Compromised': return '#f59e0b';
    case 'Dissolved': return '#ef4444';
    default: return '#52525b';
  }
};

export default function FactionRosterTab({ state, setState }) {
  const factions = state.Factions || [];
  const availableTimelines = state.Timelines || [];
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");

  const activeFaction = selectedIdx !== null ? factions[selectedIdx] : null;

  const filteredFactions = factions
    .map((f, i) => ({ ...f, _idx: i }))
    .filter(f => (f.Name || "").toLowerCase().includes(searchTerm.toLowerCase()));

  const addFaction = () => {
    const newFaction = {
      id: `faction_${Date.now()}`,
      Name: "New Faction",
      Role: "Neutral",
      Status: "Active",
      Leadership: "",
      KnownGoals: "",
      Timeline: ""
    };
    const updated = [...factions, newFaction];
    setState({ ...state, Factions: updated });
    setSelectedIdx(updated.length - 1);
  };

  const deleteFaction = async () => {
    if (selectedIdx === null) return;
    const ok = await confirm(`Delete ${factions[selectedIdx].Name}?`, { title: "Delete Faction", confirmLabel: "Delete", danger: true });
    if (!ok) return;
    const updated = factions.filter((_, i) => i !== selectedIdx);
    setState({ ...state, Factions: updated });
    setSelectedIdx(null);
  };

  const updateFaction = (field, value) => {
    if (selectedIdx === null) return;
    const updated = factions.map((f, i) =>
      i === selectedIdx ? { ...f, [field]: value } : f
    );
    setState({ ...state, Factions: updated });
  };

  return (
    <div style={styles.container}>

      {/* Header Info Block */}
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <Network size={20} style={{ marginTop: '2px', color: '#f59e0b' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Faction Roster</strong>
            <p style={{ margin: '4px 0 0', color: '#fef08a', fontSize: '13px', lineHeight: '1.4' }}>
              Track factions as operational entities. Set their narrative role, current status, leadership, and goals. Main POV factions are selectable as scene perspective in the Scene Creator.
            </p>
          </div>
        </div>
      </div>

      <div style={styles.layout}>

        {/* SIDEBAR */}
        <div style={styles.sidebar}>
          <div style={styles.searchContainer}>
            <Search size={14} color="#666" style={{ marginRight: '8px' }} />
            <input
              style={styles.searchInput}
              placeholder="Search factions..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
            <button onClick={addFaction} style={styles.addBtn} title="Add Faction">
              <Plus size={16} />
            </button>
          </div>

          <div style={styles.listContainer}>
            {filteredFactions.length === 0 && (
              <div style={styles.emptyState}>
                {searchTerm ? "No factions match." : "No factions yet. Add one above."}
              </div>
            )}
            {filteredFactions.map(f => (
              <div
                key={f.id || f._idx}
                onClick={() => setSelectedIdx(f._idx)}
                style={{
                  ...styles.factionCard,
                  borderColor: selectedIdx === f._idx ? '#f59e0b' : '#27272a',
                  background: selectedIdx === f._idx ? '#1c1a10' : '#18181b'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: getRoleColor(f.Role), flexShrink: 0 }} />
                  <div style={{ overflow: 'hidden' }}>
                    <div style={styles.factionName}>{f.Name || "Unnamed"}</div>
                    <div style={{ display: 'flex', gap: '6px', marginTop: '2px' }}>
                      <span style={{ fontSize: '10px', color: getRoleColor(f.Role), fontWeight: '700' }}>{f.Role}</span>
                      <span style={{ fontSize: '10px', color: '#3f3f46' }}>·</span>
                      <span style={{ fontSize: '10px', color: getStatusColor(f.Status) }}>{f.Status}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* EDITOR */}
        <div style={styles.editorPanel}>
          {!activeFaction ? (
            <div style={styles.emptyEditor}>Select or create a faction.</div>
          ) : (
            <div style={styles.editorContent}>

              {/* Name + Delete */}
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end', marginBottom: '20px' }}>
                <div style={{ flex: 1 }}>
                  <label style={styles.label}>FACTION NAME</label>
                  <input
                    value={activeFaction.Name}
                    onChange={e => updateFaction('Name', e.target.value)}
                    style={styles.input}
                    placeholder="e.g. The Praetorian Guard"
                  />
                </div>
                <button
                  onClick={deleteFaction}
                  style={styles.deleteBtn}
                >
                  <Trash2 size={14} /> Delete
                </button>
              </div>

              {/* Role + Status */}
              <div style={{ display: 'flex', gap: '16px', marginBottom: '20px' }}>
                <div style={{ flex: 1 }}>
                  <label style={styles.label}>NARRATIVE ROLE</label>
                  <select
                    value={activeFaction.Role}
                    onChange={e => updateFaction('Role', e.target.value)}
                    style={{ ...styles.input, color: getRoleColor(activeFaction.Role), fontWeight: '700' }}
                  >
                    {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>
                </div>
                <div style={{ flex: 1 }}>
                  <label style={styles.label}>CURRENT STATUS</label>
                  <select
                    value={activeFaction.Status}
                    onChange={e => updateFaction('Status', e.target.value)}
                    style={{ ...styles.input, color: getStatusColor(activeFaction.Status) }}
                  >
                    {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
              </div>

              {/* Timeline */}
              {availableTimelines.length > 0 && (
                <div style={{ marginBottom: '20px' }}>
                  <label style={{ ...styles.label, color: '#a855f7' }}>TIMELINE</label>
                  <TimelineDropdown
                    value={activeFaction.Timeline || ""}
                    onChange={val => updateFaction('Timeline', val)}
                    timelines={availableTimelines}
                  />
                </div>
              )}

              {/* Leadership */}
              <div style={{ marginBottom: '20px' }}>
                <label style={styles.label}>CURRENT LEADERSHIP</label>
                <input
                  value={activeFaction.Leadership}
                  onChange={e => updateFaction('Leadership', e.target.value)}
                  style={styles.input}
                  placeholder="Who is currently in charge?"
                />
              </div>

              {/* Known Goals */}
              <div>
                <label style={styles.label}>KNOWN GOALS</label>
                <textarea
                  value={activeFaction.KnownGoals}
                  onChange={e => updateFaction('KnownGoals', e.target.value)}
                  style={{ ...styles.input, height: '140px', resize: 'vertical', fontFamily: 'inherit', lineHeight: '1.6' }}
                  placeholder="What is this faction actively pursuing at this point in the story?"
                />
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: { padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' },
  infoBox: { background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.2)', padding: '16px', borderRadius: '8px', marginBottom: '20px' },
  layout: { display: 'flex', gap: '20px', flex: 1, minHeight: 0 },

  sidebar: { width: '260px', minWidth: '260px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #27272a', paddingRight: '15px' },
  searchContainer: { display: 'flex', alignItems: 'center', background: '#18181b', border: '1px solid #27272a', borderRadius: '6px', padding: '8px 10px', marginBottom: '10px', gap: '6px' },
  searchInput: { flex: 1, background: 'transparent', border: 'none', color: '#e4e4e7', outline: 'none', fontSize: '13px' },
  addBtn: { background: '#f59e0b', border: 'none', color: '#000', borderRadius: '4px', cursor: 'pointer', padding: '4px 8px', display: 'flex', alignItems: 'center', fontWeight: '700' },
  listContainer: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '4px' },
  factionCard: { padding: '10px 12px', borderRadius: '6px', cursor: 'pointer', border: '1px solid', transition: 'all 0.15s' },
  factionName: { fontSize: '13px', color: '#e4e4e7', fontWeight: '600', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  emptyState: { color: '#3f3f46', fontSize: '13px', fontStyle: 'italic', textAlign: 'center', marginTop: '30px' },

  editorPanel: { flex: 1, background: '#131315', borderRadius: '8px', border: '1px solid #27272a', overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  editorContent: { padding: '24px', overflowY: 'auto', flex: 1 },
  emptyEditor: { margin: 'auto', color: '#3f3f46', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' },

  label: { fontSize: '11px', color: '#71717a', fontWeight: '700', display: 'block', marginBottom: '6px', letterSpacing: '0.5px' },
  input: { width: '100%', padding: '10px 12px', background: '#09090b', border: '1px solid #27272a', color: '#e4e4e7', borderRadius: '6px', outline: 'none', fontSize: '14px', boxSizing: 'border-box' },
  deleteBtn: { padding: '10px 12px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#ef4444', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: '600', whiteSpace: 'nowrap' }
};