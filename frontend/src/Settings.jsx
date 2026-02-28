import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Settings as SettingsIcon, Cpu, Globe, Save, Plus, Trash2, 
  Clock, Zap, Check, Calendar, Watch 
} from 'lucide-react';

const API_URL = "http://localhost:8000";

// --- CUSTOM COMPONENTS ---

/**
 * ModelSelect
 * Dropdown for selecting AI models. Defined here to avoid ReferenceErrors.
 */
const ModelSelect = ({ label, value, options, onChange, desc }) => (
  <div>
    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
      <label style={styles.label}>{label}</label>
      <span style={{ fontSize: '11px', color: '#555' }}>{desc}</span>
    </div>
    <select 
      value={value || (options[0] || "")} 
      onChange={e => onChange(e.target.value)}
      style={styles.select}
    >
      {options.map(m => <option key={m} value={m}>{m}</option>)}
    </select>
  </div>
);

/**
 * StyledCheckbox
 * Custom UI replacement for the native checkbox.
 */
const StyledCheckbox = ({ checked, onChange, style }) => (
  <div 
    onClick={() => onChange(!checked)}
    style={{
      width: '18px',
      height: '18px',
      borderRadius: '4px',
      border: checked ? '1px solid #3b82f6' : '1px solid #52525b', 
      background: checked ? '#3b82f6' : '#27272a', 
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer',
      transition: 'all 0.2s',
      flexShrink: 0,
      ...style
    }}
  >
    {checked && <Check size={12} color="#fff" strokeWidth={4} />}
  </div>
);

const AutoResizeTextarea = ({ value, onChange, placeholder, style }) => {
  const textareaRef = useRef(null);
  const adjustHeight = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto'; 
      el.style.height = `${el.scrollHeight}px`; 
    }
  };
  useEffect(() => { adjustHeight(); }, [value]);
  return (
    <textarea
      ref={textareaRef}
      value={value}
      onChange={(e) => { onChange(e); adjustHeight(); }}
      placeholder={placeholder}
      rows={1}
      style={{
        width: '100%', background: 'transparent', border: 'none', color: '#fff',
        outline: 'none', fontFamily: 'inherit', resize: 'none', overflow: 'hidden',
        minHeight: '38px', lineHeight: '1.5', display: 'block', padding: '8px',
        fontSize: '13px', ...style
      }}
    />
  );
};

export default function Settings({ profile }) {
  const [config, setConfig] = useState({
    use_time_system: 'true',
    enable_year: 'true',
    enable_date: 'true',
    enable_clock: 'true',
    enable_chapters: 'true',
    use_timelines: 'false'
  });
  
  const [timelines, setTimelines] = useState([]);
  const [worldState, setWorldState] = useState(null);
  const [availableModels, setAvailableModels] = useState([]); 
  const [loading, setLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  useEffect(() => { if (profile) fetchData(); }, [profile]);

  const fetchData = async () => {
    try {
      const [settingsRes, stateRes, modelsRes] = await Promise.all([
        axios.get(`${API_URL}/settings/${profile}`),
        axios.get(`${API_URL}/state/${profile}`),
        axios.get(`${API_URL}/settings/models`) 
      ]);
      setConfig(prev => ({ ...prev, ...settingsRes.data }));
      setWorldState(stateRes.data);
      setTimelines(stateRes.data.Timelines || []);
      setAvailableModels(modelsRes.data || ["gemini-1.5-flash"]); 
      setLoading(false);
      setHasUnsavedChanges(false);
    } catch (err) {
      console.error("Critical Error loading settings:", err);
      alert("Failed to load settings configuration.");
    }
  };

  const handleConfigChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setHasUnsavedChanges(true);
  };

  const handleTimelineChange = (index, field, value) => {
    const newTimelines = [...timelines];
    newTimelines[index] = { ...newTimelines[index], [field]: value };
    setTimelines(newTimelines);
    setHasUnsavedChanges(true);
  };

  const addTimeline = () => {
    setTimelines([...timelines, { Name: "New Timeline", Description: "" }]);
    setHasUnsavedChanges(true);
  };

  const removeTimeline = (index) => {
    if (!confirm("Delete this timeline?")) return;
    const newTimelines = [...timelines];
    newTimelines.splice(index, 1);
    setTimelines(newTimelines);
    setHasUnsavedChanges(true);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const updates = [
        ['model_scene', config.model_scene],
        ['model_chat', config.model_chat],
        ['model_reaction', config.model_reaction],
        ['model_analysis', config.model_analysis],
        ['model_retrieval', config.model_retrieval],
        ['default_timezone', config.default_timezone],
        ['enable_chapters', config.enable_chapters],
        ['use_time_system', config.use_time_system],
        ['enable_year', config.enable_year],
        ['enable_date', config.enable_date],
        ['enable_clock', config.enable_clock],
        ['use_timelines', config.use_timelines]
      ];

      for (let [key, val] of updates) {
        if (val !== undefined) await axios.post(`${API_URL}/settings/update/${profile}?key=${key}&value=${val}`);
      }

      if (worldState) {
        const newState = { ...worldState, Timelines: timelines };
        await axios.post(`${API_URL}/state/save/${profile}`, newState);
      }
      setHasUnsavedChanges(false);
    } catch (err) {
      alert("Save operation failed.");
    } finally { setIsSaving(false); }
  };

  if (loading) return <div style={styles.loadingState}>Loading System Configuration...</div>;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2 style={styles.title}><SettingsIcon size={28} color="#94a3b8" /> System Configuration</h2>
        <p style={styles.subtitle}>Configure AI routing behaviors, narrative mechanics, and simulation rules.</p>
      </div>

      <div style={styles.grid}>
        
        {/* --- CARD 1: AI MODELS --- */}
        <div style={styles.card}>
          <div style={styles.cardHeader}><Cpu size={18} color="#a855f7" /> <span>AI Model Routing</span></div>
          <div style={styles.cardContent}>
            <ModelSelect label="Scene Writer" value={config.model_scene} options={availableModels} onChange={v => handleConfigChange('model_scene', v)} desc="Generates story prose." />
            <ModelSelect label="Co-Author" value={config.model_chat} options={availableModels} onChange={v => handleConfigChange('model_chat', v)} desc="Handles chat & brainstorming." />
            <ModelSelect label="Reaction Engine" value={config.model_reaction} options={availableModels} onChange={v => handleConfigChange('model_reaction', v)} desc="Simulates faction responses." />
            <ModelSelect label="Logic & Strategy" value={config.model_analysis} options={availableModels} onChange={v => handleConfigChange('model_analysis', v)} desc="War Room & State Analysis." />
            <ModelSelect label="Librarian / Retrieval" value={config.model_retrieval} options={availableModels} onChange={v => handleConfigChange('model_retrieval', v)} desc="Vector Search (Flash recommended)." />
          </div>
        </div>

        {/* --- CARD 2: WORLD MECHANICS --- */}
        <div style={styles.card}>
          <div style={styles.cardHeader}><Globe size={18} color="#3b82f6" /> <span>World Mechanics</span></div>
          <div style={styles.cardContent}>
            
            {/* CONDITIONAL TIMEZONE: Only shows if Master Time AND Clock Time are enabled */}
            {config.use_time_system === 'true' && config.enable_clock === 'true' && (
              <div>
                <label style={styles.label}>Default Timezone</label>
                <div style={styles.timeInputWrapper}>
                  <Clock size={16} color="#666" />
                  <input value={config.default_timezone || "UTC"} onChange={e => handleConfigChange('default_timezone', e.target.value)} style={styles.input} />
                </div>
              </div>
            )}

            {/* MASTER TIME TOGGLE */}
            <div style={styles.toggleContainer}>
              <StyledCheckbox 
                checked={config.use_time_system === 'true'} 
                onChange={val => handleConfigChange('use_time_system', String(val))}
              />
              <div>
                <div style={styles.toggleTitle}>Enable Time System (Master)</div>
                <div style={styles.toggleDesc}>Master switch for all temporal tracking features.</div>
              </div>
            </div>

            {/* GRANULAR TIME CONTROLS */}
            {config.use_time_system === 'true' && (
              <div style={styles.subSettingsContainer}>
                <div style={styles.subToggle}>
                  <StyledCheckbox 
                    checked={config.enable_year === 'true'} 
                    onChange={val => handleConfigChange('enable_year', String(val))}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Calendar size={14} color="#aaa" /> <span style={styles.subLabel}>Track Year (e.g. 2024)</span>
                  </div>
                </div>
                <div style={styles.subToggle}>
                  <StyledCheckbox 
                    checked={config.enable_date === 'true'} 
                    onChange={val => handleConfigChange('enable_date', String(val))}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Calendar size={14} color="#aaa" /> <span style={styles.subLabel}>Track Calendar Date (e.g. March 6)</span>
                  </div>
                </div>
                <div style={styles.subToggle}>
                  <StyledCheckbox 
                    checked={config.enable_clock === 'true'} 
                    onChange={val => handleConfigChange('enable_clock', String(val))}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Watch size={14} color="#aaa" /> <span style={styles.subLabel}>Track Clock Time (e.g. 14:00)</span>
                  </div>
                </div>
              </div>
            )}

            <div style={styles.toggleContainer}>
              <StyledCheckbox 
                checked={config.enable_chapters === 'true'} 
                onChange={val => handleConfigChange('enable_chapters', String(val))}
              />
              <div>
                <div style={styles.toggleTitle}>Enable Chapter System</div>
                <div style={styles.toggleDesc}>Organize files with "Ch01_" prefixes and sequential numbering.</div>
              </div>
            </div>

            <div style={styles.toggleContainer}>
              <StyledCheckbox 
                checked={config.use_timelines === 'true'} 
                onChange={val => handleConfigChange('use_timelines', String(val))}
              />
              <div>
                <div style={styles.toggleTitle}>Enable Multiverse / Timelines</div>
                <div style={styles.toggleDesc}>Allow parallel realities and diverging history tracks.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {config.use_timelines === 'true' && (
        <div style={{ ...styles.card, marginTop: '30px', marginBottom: '80px' }}>
          <div style={{...styles.cardHeader, borderBottom: '1px solid #333', paddingBottom: '15px', marginBottom: '15px'}}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><Zap size={18} color="#eab308" /> <span>Multiverse Configuration</span></div>
            <button onClick={addTimeline} style={styles.addBtn}><Plus size={14} /> Add Timeline</button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {timelines.map((tl, i) => (
              <div key={i} style={styles.timelineItem}>
                <div style={styles.timelineHeader}>
                  <div style={{ flex: 1 }}>
                    <label style={styles.fieldLabel}>TIMELINE NAME</label>
                    <input value={tl.Name} onChange={e => handleTimelineChange(i, "Name", e.target.value)} style={styles.inputBordered} placeholder="e.g. Timeline Alpha" />
                  </div>
                  <button onClick={() => removeTimeline(i)} style={styles.delBtn}><Trash2 size={16} /></button>
                </div>
                <div style={{ width: '100%' }}>
                  <label style={styles.fieldLabel}>CONTEXT / DIVERGENCE RULES</label>
                  <div style={styles.textareaWrapper}>
                    <AutoResizeTextarea value={tl.Description} onChange={e => handleTimelineChange(i, "Description", e.target.value)} placeholder="Describe how this timeline differs..." />
                  </div>
                </div>
              </div>
            ))}
            {timelines.length === 0 && <div style={styles.emptyState}>No active timelines defined.</div>}
          </div>
        </div>
      )}

      <button onClick={handleSave} disabled={isSaving || !hasUnsavedChanges} style={{...styles.fab, background: hasUnsavedChanges ? '#22c55e' : '#27272a', color: hasUnsavedChanges ? '#000' : '#666', cursor: (isSaving || !hasUnsavedChanges) ? 'default' : 'pointer', width: hasUnsavedChanges ? '180px' : '150px', border: hasUnsavedChanges ? '1px solid #16a34a' : '1px solid #333' }}>
        {isSaving ? "Saving..." : <>{hasUnsavedChanges ? <Save size={18} /> : <Check size={18} />}<span>{hasUnsavedChanges ? "Save Config" : "Config Saved"}</span></>}
      </button>
    </div>
  );
}

// --- CSS STYLES ---
const styles = {
  container: { padding: '30px', height: '100%', boxSizing: 'border-box', overflowY: 'auto', width: '100%', position: 'relative' },
  loadingState: { padding: '30px', color: '#666' },
  header: { marginBottom: '30px' },
  title: { margin: 0, display: 'flex', alignItems: 'center', gap: '12px', fontSize: '24px', color: '#e4e4e7' },
  subtitle: { margin: '5px 0 0 0', color: '#64748b', fontSize: '14px', marginLeft: '40px' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' },
  card: { background: '#09090b', border: '1px solid #27272a', borderRadius: '8px', padding: '20px' },
  cardHeader: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', fontSize: '16px', fontWeight: 'bold', color: '#e4e4e7', marginBottom: '20px' },
  cardContent: { display: 'flex', flexDirection: 'column', gap: '20px' },
  label: { fontSize: '13px', fontWeight: '600', color: '#cbd5e1' },
  select: { width: '100%', padding: '10px', background: '#111', border: '1px solid #333', color: '#fff', borderRadius: '6px', fontSize: '13px', outline: 'none' },
  input: { background: 'transparent', border: 'none', color: '#fff', fontSize: '13px', outline: 'none', fontFamily: 'inherit', width: '100%' },
  timeInputWrapper: { display: 'flex', alignItems: 'center', gap: '10px', background: '#111', padding: '10px', borderRadius: '6px', border: '1px solid #333' },
  toggleContainer: { display: 'flex', gap: '12px', alignItems: 'center', background: '#18181b', padding: '12px', borderRadius: '6px', border: '1px solid #27272a' },
  toggleTitle: { fontWeight: 'bold', fontSize: '14px', color: '#eee' },
  toggleDesc: { fontSize: '12px', color: '#888' },
  subSettingsContainer: { marginLeft: '20px', padding: '10px 10px 10px 20px', borderLeft: '2px solid #333', display: 'flex', flexDirection: 'column', gap: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '0 6px 6px 0' },
  subToggle: { display: 'flex', alignItems: 'center', gap: '12px' },
  subLabel: { fontSize: '13px', color: '#aaa', fontWeight: '500' },
  timelineItem: { display: 'flex', flexDirection: 'column', gap: '15px', background: '#111', padding: '15px', borderRadius: '8px', border: '1px solid #222' },
  timelineHeader: { display: 'flex', gap: '20px', alignItems: 'flex-end' },
  fieldLabel: { fontSize: '11px', color: '#666', marginBottom: '6px', display: 'block', fontWeight: 'bold', letterSpacing: '0.5px' },
  inputBordered: { background: '#09090b', border: '1px solid #333', color: '#fff', fontSize: '13px', outline: 'none', width: '100%', padding: '10px', borderRadius: '4px', boxSizing: 'border-box' },
  textareaWrapper: { background: '#09090b', border: '1px solid #333', borderRadius: '4px', padding: '2px' },
  addBtn: { background: '#222', border: '1px solid #444', color: '#eee', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', transition: 'all 0.2s' },
  delBtn: { background: 'transparent', border: '1px solid #ef4444', color: '#ef4444', cursor: 'pointer', padding: '8px', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'background 0.2s', height: '36px', width: '36px' },
  emptyState: { color: '#666', fontStyle: 'italic', padding: '10px', textAlign: 'center' },
  fab: { position: 'fixed', bottom: '30px', right: '40px', height: '48px', borderRadius: '24px', fontSize: '13px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', zIndex: 1000, boxShadow: '0 4px 12px rgba(0,0,0,0.3)' }
};