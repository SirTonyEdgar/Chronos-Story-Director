import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Search, Plus, Save, Trash2, Upload, Edit2, FileText, Loader2, Zap, ChevronDown } from 'lucide-react';
import { API_URL } from '../config';
import { toast, confirm } from '../components/Notifications';

/**
 * Custom Timeline Dropdown
 */
const TimelineDropdown = ({ value, onChange, timelines }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) setIsOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const displayValue = value || "Universal (All Timelines)";

  return (
    <div ref={containerRef} style={{ position: 'relative', flex: 1, marginRight: '20px' }}>
      
      {/* Clickable Header Box */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{ 
          display: 'flex', alignItems: 'center', gap: '8px', 
          background: 'rgba(168, 85, 247, 0.1)', border: '1px solid #a855f7', 
          padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', 
          color: '#e4e4e7', fontSize: '13px', transition: 'all 0.2s'
        }}
      >
        <Zap size={14} color="#a855f7" />
        <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {displayValue}
        </span>
        <ChevronDown size={14} color="#a855f7" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
      </div>

      {/* Floating Menu */}
      {isOpen && (
        <div style={{ 
          position: 'absolute', top: '100%', left: 0, right: 0, 
          background: '#18181b', border: '1px solid #a855f7', borderRadius: '6px', 
          marginTop: '4px', zIndex: 100, overflow: 'hidden', 
          boxShadow: '0 10px 25px rgba(0,0,0,0.8)' 
        }}>
          <div
            onClick={() => { onChange(""); setIsOpen(false); }}
            style={{ 
              padding: '10px 12px', cursor: 'pointer', fontSize: '13px', 
              color: value === "" ? '#a855f7' : '#a1a1aa', 
              background: value === "" ? 'rgba(168, 85, 247, 0.1)' : 'transparent',
              fontWeight: value === "" ? 'bold' : 'normal'
            }}
          >
            Universal (All Timelines)
          </div>
          {timelines.map(tl => (
            <div
              key={tl.Name}
              onClick={() => { onChange(tl.Name); setIsOpen(false); }}
              style={{ 
                padding: '10px 12px', cursor: 'pointer', fontSize: '13px', 
                color: value === tl.Name ? '#a855f7' : '#a1a1aa', 
                background: value === tl.Name ? 'rgba(168, 85, 247, 0.1)' : 'transparent', 
                borderTop: '1px solid #27272a',
                fontWeight: value === tl.Name ? 'bold' : 'normal'
              }}
            >
              {tl.Name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Shared Editor Component
 * =======================
 * A comprehensive CRUD interface for managing text-based knowledge fragments 
 * (Lore, Rules, Plans, Facts).
 */
export default function SharedEditor({ profile, category, icon, color, description, placeholder }) {
  // --- STATE MANAGEMENT ---
  const [items, setItems] = useState([]);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  
  // Multiverse State
  const [availableTimelines, setAvailableTimelines] = useState([]);
  
  // Editor Fields
  const [editTitle, setEditTitle] = useState("");
  const [editContent, setEditContent] = useState("");
  const [editTimeline, setEditTimeline] = useState("");
  
  // UX State
  const [isSaving, setIsSaving] = useState(false);
  const [isFocused, setIsFocused] = useState(false);

  // Refs
  const fileInputRef = useRef(null);

  // --- EFFECTS ---

  useEffect(() => {
    if (profile) {
      fetchItems();
      fetchTimelines();
      setSelectedId(null);
    }
  }, [profile, category]);

  useEffect(() => {
    if (selectedId) {
      const item = items.find(i => i.id === selectedId);
      if (item) {
        setEditTitle(item.name);
        setEditContent(item.content);
        setEditTimeline(item.timeline || "");
      }
    } else {
      setEditTitle("");
      setEditContent("");
      setEditTimeline("");
    }
  }, [selectedId, items]);

  // --- API OPERATIONS ---

  const fetchItems = async () => {
    try {
      const res = await axios.get(`${API_URL}/knowledge/list/${profile}/${category}`);
      setItems(res.data || []);
    } catch (err) { 
      console.error(`Error loading ${category}:`, err); 
    }
  };

  const fetchTimelines = async () => {
    try {
      const res = await axios.get(`${API_URL}/state/${profile}`);
      setAvailableTimelines(res.data.Timelines || []);
    } catch (err) { console.error("Error fetching timelines:", err); }
  };

  const handleSave = async () => {
    if (!editTitle.trim()) return toast("Please enter a title before saving.", "warning");
    
    setIsSaving(true);
    try {
      const payload = {
        name: editTitle,
        content: editContent || "...",
        category: category,
        timeline: editTimeline
      };

      if (selectedId) {
        await axios.post(`${API_URL}/knowledge/update/${profile}`, { ...payload, id: selectedId });
      } else {
        await axios.post(`${API_URL}/knowledge/create/${profile}`, payload);
      }
      
      await fetchItems();
      toast("Entry saved successfully.", "success");
    } catch (err) {
      toast("Save failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedId) return;
    const ok = await confirm("Are you sure?", { title: "Delete Entry", confirmLabel: "Delete", danger: true });
    if (!ok) return;
    
    try {
      await axios.post(`${API_URL}/knowledge/delete/${profile}`, { id: selectedId });
      setSelectedId(null);
      fetchItems();
    } catch (err) {
      toast("Delete failed: " + err.message, "error");
    }
  };

  const handleCreateNew = () => {
    setSelectedId(null);
    setEditTitle("");
    setEditContent("");
    setEditTimeline("");
  };

  // --- FILE IMPORT UTILITIES ---

  const triggerFileUpload = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSelectedId(null);
    setEditTitle(file.name); 

    const reader = new FileReader();

    if (file.type === "text/plain" || file.name.endsWith(".md") || file.name.endsWith(".txt") || file.name.endsWith(".json")) {
      reader.onload = (event) => {
        setEditContent(event.target.result);
      };
      reader.readAsText(file);
    } 
    else if (file.type === "application/pdf") {
      toast("PDF imported — paste text content manually until OCR is enabled.", "warning");
      setEditContent(`[PDF Imported: ${file.name}]\n\n(Please copy/paste text content here manually until OCR is enabled.)`);
    } 
    else {
      toast("Unsupported file type. Use .txt, .md, or .json.", "warning");
    }

    e.target.value = null;
  };

  const filteredItems = items.filter(i => 
    i.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={styles.container}>
      
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        style={{ display: 'none' }} 
        accept=".txt,.md,.json,.pdf"
      />

      {/* --- LEFT SIDEBAR (NAVIGATION) --- */}
      <div style={styles.sidebar}>
        
        <div style={styles.sidebarHeader}>
          <div style={{ ...styles.infoBadge, color: color, background: `${color}15`, border: `1px solid ${color}30` }}>
            {description}
          </div>

          <div style={styles.searchContainer}>
            <Search size={14} color="#666" />
            <input 
              value={search} 
              onChange={e => setSearch(e.target.value)} 
              placeholder={`Search ${category}...`}
              style={styles.searchInput}
            />
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={handleCreateNew} style={{ ...styles.createBtn, flex: 1 }}>
              <Plus size={14} /> New
            </button>
            <button onClick={triggerFileUpload} style={{ ...styles.createBtn, flex: 1 }} title="Import Text File">
              <Upload size={14} /> Import
            </button>
          </div>
        </div>

        <div style={styles.listContainer}>
          {filteredItems.map(item => (
            <div 
              key={item.id} 
              onClick={() => setSelectedId(item.id)}
              style={{
                ...styles.listItem,
                background: selectedId === item.id ? '#27272a' : 'transparent',
                color: selectedId === item.id ? '#fff' : '#888',
                borderLeft: selectedId === item.id ? `3px solid ${color}` : '3px solid transparent',
                fontWeight: selectedId === item.id ? '600' : '400'
              }}
            >
              <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <FileText size={13} style={{ marginRight: '8px', opacity: selectedId === item.id ? 1 : 0.5 }} />
                  <span style={styles.itemText}>{item.name}</span>
                </div>
                {item.timeline && (
                  <span style={{ fontSize: '10px', color: '#a855f7', marginLeft: '21px', marginTop: '2px' }}>
                    [{item.timeline}]
                  </span>
                )}
              </div>
            </div>
          ))}
          {filteredItems.length === 0 && (
            <div style={styles.emptyState}>No entries found.</div>
          )}
        </div>
      </div>

      {/* --- RIGHT PANEL (EDITOR) --- */}
      <div style={styles.editorPanel}>
        
        <div style={styles.toolbar}>
          
          <div style={{ position: 'relative', flex: 2, marginRight: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <input 
              value={editTitle} 
              onChange={e => setEditTitle(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Untitled Entry"
              style={{
                ...styles.titleInput,
                borderBottom: isFocused ? `2px solid ${color}` : '1px dashed #444',
                color: editTitle ? '#fff' : '#71717a'
              }}
            />
            {!isFocused && !editTitle && (
              <div style={{ position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', opacity: 0.5, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '11px', color: '#666' }}>Click to Edit Title</span>
                <Edit2 size={14} color="#666" />
              </div>
            )}
          </div>

          {/* NEW: Custom Sleek Timeline Dropdown */}
          {availableTimelines.length > 0 && (
            <TimelineDropdown 
              value={editTimeline} 
              onChange={setEditTimeline} 
              timelines={availableTimelines} 
            />
          )}

          <div style={{ display: 'flex', gap: '10px' }}>
            {selectedId && (
              <button 
                onClick={handleDelete} 
                style={{ ...styles.actionBtn, color: '#ef4444', borderColor: '#ef444440' }}
                title="Delete Entry"
              >
                <Trash2 size={16} />
              </button>
            )}
            <button 
              onClick={handleSave} 
              disabled={isSaving}
              style={{ ...styles.actionBtn, background: color, borderColor: color, color: '#fff' }}
            >
              {isSaving ? <Loader2 className="spin" size={16} /> : <Save size={16} />}
              <span style={{ marginLeft: '6px' }}>{isSaving ? "Saving..." : "Save"}</span>
            </button>
          </div>
        </div>

        <textarea 
          value={editContent} 
          onChange={e => setEditContent(e.target.value)}
          placeholder={placeholder || "Start writing or import a file..."}
          style={styles.textArea}
        />
      </div>
    </div>
  );
}

const styles = {
  container: { display: 'flex', height: '100%', gap: '0', border: '1px solid #333', borderRadius: '8px', overflow: 'hidden', background: '#09090b', width: '100%' },
  sidebar: { width: '320px', minWidth: '320px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #333', background: '#111' },
  sidebarHeader: { padding: '15px', borderBottom: '1px solid #222', display: 'flex', flexDirection: 'column', gap: '12px' },
  infoBadge: { fontSize: '12px', padding: '10px', borderRadius: '6px', lineHeight: '1.5' },
  searchContainer: { display: 'flex', alignItems: 'center', background: '#1a1a1a', border: '1px solid #333', borderRadius: '6px', padding: '0 10px' },
  searchInput: { flex: 1, background: 'transparent', border: 'none', color: '#eee', padding: '10px 8px', fontSize: '13px', outline: 'none' },
  createBtn: { padding: '8px', background: '#222', border: '1px dashed #444', color: '#888', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px', transition: 'all 0.2s ease' },
  listContainer: { flex: 1, overflowY: 'auto', padding: '10px', display: 'flex', flexDirection: 'column', gap: '4px' },
  listItem: { padding: '10px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', transition: 'background 0.1s ease' },
  itemText: { whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 },
  emptyState: { padding: '30px', textAlign: 'center', color: '#444', fontSize: '12px', fontStyle: 'italic' },
  editorPanel: { flex: 1, display: 'flex', flexDirection: 'column', background: '#0e0e0e' },
  toolbar: { height: '70px', borderBottom: '1px solid #222', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 25px', background: '#111' },
  titleInput: { background: 'transparent', border: 'none', fontSize: '18px', fontWeight: '700', outline: 'none', width: '100%', padding: '8px 0', transition: 'border-color 0.2s ease, color 0.2s ease' },
  actionBtn: { padding: '8px 16px', background: '#18181b', border: '1px solid #333', color: '#aaa', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px', minWidth: '80px', justifyContent: 'center' },
  textArea: { flex: 1, width: '100%', background: 'transparent', color: '#e4e4e7', border: 'none', padding: '40px', resize: 'none', outline: 'none', fontSize: '15px', lineHeight: '1.8', fontFamily: 'monospace', boxSizing: 'border-box' }
};