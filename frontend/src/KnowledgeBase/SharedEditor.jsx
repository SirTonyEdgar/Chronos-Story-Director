import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Search, Plus, Save, Trash2, Upload, Edit2, FileText, Loader2 } from 'lucide-react';

const API_URL = "http://localhost:8000";

/**
 * Shared Editor Component
 * =======================
 * A comprehensive CRUD interface for managing text-based knowledge fragments 
 * (Lore, Rules, Plans, Facts).
 * * Features:
 * - Real-time search and filtering.
 * - Client-side file import (.txt, .md, .json).
 * - Intuitive "Click-to-Edit" title interface.
 * - Auto-saving context per profile.
 *
 * @param {string} profile - Active project profile identifier.
 * @param {string} category - The specific knowledge domain (e.g., "Lore").
 * @param {string} color - Accent color for UI elements.
 */
export default function SharedEditor({ profile, category, icon, color, description, placeholder }) {
  // --- STATE MANAGEMENT ---
  const [items, setItems] = useState([]);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  
  // Editor Fields
  const [editTitle, setEditTitle] = useState("");
  const [editContent, setEditContent] = useState("");
  
  // UX State
  const [isSaving, setIsSaving] = useState(false);
  const [isFocused, setIsFocused] = useState(false); // Track title focus state

  // Refs
  const fileInputRef = useRef(null);

  // --- EFFECTS ---

  /**
   * Reload data when the profile or category changes.
   * Resets selection to ensure clean state.
   */
  useEffect(() => {
    if (profile) {
      fetchItems();
      setSelectedId(null);
    }
  }, [profile, category]);

  /**
   * Sync editor fields with the selected item.
   * Clears fields if no item is selected (New Entry mode).
   */
  useEffect(() => {
    if (selectedId) {
      const item = items.find(i => i.id === selectedId);
      if (item) {
        setEditTitle(item.name);
        setEditContent(item.content);
      }
    } else {
      setEditTitle("");
      setEditContent("");
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

  const handleSave = async () => {
    if (!editTitle.trim()) return alert("Please enter a title before saving.");
    
    setIsSaving(true);
    try {
      const payload = {
        name: editTitle,
        content: editContent || "...",
        category
      };

      if (selectedId) {
        // Update existing
        await axios.post(`${API_URL}/knowledge/update/${profile}`, { ...payload, id: selectedId });
      } else {
        // Create new
        await axios.post(`${API_URL}/knowledge/create/${profile}`, payload);
      }
      
      await fetchItems();
      alert("Entry saved successfully.");
    } catch (err) {
      alert("Save failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedId) return;
    if (!confirm("Are you sure you want to permanently delete this entry?")) return;
    
    try {
      await axios.post(`${API_URL}/knowledge/delete/${profile}`, { id: selectedId });
      setSelectedId(null);
      fetchItems();
    } catch (err) {
      alert("Delete failed: " + err.message);
    }
  };

  const handleCreateNew = () => {
    setSelectedId(null);
    setEditTitle("");
    setEditContent("");
  };

  // --- FILE IMPORT UTILITIES ---

  const triggerFileUpload = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Reset editor to "New Entry" mode populated with file data
    setSelectedId(null);
    setEditTitle(file.name); 

    const reader = new FileReader();

    // Text-based format handling
    if (file.type === "text/plain" || file.name.endsWith(".md") || file.name.endsWith(".txt") || file.name.endsWith(".json")) {
      reader.onload = (event) => {
        setEditContent(event.target.result);
      };
      reader.readAsText(file);
    } 
    // Fallback for PDFs or binaries (placeholder logic)
    else if (file.type === "application/pdf") {
      alert("Note: PDF text extraction requires backend processing. A placeholder has been created.");
      setEditContent(`[PDF Imported: ${file.name}]\n\n(Please copy/paste text content here manually until OCR is enabled.)`);
    } 
    else {
      alert("Unsupported file type. Please upload .txt, .md, or .json files.");
    }

    // Reset input to allow re-uploading the same file if needed
    e.target.value = null;
  };

  // Filter list based on search query
  const filteredItems = items.filter(i => 
    i.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={styles.container}>
      
      {/* Invisible File Input */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        style={{ display: 'none' }} 
        accept=".txt,.md,.json,.pdf"
      />

      {/* --- LEFT SIDEBAR (NAVIGATION) --- */}
      <div style={styles.sidebar}>
        
        {/* Sidebar Header */}
        <div style={styles.sidebarHeader}>
          
          {/* Context Badge */}
          <div style={{ 
            ...styles.infoBadge,
            color: color, 
            background: `${color}15`, 
            border: `1px solid ${color}30`
          }}>
            {description}
          </div>

          {/* Search Bar */}
          <div style={styles.searchContainer}>
            <Search size={14} color="#666" />
            <input 
              value={search} 
              onChange={e => setSearch(e.target.value)} 
              placeholder={`Search ${category}...`}
              style={styles.searchInput}
            />
          </div>

          {/* Primary Actions */}
          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={handleCreateNew} style={{ ...styles.createBtn, flex: 1 }}>
              <Plus size={14} /> New
            </button>
            <button onClick={triggerFileUpload} style={{ ...styles.createBtn, flex: 1 }} title="Import Text File">
              <Upload size={14} /> Import
            </button>
          </div>
        </div>

        {/* Item List */}
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
              <FileText size={13} style={{ marginRight: '8px', opacity: selectedId === item.id ? 1 : 0.5 }} />
              <span style={styles.itemText}>{item.name}</span>
            </div>
          ))}
          {filteredItems.length === 0 && (
            <div style={styles.emptyState}>No entries found.</div>
          )}
        </div>
      </div>

      {/* --- RIGHT PANEL (EDITOR) --- */}
      <div style={styles.editorPanel}>
        
        {/* Editor Toolbar */}
        <div style={styles.toolbar}>
          
          {/* Intuitive Title Input */}
          <div style={{ position: 'relative', flex: 1, marginRight: '20px' }}>
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
            {/* Visual Hint: Pencil Icon (Visible when empty or unfocused) */}
            {!isFocused && !editTitle && (
              <div style={{ position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', opacity: 0.5, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ fontSize: '11px', color: '#666' }}>Click to Edit Title</span>
                <Edit2 size={14} color="#666" />
              </div>
            )}
          </div>

          {/* Toolbar Actions */}
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

        {/* Main Content Area */}
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

/**
 * CSS-in-JS Styles
 * Optimized for dark mode and fluid layouts.
 */
const styles = {
  container: {
    display: 'flex', 
    height: '100%', 
    gap: '0', 
    border: '1px solid #333', 
    borderRadius: '8px', 
    overflow: 'hidden', 
    background: '#09090b', 
    width: '100%'
  },
  sidebar: {
    width: '320px', 
    minWidth: '320px', 
    display: 'flex', 
    flexDirection: 'column', 
    borderRight: '1px solid #333', 
    background: '#111'
  },
  sidebarHeader: {
    padding: '15px', 
    borderBottom: '1px solid #222', 
    display: 'flex', 
    flexDirection: 'column', 
    gap: '12px'
  },
  infoBadge: {
    fontSize: '12px', 
    padding: '10px', 
    borderRadius: '6px', 
    lineHeight: '1.5'
  },
  searchContainer: {
    display: 'flex', 
    alignItems: 'center', 
    background: '#1a1a1a', 
    border: '1px solid #333', 
    borderRadius: '6px', 
    padding: '0 10px'
  },
  searchInput: {
    flex: 1, 
    background: 'transparent', 
    border: 'none', 
    color: '#eee', 
    padding: '10px 8px', 
    fontSize: '13px', 
    outline: 'none'
  },
  createBtn: {
    padding: '8px', 
    background: '#222', 
    border: '1px dashed #444', 
    color: '#888', 
    borderRadius: '6px', 
    cursor: 'pointer', 
    fontSize: '12px',
    display: 'flex', 
    alignItems: 'center', 
    justifyContent: 'center', 
    gap: '6px',
    transition: 'all 0.2s ease'
  },
  listContainer: {
    flex: 1, 
    overflowY: 'auto', 
    padding: '10px', 
    display: 'flex', 
    flexDirection: 'column', 
    gap: '4px'
  },
  listItem: {
    padding: '10px 12px', 
    borderRadius: '6px', 
    cursor: 'pointer', 
    fontSize: '13px',
    display: 'flex', 
    alignItems: 'center',
    transition: 'background 0.1s ease'
  },
  itemText: {
    whiteSpace: 'nowrap', 
    overflow: 'hidden', 
    textOverflow: 'ellipsis',
    flex: 1
  },
  emptyState: {
    padding: '30px', 
    textAlign: 'center', 
    color: '#444', 
    fontSize: '12px',
    fontStyle: 'italic'
  },
  editorPanel: {
    flex: 1, 
    display: 'flex', 
    flexDirection: 'column', 
    background: '#0e0e0e'
  },
  toolbar: {
    height: '70px', 
    borderBottom: '1px solid #222', 
    display: 'flex', 
    justifyContent: 'space-between', 
    alignItems: 'center', 
    padding: '0 25px',
    background: '#111'
  },
  titleInput: {
    background: 'transparent', 
    border: 'none', 
    fontSize: '18px', 
    fontWeight: '700', 
    outline: 'none', 
    width: '100%',
    padding: '8px 0',
    transition: 'border-color 0.2s ease, color 0.2s ease'
  },
  actionBtn: {
    padding: '8px 16px', 
    background: '#18181b', 
    border: '1px solid #333', 
    color: '#aaa', 
    borderRadius: '6px', 
    cursor: 'pointer', 
    fontSize: '13px', 
    fontWeight: '600', 
    display: 'flex', 
    alignItems: 'center', 
    gap: '8px',
    minWidth: '80px',
    justifyContent: 'center'
  },
  textArea: {
    flex: 1, 
    width: '100%', 
    background: 'transparent', 
    color: '#e4e4e7', 
    border: 'none', 
    padding: '40px', 
    resize: 'none', 
    outline: 'none', 
    fontSize: '15px', 
    lineHeight: '1.8', 
    fontFamily: 'monospace',
    boxSizing: 'border-box'
  }
};