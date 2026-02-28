import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MessageCircle, RefreshCw, Undo2, Send, Database, Edit2, Trash2, Save, X, Clock, Search, ChevronDown, ChevronRight, AlertTriangle } from 'lucide-react';

const API_URL = "http://localhost:8000";

const REACTION_TEMPLATES = {
  "👤 Individual / Personal": [
    "Internal Monologue / Private Thoughts", "Personal Diary / Journal Entry",
    "Direct Speech / Live Reaction", "Private Letter / Correspondence"
  ],
  "🏛️ Political / Bureaucratic": [
    "Official Decree / Executive Order", "Senate / Council Debate",
    "Diplomatic Cable / Envoy Message", "Propaganda Broadcast"
  ],
  "⚔️ Military / Tactical": [
    "Combat Report / Sitrep", "Strategy Meeting / War Room",
    "Radio Chatter / Field Comms", "Officer's Log"
  ],
  "🕵️ Underground / Criminal": [
    "Thieves' Cant / Code Words", "Black Market Transaction Log",
    "Encrypted Channel / Dark Web Post", "Anonymous Tip"
  ],
  "📢 Public Discourse / Media": [
    "News Front Page / Headline", "Social Media Feed / Viral Post",
    "Town Crier / Public Announcement", "Commoner's Gossip"
  ],
  "✨ Custom / Specific": ["Manual Input"]
};

export default function ReactionTool({ profile }) {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState("");
  const [faction, setFaction] = useState("");
  const [deletingId, setDeletingId] = useState(null);
  
  const [editFaction, setEditFaction] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedId, setExpandedId] = useState(null);

  // Style State
  const [category, setCategory] = useState("👤 Individual / Personal");
  const [format, setFormat] = useState("Internal Monologue / Private Thoughts");
  const [customFormat, setCustomFormat] = useState("");
  const [instructions, setInstructions] = useState("");
  const [isPublic, setIsPublic] = useState(true);

  // Output & History State
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState("");
  const [activeTab, setActiveTab] = useState("output"); // 'output' or 'history'
  const [history, setHistory] = useState([]);
  
  // Inline Edit State
  const [editingId, setEditingId] = useState(null);
  const [editText, setEditText] = useState("");

  useEffect(() => {
    if (profile) {
      fetchFiles();
      fetchHistory();
    }
  }, [profile]);

  const fetchFiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/files/${profile}`);
      setFiles(res.data || []);
      if (res.data.length > 0) setSelectedFile(res.data[0]);
    } catch (err) { console.error("Error fetching files:", err); }
  };

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_URL}/reaction/history/${profile}`);
      setHistory(res.data || []);
    } catch (err) { console.error("Error fetching history:", err); }
  };

  const handleSimulate = async () => {
    if (!faction) return alert("Please specify a Target Faction.");
    setLoading(true);
    setOutput("");
    setActiveTab("output");

    const finalFormat = format === "Manual Input" ? customFormat : format;
    const stylePrompt = `${category} -> ${finalFormat}`;

    try {
      const res = await axios.post(`${API_URL}/reaction/generate/${profile}`, {
        scene_file: selectedFile,
        faction: faction,
        format_style: stylePrompt,
        public_only: isPublic,
        custom_instructions: instructions
      });
      setOutput(res.data.content);
      fetchHistory();
    } catch (err) {
      alert("Error: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const handleUndo = async () => {
    if(!confirm(`Undo last reaction for ${faction}? This removes it from the text file AND database.`)) return;
    try {
      const res = await axios.post(`${API_URL}/reaction/undo/${profile}`, {
        scene_file: selectedFile,
        faction: faction
      });
      alert(`Undo Complete: ${res.data.file_message}`);
      setOutput(""); 
      fetchHistory();
    } catch (err) {
      alert("Undo Failed: " + err.message);
    }
  };

  // --- CRUD Handlers ---
const handleDeleteClick = (id) => {
    setDeletingId(id);
  };

  const confirmDelete = async () => {
    if (!deletingId) return;
    try {
      await axios.delete(`${API_URL}/reaction/delete/${profile}/${deletingId}`);
      fetchHistory();
    } catch (err) { 
      alert("Delete Failed: " + err.message); 
    } finally {
      setDeletingId(null);
    }
  };

  const handleStartEdit = (item) => {
    setEditingId(item.id);
    setEditText(item.text);
    setEditFaction(item.faction);
  };

  const handleSaveEdit = async (id) => {
    try {
      await axios.put(`${API_URL}/reaction/edit/${profile}/${id}`, { 
        new_text: editText, 
        new_faction: editFaction
      });
      setEditingId(null);
      fetchHistory();
    } catch (err) { alert("Save Failed: " + err.message); }
  };

  const filteredHistory = history.filter(item => 
    item.faction.toLowerCase().includes(searchQuery.toLowerCase()) || 
    item.scene.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.text.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div style={{ display: 'flex', height: '100%', padding: '30px', boxSizing: 'border-box', flexDirection: 'column' }}>
      
      <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0, fontSize: '24px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <MessageCircle size={24} color="#a855f7" /> Faction Reaction Engine
        </h2>
      </div>

      <div style={{ flex: 1, display: 'flex', gap: '20px', overflow: 'hidden' }}>
        
        {/* --- LEFT PANEL: CONTROLS --- */}
        <div style={{ width: '400px', background: '#111', border: '1px solid #333', borderRadius: '8px', padding: '20px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' }}>
          
          <div style={groupStyle}>
            <label style={labelStyle}>Target Scene</label>
            <select value={selectedFile} onChange={e => setSelectedFile(e.target.value)} style={inputStyle}>
              {files.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>

          <div style={groupStyle}>
            <label style={labelStyle}>Target Faction / Character</label>
            <input 
              value={faction} 
              onChange={e => setFaction(e.target.value)} 
              placeholder="e.g. The City Guard" 
              style={inputStyle} 
            />
          </div>

          <div style={{height: '1px', background: '#333', margin: '5px 0'}} />

          <div style={groupStyle}>
            <label style={labelStyle}>Perspective / Era</label>
            <select value={category} onChange={e => { setCategory(e.target.value); setFormat(REACTION_TEMPLATES[e.target.value][0]); }} style={inputStyle}>
              {Object.keys(REACTION_TEMPLATES).map(cat => <option key={cat} value={cat}>{cat}</option>)}
            </select>
          </div>

          <div style={groupStyle}>
            <label style={labelStyle}>Format / Medium</label>
            <select value={format} onChange={e => setFormat(e.target.value)} style={inputStyle}>
              {REACTION_TEMPLATES[category].map(fmt => <option key={fmt} value={fmt}>{fmt}</option>)}
            </select>
          </div>

          {format === "Manual Input" && (
            <input 
              value={customFormat} onChange={e => setCustomFormat(e.target.value)} 
              placeholder="Describe custom format..." 
              style={{...inputStyle, border: '1px dashed #a855f7'}} 
            />
          )}

          <div style={groupStyle}>
            <label style={labelStyle}>Additional Instructions</label>
            <textarea 
              value={instructions} 
              onChange={e => setInstructions(e.target.value)}
              placeholder="e.g. They should sound terrified..."
              style={{...inputStyle, height: '80px', resize: 'none'}}
            />
          </div>

          {/* Custom Toggle Switch */}
          <label style={{display: 'flex', alignItems: 'center', gap: '10px', fontSize: '13px', color: isPublic ? '#fff' : '#888', cursor: 'pointer', transition: 'color 0.2s', marginTop: '5px'}}>
            
            <input 
              type="checkbox" 
              checked={isPublic} 
              onChange={e => setIsPublic(e.target.checked)} 
              style={{ display: 'none' }} 
            />
            
            {/* The Pill/Switch */}
            <div style={{ width: '36px', height: '20px', background: isPublic ? '#a855f7' : '#333', borderRadius: '10px', position: 'relative', transition: 'background 0.2s' }}>
              <div style={{ width: '14px', height: '14px', background: '#fff', borderRadius: '50%', position: 'absolute', top: '3px', left: isPublic ? '19px' : '3px', transition: 'left 0.2s' }} />
            </div>
            
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: isPublic ? 'bold' : 'normal' }}>
              👁️ Public Knowledge Only (Ignore Secrets)
            </span>
            
          </label>

          <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
            <button 
              onClick={handleSimulate} 
              disabled={loading}
              style={{ flex: 1, padding: '12px', background: '#a855f7', color: 'white', border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}
            >
              {loading ? <RefreshCw className="spin" size={18}/> : <Send size={18}/>}
              Simulate
            </button>
            
            {/* UNDO BUTTON - Only renders if there is an active output on the screen */}
            {output && (
              <button 
                onClick={handleUndo} 
                title="Undo Last Memory & Text"
                style={{ padding: '12px 20px', background: '#333', color: '#ef4444', border: '1px solid #ef4444', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                <Undo2 size={18}/>
              </button>
            )}
          </div>
        </div>

        {/* --- RIGHT PANEL: OUTPUT & HISTORY --- */}
        <div style={{ flex: 1, background: '#0e0e0e', border: '1px solid #333', borderRadius: '8px', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          
          {/* Header Tabs */}
          <div style={{ display: 'flex', borderBottom: '1px solid #333', background: '#151515' }}>
            <button 
              onClick={() => setActiveTab("output")}
              style={{ flex: 1, padding: '15px', background: activeTab === 'output' ? '#0e0e0e' : 'transparent', border: 'none', borderBottom: activeTab === 'output' ? '2px solid #a855f7' : '2px solid transparent', color: activeTab === 'output' ? '#fff' : '#888', fontWeight: 'bold', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
            >
              <MessageCircle size={18} /> Current Output
            </button>
            <button 
              onClick={() => setActiveTab("history")}
              style={{ flex: 1, padding: '15px', background: activeTab === 'history' ? '#0e0e0e' : 'transparent', border: 'none', borderBottom: activeTab === 'history' ? '2px solid #a855f7' : '2px solid transparent', color: activeTab === 'history' ? '#fff' : '#888', fontWeight: 'bold', cursor: 'pointer', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '8px' }}
            >
              <Database size={18} /> Database History
            </button>
          </div>

          {/* Content Area */}
          <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
            
            {/* TAB: OUTPUT */}
            {activeTab === 'output' && (
              output ? (
                <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', fontSize: '14px', color: '#e4e4e7' }}>
                  {output}
                </div>
              ) : (
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#333' }}>
                  <MessageCircle size={64} opacity={0.2} />
                  <p>Reaction Output will appear here</p>
                </div>
              )
            )}

            {/* TAB: HISTORY */}
            {activeTab === 'history' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
                
                {/* Modern Search Bar */}
                <div style={{ position: 'relative', marginBottom: '10px' }}>
                  <Search size={16} color="#888" style={{ position: 'absolute', left: '12px', top: '12px' }} />
                  <input 
                    type="text" 
                    placeholder="Search past reactions by faction, scene, or keyword..." 
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    style={{ ...inputStyle, width: '100%', paddingLeft: '38px', boxSizing: 'border-box' }}
                  />
                </div>

                {filteredHistory.length === 0 ? (
                  <p style={{ color: '#666', textAlign: 'center', marginTop: '40px' }}>No memories found.</p>
                ) : (
                  filteredHistory.map(item => {
                    const isExpanded = expandedId === item.id || editingId === item.id;

                    return (
                      <div key={item.id} style={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: '6px', overflow: 'hidden' }}>
                        
                        {/* Clickable Accordion Header */}
                        <div 
                          onClick={() => setExpandedId(isExpanded ? null : item.id)}
                          style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', padding: '15px', cursor: 'pointer', background: isExpanded ? '#222' : '#1a1a1a', borderBottom: isExpanded ? '1px solid #333' : 'none', transition: 'background 0.2s' }}
                        >
                          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', width: '100%' }}>
                            <div style={{ marginTop: '2px' }}>
                              {isExpanded ? <ChevronDown size={18} color="#888" /> : <ChevronRight size={18} color="#888" />}
                            </div>
                            <div style={{ width: '100%', paddingRight: '15px' }}>
                              
                              {/* Editable Title */}
                              {editingId === item.id ? (
                                <input 
                                  value={editFaction}
                                  onChange={e => setEditFaction(e.target.value)}
                                  onClick={e => e.stopPropagation()} 
                                  style={{ ...inputStyle, fontSize: '15px', color: '#a855f7', padding: '6px', width: '100%', fontWeight: 'bold' }}
                                />
                              ) : (
                                <strong style={{ color: '#a855f7', fontSize: '15px' }}>{item.faction}</strong>
                              )}

                              <div style={{ display: 'flex', alignItems: 'center', gap: '5px', color: '#888', fontSize: '12px', marginTop: '4px' }}>
                                <Clock size={12} /> {item.timestamp} | Scene: {item.scene}
                              </div>
                            </div>
                          </div>
                          
                          {/* Action Buttons (stopPropagation prevents accordion toggle when clicking buttons) */}
                          <div style={{ display: 'flex', gap: '10px' }} onClick={e => e.stopPropagation()}>
                            {editingId === item.id ? (
                              <>
                                <button onClick={() => handleSaveEdit(item.id)} style={actionBtnStyle('#10b981')} title="Save"><Save size={16} /></button>
                                <button onClick={() => setEditingId(null)} style={actionBtnStyle('#888')} title="Cancel"><X size={16} /></button>
                              </>
                            ) : (
                              <>
                                <button onClick={() => { handleStartEdit(item); setExpandedId(item.id); }} style={actionBtnStyle('#3b82f6')} title="Edit Text"><Edit2 size={16} /></button>
                                <button onClick={() => handleDeleteClick(item.id)} style={actionBtnStyle('#ef4444')} title="Delete Memory"><Trash2 size={16} /></button>
                              </>
                            )}
                          </div>
                        </div>

                        {/* Expandable Body */}
                        {isExpanded && (
                          <div style={{ padding: '15px', animation: 'fadeIn 0.2s ease-in-out' }}>
                            {editingId === item.id ? (
                              <textarea 
                                value={editText}
                                onChange={e => setEditText(e.target.value)}
                                rows={Math.max(5, editText.split('\n').length)}
                                style={{ ...inputStyle, width: '100%', resize: 'vertical', boxSizing: 'border-box', lineHeight: '1.6' }}
                              />
                            ) : (
                              <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6', fontSize: '13px', color: '#ccc' }}>
                                {item.text}
                              </div>
                            )}
                          </div>
                        )}

                      </div>
                    );
                  })
                )}
              </div>
            )}

          </div>
        </div>
      </div>
      {/* --- DELETE WARNING MODAL --- */}
      {deletingId && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.75)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', animation: 'fadeIn 0.2s ease-out' }}>
          <div style={{ background: '#111', border: '1px solid #ef4444', borderRadius: '8px', padding: '24px', width: '450px', maxWidth: '90%', boxShadow: '0 10px 30px rgba(0,0,0,0.8)' }}>
            
            <h3 style={{ margin: '0 0 15px 0', color: '#ef4444', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '18px' }}>
              <AlertTriangle size={22} /> Delete Database Memory
            </h3>
            
            <p style={{ fontSize: '14px', color: '#ccc', lineHeight: '1.6', marginBottom: '15px' }}>
              Are you sure you want to delete this memory from the <strong>Faction Database</strong>? 
            </p>
            
            <div style={{ background: '#2a0a0a', borderLeft: '3px solid #ef4444', padding: '12px', marginBottom: '24px', borderRadius: '0 4px 4px 0' }}>
              <p style={{ margin: 0, fontSize: '13px', color: '#e4e4e7', lineHeight: '1.5' }}>
                <strong style={{ color: '#ef4444' }}>Note:</strong> This does <em>NOT</em> delete the text from your actual Scene file. You must edit the file manually if you want it removed from the narrative.
              </p>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button 
                onClick={() => setDeletingId(null)} 
                style={{ padding: '8px 16px', background: 'transparent', border: '1px solid #444', color: '#ccc', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
              >
                Cancel
              </button>
              <button 
                onClick={confirmDelete} 
                style={{ padding: '8px 16px', background: '#ef4444', border: 'none', color: '#fff', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
              >
                Delete Memory
              </button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

// --- Helper Styles ---
const groupStyle = { display: 'flex', flexDirection: 'column', gap: '6px' };
const labelStyle = { fontSize: '12px', fontWeight: 'bold', color: '#888' };
const inputStyle = { background: '#1a1a1a', border: '1px solid #333', color: '#fff', padding: '10px', borderRadius: '4px', outline: 'none', fontSize: '13px' };
const actionBtnStyle = (color) => ({
  background: 'transparent', border: `1px solid ${color}`, color: color,
  borderRadius: '4px', padding: '6px', cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center'
});