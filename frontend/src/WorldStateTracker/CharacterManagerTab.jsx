import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  UserCircle, Users, Plus, Trash2, Star, Shield, 
  Swords, Search, Save, Link as LinkIcon, Network, X, Clock, Edit2, Check
} from 'lucide-react';

const API_URL = "http://localhost:8000";

// --- CONFIGURATION ---

const ROLES = [
  { id: "POV", label: "Main (POV)", color: "#3b82f6", icon: <Star size={14} /> },
  { id: "Support", label: "Support", color: "#22c55e", icon: <Shield size={14} /> },
  { id: "Antagonist", label: "Antagonist", color: "#ef4444", icon: <Swords size={14} /> },
  { id: "Minor", label: "Minor", color: "#71717a", icon: <Users size={14} /> }
];

const AVATAR_OPTIONS = [
  "Male", "Female", "Neutral", "Villain", "Leader/Noble", 
  "Official/Diplomat", "Wizard", "Tech/Cyborg", "Soldier", 
  "Knight", "Child", "Student", "Cat", "Family", "Organization/Corp"
];

const getAvatarPath = (iconKey) => {
  const map = {
    "Male": "male.png", "Female": "female.png", "Neutral": "neutral.png",
    "Villain": "villain.png", "Leader/Noble": "crown.png", "Official/Diplomat": "diplomat.png",
    "Wizard": "wizard.png", "Tech/Cyborg": "cyborg.png", "Soldier": "soldier.png",
    "Knight": "knight.png", "Child": "child.png", "Student": "student.png", "Cat": "cat.png",
    "Family": "family.png", "Organization/Corp": "briefcase.png"
  };
  const filename = map[iconKey] || "neutral.png";
  return `/icons/${filename}`;
};

export default function CharacterManagerTab({ state, setState, profile }) {
  const [selectedId, setSelectedId] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [settings, setSettings] = useState({ protagonist: "" });
  const [isSavingSetting, setIsSavingSetting] = useState(false);

  // Link form state
  const [newLinkTarget, setNewLinkTarget] = useState("");
  const [newLinkType, setNewLinkType] = useState("");
  
  // Link Editing State
  const [editingLinkIndex, setEditingLinkIndex] = useState(null);
  const [editLinkTarget, setEditLinkTarget] = useState("");
  const [editLinkType, setEditLinkType] = useState("");

  useEffect(() => {
    if (state && !state.Cast) {
      normalizeDataStructure();
    } else if (state && state.Cast && state.Cast.length > 0 && !selectedId) {
      const firstPov = state.Cast.find(c => c.Role === "POV");
      setSelectedId(firstPov ? firstPov.id : state.Cast[0].id);
    }
    if (profile) fetchSettings();
  }, [state, profile]);

  const normalizeDataStructure = () => {
    const oldProtagonist = state["Protagonist Status"] || { Name: "New Hero" };
    const oldAllies = state["Allies"] || [];
    const newCast = [];

    newCast.push({
      ...oldProtagonist,
      id: `char_${Date.now()}`,
      Role: "POV",
      Tags: ["Protagonist"],
      Links: [] 
    });

    oldAllies.forEach((ally, index) => {
      newCast.push({
        ...ally,
        id: `char_${Date.now() + index + 1}`,
        Role: "Support",
        Tags: [ally.Relation || "Ally"],
        Links: []
      });
    });

    setState({ ...state, Cast: newCast });
    if (newCast.length > 0) setSelectedId(newCast[0].id);
  };

  const fetchSettings = async () => {
    try {
      const activeProfile = profile || localStorage.getItem("lastProfile");
      const res = await axios.get(`${API_URL}/settings/${activeProfile}`);
      setSettings(res.data);
    } catch (err) { console.error(err); }
  };

  // --- CRUD OPERATIONS ---

  const getCast = () => state.Cast || [];
  const activeChar = getCast().find(c => c.id === selectedId) || {};

  const updateCharacter = (field, value) => {
    const updates = typeof field === 'object' ? field : { [field]: value };

    const updatedCast = getCast().map(c => {
      if (c.id === selectedId) {
        return { ...c, ...updates };
      }
      return c;
    });
    
    const newState = { ...state, Cast: updatedCast };

    // Legacy Sync
    const newActiveChar = updatedCast.find(c => c.id === selectedId);
    if (newActiveChar && newActiveChar.Role === "POV") {
      newState["Protagonist Status"] = { 
        ...newState["Protagonist Status"], 
        ...newActiveChar
      };
    }

    setState(newState);
  };

  const updateGlobalDate = (field, value) => {
    setState({ ...state, [field]: value });
  };

  // --- LINKING LOGIC ---

  const addLink = () => {
    if (!newLinkTarget || !newLinkType.trim()) return;
    const currentLinks = activeChar.Links || [];
    
    // Prevent duplicates to same target
    const filteredLinks = currentLinks.filter(l => l.targetId !== newLinkTarget);
    
    const newLink = { targetId: newLinkTarget, type: newLinkType };
    updateCharacter("Links", [...filteredLinks, newLink]);
    setNewLinkTarget("");
    setNewLinkType("");
  };

  const removeLink = (targetId) => {
    if (!window.confirm("Remove this connection?")) return;
    const currentLinks = activeChar.Links || [];
    updateCharacter("Links", currentLinks.filter(l => l.targetId !== targetId));
  };

  const startEditingLink = (index, link) => {
    setEditingLinkIndex(index);
    setEditLinkTarget(link.targetId);
    setEditLinkType(link.type);
  };

  const saveEditedLink = (index) => {
    if (!editLinkTarget || !editLinkType.trim()) return;
    
    const currentLinks = [...(activeChar.Links || [])];
    
    // Update the specific link at index
    currentLinks[index] = { targetId: editLinkTarget, type: editLinkType };
    
    updateCharacter("Links", currentLinks);
    setEditingLinkIndex(null);
  };

  const cancelEditLink = () => {
    setEditingLinkIndex(null);
  };

  const addCharacter = () => {
    const newChar = {
      id: `char_${Date.now()}`,
      Name: "New Character",
      Role: "Support",
      Icon: "Neutral",
      Age: "Unknown",
      Tags: [],
      Links: []
    };
    const newState = { ...state, Cast: [...getCast(), newChar] };
    setState(newState);
    setSelectedId(newChar.id);
  };

  const deleteCharacter = (id) => {
    if (!window.confirm("Permanently delete this character?")) return;
    const newState = { ...state, Cast: getCast().filter(c => c.id !== id) };
    setState(newState);
    if (selectedId === id) setSelectedId(null);
  };

  const handleDOBChange = (value) => {
    updateCharacter("DOB", value);
  };

  const handleSettingBlur = async () => {
    const activeProfile = profile || localStorage.getItem("lastProfile");
    if (!activeProfile) return;
    setIsSavingSetting(true);
    try {
      await axios.post(`${API_URL}/settings/update/${activeProfile}?key=protagonist&value=${settings.protagonist}`);
    } catch (err) { console.error(err); } 
    finally { setTimeout(() => setIsSavingSetting(false), 600); }
  };

  const filteredCast = getCast().filter(c => 
    (c.Name || "").toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div style={styles.container}>
      
      {/* Header Info Block */}
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <Users size={20} style={{ marginTop: '2px', color: '#60a5fa' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Cast Roster</strong>
            <p style={{ margin: '4px 0 0', color: '#93c5fd', fontSize: '13px', lineHeight: '1.4' }}>
              Manage the ensemble cast. Define specific ages or use vague terms (e.g., "Ancient"). 
              Link characters to visualize the social web.
            </p>
          </div>
        </div>
      </div>

      <div style={styles.layout}>
        
        {/* Sidebar */}
        <div style={styles.sidebar}>
          <div style={styles.searchContainer}>
            <Search size={14} color="#666" style={{ marginRight: '8px' }} />
            <input 
              style={styles.searchInput} 
              placeholder="Search cast..." 
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
            <button onClick={addCharacter} style={styles.addBtn} title="Add Character">
              <Plus size={16} />
            </button>
          </div>

          <div style={styles.listContainer}>
            {filteredCast.map(char => {
              const roleStyle = ROLES.find(r => r.id === char.Role) || ROLES[3];
              return (
                <div 
                  key={char.id} 
                  onClick={() => setSelectedId(char.id)}
                  style={{
                    ...styles.charCard,
                    borderColor: selectedId === char.id ? '#3b82f6' : '#27272a',
                    background: selectedId === char.id ? '#1e293b' : '#18181b'
                  }}
                >
                  <img src={getAvatarPath(char.Icon)} alt="avatar" style={styles.listIcon} />
                  <div style={{ overflow: 'hidden' }}>
                    <div style={styles.charName}>{char.Name}</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '11px', color: roleStyle.color }}>
                      {roleStyle.icon} {roleStyle.label}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Editor */}
        <div style={styles.editor}>
          {activeChar.id ? (
            <>
              {/* Header */}
              <div style={styles.editorHeader}>
                <div style={{ flex: 1 }}>
                  <label style={styles.label}>NARRATIVE NAME</label>
                  <input 
                    style={{ ...styles.input, fontSize: '18px', fontWeight: 'bold' }} 
                    value={activeChar.Name || ""} 
                    onChange={e => updateCharacter("Name", e.target.value)}
                  />
                </div>
                <div style={{ width: '150px' }}>
                  <label style={styles.label}>NARRATIVE ROLE</label>
                  <select 
                    style={styles.input} 
                    value={activeChar.Role || "Support"} 
                    onChange={e => updateCharacter("Role", e.target.value)}
                  >
                    {ROLES.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
                  </select>
                </div>
                <button onClick={() => deleteCharacter(activeChar.id)} style={styles.deleteBtn}>
                  <Trash2 size={18} />
                </button>
              </div>

              <div style={styles.grid}>
                {/* --- LEFT COLUMN: IDENTITY & SOCIAL --- */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                  <div>
                    <label style={styles.label}>ALIASES / TITLES</label>
                    <textarea 
                      style={{...styles.input, height: '60px', resize: 'none'}} 
                      value={activeChar.Aliases || ""} 
                      onChange={e => updateCharacter("Aliases", e.target.value)}
                      placeholder="e.g. Lord Commander"
                    />
                  </div>
                  <div>
                    <label style={styles.label}>CURRENT GOAL</label>
                    <textarea 
                      style={{...styles.input, height: '80px', resize: 'none'}} 
                      value={activeChar.Goal || ""} 
                      onChange={e => updateCharacter("Goal", e.target.value)}
                      placeholder="Primary objective..."
                    />
                  </div>
                  <div>
                    <label style={styles.label}>MAP AVATAR</label>
                    <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
                      <select 
                        style={{...styles.input, flex: 1}} 
                        value={activeChar.Icon || "Male"} 
                        onChange={(e) => updateCharacter("Icon", e.target.value)}
                      >
                        {AVATAR_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                      </select>
                      <div style={styles.avatarPreview}>
                        <img src={getAvatarPath(activeChar.Icon)} alt="avatar" style={{width: '80%'}} />
                      </div>
                    </div>
                  </div>

                  {/* SOCIAL WEB CONTAINER (SCROLLABLE) */}
                  <div style={styles.socialBox}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px'}}>
                      <Network size={16} color="#a855f7" />
                      <strong style={{color: '#fff', fontSize: '13px'}}>Social Web</strong>
                    </div>

                    <div style={{ marginBottom: '15px' }}>
                      <label style={styles.label}>PRIMARY ORBIT (CLUSTERING)</label>
                      <select 
                        style={styles.input} 
                        value={activeChar.Orbit || ""} 
                        onChange={e => updateCharacter("Orbit", e.target.value)}
                      >
                        <option value="">(None / Independent)</option>
                        {getCast().filter(c => c.id !== activeChar.id).map(c => (
                          <option key={c.id} value={c.id}>{c.Name}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label style={styles.label}>DIRECT CONNECTIONS</label>
                      
                      {/* ADD NEW LINK ROW */}
                      <div style={{ display: 'flex', gap: '5px', marginBottom: '8px', paddingBottom: '8px', borderBottom: '1px solid #27272a' }}>
                        <select 
                          style={{...styles.input, flex: 1, fontSize: '12px'}}
                          value={newLinkTarget}
                          onChange={e => setNewLinkTarget(e.target.value)}
                        >
                          <option value="">Select Character...</option>
                          {getCast().filter(c => c.id !== activeChar.id).map(c => (
                            <option key={c.id} value={c.id}>{c.Name}</option>
                          ))}
                        </select>
                        <input 
                          type="text"
                          style={{...styles.input, width: '100px', fontSize: '12px'}}
                          value={newLinkType}
                          onChange={e => setNewLinkType(e.target.value)}
                          placeholder="Label..."
                        />
                        <button onClick={addLink} style={styles.addLinkBtn}>
                          <Plus size={14}/>
                        </button>
                      </div>

                      {/* SCROLLABLE LINK LIST */}
                      <div style={styles.scrollableList}>
                        {(activeChar.Links || []).map((link, i) => {
                          const target = getCast().find(c => c.id === link.targetId);
                          const isEditing = editingLinkIndex === i;

                          return (
                            <div key={i} style={styles.linkItem}>
                              <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                                <LinkIcon size={12} color="#60a5fa" style={{flexShrink: 0}} />
                                
                                {isEditing ? (
                                  // --- EDIT MODE ---
                                  <>
                                    <select 
                                      style={{...styles.input, flex: 1, fontSize: '11px', margin: '0 4px', padding: '2px 4px', height: '24px'}}
                                      value={editLinkTarget}
                                      onChange={e => setEditLinkTarget(e.target.value)}
                                    >
                                      {getCast().filter(c => c.id !== activeChar.id).map(c => (
                                        <option key={c.id} value={c.id}>{c.Name}</option>
                                      ))}
                                    </select>
                                    <input 
                                      style={{...styles.input, width: '80px', fontSize: '11px', padding: '2px 4px', height: '24px'}}
                                      value={editLinkType}
                                      onChange={e => setEditLinkType(e.target.value)}
                                    />
                                    <div style={{display: 'flex', gap: '4px', marginLeft: '4px'}}>
                                      <ActionBtn onClick={() => saveEditedLink(i)} color="#22c55e" icon={<Check size={12} />} />
                                      <ActionBtn onClick={cancelEditLink} color="#ef4444" icon={<X size={12} />} />
                                    </div>
                                  </>
                                ) : (
                                  // --- VIEW MODE ---
                                  <>
                                    <span style={{flex: 1, marginLeft: '8px', fontSize: '13px', color: '#fff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                                      {target ? target.Name : "Unknown"} 
                                    </span>
                                    <span style={{
                                      fontSize: '11px', color: '#a1a1aa', background: '#222', 
                                      padding: '2px 6px', borderRadius: '4px', marginRight: '8px'
                                    }}>
                                      {link.type}
                                    </span>
                                    <div style={{display: 'flex', gap: '4px'}}>
                                      <ActionBtn onClick={() => startEditingLink(i, link)} color="#60a5fa" icon={<Edit2 size={12} />} />
                                      <ActionBtn onClick={() => removeLink(link.targetId)} color="#ef4444" icon={<Trash2 size={12} />} />
                                    </div>
                                  </>
                                )}
                              </div>
                            </div>
                          );
                        })}
                        {(!activeChar.Links || activeChar.Links.length === 0) && 
                          <div style={{fontSize: '11px', color: '#555', fontStyle: 'italic', padding: '10px', textAlign: 'center'}}>
                            No direct connections yet.
                          </div>
                        }
                      </div>
                    </div>
                  </div>
                </div>

                {/* --- RIGHT COLUMN: TIME, TAGS, CONTEXT --- */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '25px' }}>
                  
                  {/* GLOBAL WORLD CLOCK */}
                  <div style={styles.timeBox}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '15px'}}>
                      <Clock size={16} color="#60a5fa" />
                      <strong style={{color: '#fff', fontSize: '13px'}}>World Clock (Global)</strong>
                    </div>
                    <div style={{ display: 'flex', gap: '15px' }}>
                      <div style={{ flex: 1 }}>
                        <label style={styles.label}>CURRENT ERA / YEAR</label>
                        <input 
                          type="text" 
                          style={{...styles.input, fontWeight: 'bold', color: '#60a5fa'}} 
                          value={state.Current_Year || ""} 
                          onChange={(e) => updateGlobalDate("Current_Year", e.target.value)}
                          placeholder="e.g. 2024"
                        />
                      </div>
                      <div style={{ flex: 1 }}>
                        <label style={styles.label}>CURRENT DATE / SEASON</label>
                        <input 
                          type="text" 
                          style={styles.input} 
                          value={state.Current_Date || ""} 
                          onChange={(e) => updateGlobalDate("Current_Date", e.target.value)} 
                          placeholder="e.g. March 6"
                        />
                      </div>
                    </div>
                  </div>

                  {/* FULL DATE OF BIRTH */}
                  <div>
                    <label style={styles.label}>FULL DATE OF BIRTH (OPTIONAL)</label>
                    <input 
                      type="text" 
                      style={styles.input} 
                      value={activeChar.DOB || ""} 
                      onChange={(e) => handleDOBChange(e.target.value)}
                      placeholder="e.g. March 6 (Enter Year below)"
                    />
                  </div>

                  {/* Character Age & Origin */}
                  <div style={{ display: 'flex', gap: '15px' }}>
                    <div style={{ flex: 1 }}>
                      <label style={styles.label}>BIRTH YEAR / ORIGIN</label>
                      <input 
                        type="text" 
                        style={styles.input} 
                        value={activeChar.Birth_Year || ""} 
                        onChange={(e) => {
                          const val = e.target.value;
                          const updates = { Birth_Year: val };
                          const currentYear = parseInt(state.Current_Year);
                          const bYear = parseInt(val);
                          if (!isNaN(currentYear) && !isNaN(bYear)) {
                            updates.Age = (currentYear - bYear).toString();
                          }
                          updateCharacter(updates);
                        }} 
                        placeholder="e.g. 1990"
                      />
                    </div>
                    <div style={{ flex: 1 }}>
                      <label style={styles.label}>AGE / DURATION</label>
                      <input 
                        type="text" 
                        style={{...styles.input, borderColor: '#3f3f46', background: '#09090b'}} 
                        value={activeChar.Age || ""} 
                        onChange={(e) => updateCharacter("Age", e.target.value)}
                        placeholder="Unknown"
                      />
                    </div>
                  </div>

                  <div>
                    <label style={styles.label}>KEYWORDS / GROUP TAGS</label>
                    <input 
                      style={styles.input} 
                      value={Array.isArray(activeChar.Tags) ? activeChar.Tags.join(", ") : activeChar.Tags || ""} 
                      onChange={e => updateCharacter("Tags", e.target.value.split(",").map(s=>s.trim()))} 
                      placeholder="e.g. Stark, Night's Watch"
                    />
                  </div>

                  {/* Context (POV Only) */}
                  {activeChar.Role === 'POV' && (
                    <div style={styles.contextBox}>
                      <label style={{...styles.label, color: '#fbbf24'}}>TRUE IDENTITY (AI CONTEXT)</label>
                      <input 
                        style={{ ...styles.input, borderColor: '#fbbf24' }} 
                        value={settings.protagonist || ""} 
                        onChange={(e) => setSettings({...settings, protagonist: e.target.value})}
                        onBlur={handleSettingBlur} 
                        placeholder="Core Identity for retrieval..."
                      />
                      <p style={{ fontSize: '10px', color: '#888', marginTop: '4px' }}>
                        * This sets the 'Protagonist' reference for the AI.
                      </p>
                    </div>
                  )}

                </div>
              </div>
            </>
          ) : (
            <div style={styles.emptyState}>Select or Create a Character</div>
          )}
        </div>
      </div>
    </div>
  );
}

const ActionBtn = ({ onClick, color, icon }) => (
  <div onClick={onClick} style={{
    cursor: 'pointer', padding: '4px', display: 'flex', alignItems: 'center', 
    background: 'rgba(0,0,0,0.3)', borderRadius: '4px', color: color
  }}>
    {icon}
  </div>
);

// --- STYLES ---
const styles = {
  container: { padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' },
  infoBox: { background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)', color: '#93c5fd', padding: '16px', borderRadius: '8px', fontSize: '13px', marginBottom: '20px' },
  layout: { display: 'flex', gap: '20px', flex: 1, minHeight: 0 },
  sidebar: { 
    width: '260px', 
    display: 'flex', 
    flexDirection: 'column', 
    borderRight: '1px solid #27272a', 
    paddingRight: '15px',
    overflow: 'hidden'
  },
  searchContainer: { display: 'flex', alignItems: 'center', background: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', padding: '8px', marginBottom: '15px' },
  searchInput: { background: 'transparent', border: 'none', color: '#fff', outline: 'none', fontSize: '13px', flex: 1 },
  addBtn: { background: '#2563eb', border: 'none', borderRadius: '4px', color: '#fff', cursor: 'pointer', padding: '4px', display: 'flex' },
  listContainer: { overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' },
  charCard: { display: 'flex', alignItems: 'center', gap: '10px', padding: '10px', borderRadius: '6px', border: '1px solid', cursor: 'pointer', transition: 'all 0.2s' },
  listIcon: { width: '32px', height: '32px', objectFit: 'contain' },
  charName: { fontSize: '13px', fontWeight: 'bold', color: '#e4e4e7', marginBottom: '2px' },
  editor: { flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'auto' },
  editorHeader: { display: 'flex', gap: '20px', alignItems: 'flex-end', marginBottom: '25px', borderBottom: '1px solid #27272a', paddingBottom: '20px' },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' },
  label: { fontSize: '11px', color: '#71717a', fontWeight: '700', display: 'block', marginBottom: '6px', letterSpacing: '0.5px' },
  input: { width: '100%', padding: '10px', background: '#18181b', border: '1px solid #333', color: '#fff', borderRadius: '6px', outline: 'none', fontSize: '14px', boxSizing: 'border-box', transition: 'border-color 0.2s' },
  deleteBtn: { background: '#450a0a', border: '1px solid #7f1d1d', color: '#ef4444', borderRadius: '6px', padding: '10px', cursor: 'pointer', height: '42px' },
  avatarPreview: { width: '42px', height: '42px', background: '#09090b', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid #333' },
  contextBox: { marginTop: '10px', padding: '15px', background: '#1e1b4b', border: '1px solid #4338ca', borderRadius: '8px' },
  emptyState: { display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#52525b', fontSize: '16px', fontWeight: '500' },
  
  // Social Web Styles
  socialBox: { background: '#131315', padding: '15px', borderRadius: '8px', border: '1px solid #27272a' },
  addLinkBtn: { 
    background: '#2563eb', border: 'none', borderRadius: '4px', color: '#fff', 
    padding: '0 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', 
    justifyContent: 'center' 
  },
  // Added for scrolling:
  scrollableList: { 
    maxHeight: '400px', 
    overflowY: 'auto', 
    display: 'flex', 
    flexDirection: 'column', 
    gap: '6px', 
    paddingRight: '4px' 
  },
  linkItem: { display: 'flex', alignItems: 'center', background: '#09090b', padding: '6px 8px', borderRadius: '4px', border: '1px solid #222' },
  
  // Time Box
  timeBox: { background: '#131315', padding: '15px', borderRadius: '8px', border: '1px solid #27272a' }
};