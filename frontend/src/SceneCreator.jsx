import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Play, Save, Trash2, RefreshCw, 
  PenTool, BookOpen, Edit, FileMinus, 
  Clock, ChevronDown, Check, X, Merge, Plus
} from 'lucide-react';
import { API_URL } from './config';
import { toast, confirm } from './components/Notifications';

/**
 * MultiSelect Component
 * A custom dropdown for selecting multiple context files.
 */
const MultiSelect = ({ options, selected, onChange, placeholder }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  // Handle outside clicks to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggleOption = (option) => {
    if (selected.includes(option)) {
      onChange(selected.filter(item => item !== option));
    } else {
      onChange([...selected, option]);
    }
  };

  const removeTag = (e, option) => {
    e.stopPropagation();
    onChange(selected.filter(item => item !== option));
  };

  return (
    <div style={{ position: 'relative', width: '100%', marginBottom: '15px' }} ref={containerRef}>
      <label style={styles.label}>Transition From (Context)</label>
      
      {/* Trigger Box */}
      <div 
        onClick={() => setIsOpen(!isOpen)}
        style={{
          ...styles.input,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          cursor: 'pointer',
          minHeight: '42px',
          height: 'auto',
          flexWrap: 'wrap',
          gap: '6px',
          padding: '6px 12px'
        }}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', flex: 1 }}>
          {selected.length === 0 && <span style={{ color: '#52525b', fontSize: '13px' }}>{placeholder}</span>}
          {selected.map(item => (
            <span key={item} style={styles.tag}>
              {item}
              <X 
                size={12} 
                style={{ marginLeft: '6px', cursor: 'pointer', opacity: 0.7 }} 
                onClick={(e) => removeTag(e, item)}
              />
            </span>
          ))}
        </div>
        <ChevronDown size={16} color="#71717a" />
      </div>

      {/* Dropdown Menu */}
      {isOpen && (
        <div style={styles.dropdownMenu}>
          {options.map(option => (
            <div 
              key={option} 
              onClick={() => toggleOption(option)}
              style={{
                ...styles.dropdownItem,
                background: selected.includes(option) ? '#27272a' : 'transparent',
                color: selected.includes(option) ? '#fff' : '#a1a1aa'
              }}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {option}
              </span>
              {selected.includes(option) && <Check size={14} color="#ef4444" />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/**
 * Scene Creator Module
 * ====================
 * The primary narrative engine interface.
 * Features state persistence, AI generation, and granular context control.
 */
export default function SceneCreator({ profile }) { 
  // --- NAVIGATION STATE ---
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem("chronos_scene_tab") || "write";
  });

  useEffect(() => {
    localStorage.setItem("chronos_scene_tab", activeTab);
  }, [activeTab]);

  // --- DATA STATE ---
  const [files, setFiles] = useState([]);
  
  // Configuration State
  const [useTimeSystem, setUseTimeSystem] = useState(true);
  const [useChapters, setUseChapters] = useState(true);
  
  // Granular Time Configs
  const [showYear, setShowYear] = useState(true);
  const [showDate, setShowDate] = useState(true);
  const [showClock, setShowClock] = useState(true);

  // Generation Inputs
  const [chapter, setChapter] = useState(1);
  const [part, setPart] = useState(1);
  const [title, setTitle] = useState("");
  const [brief, setBrief] = useState("");
  const [timeline, setTimeline] = useState("");
  const [availableTimelines, setAvailableTimelines] = useState([]);
  
  // Context Selection
  const [selectedContext, setSelectedContext] = useState([]);

  // Chronology Inputs
  const [year, setYear] = useState("");
  const [dateStr, setDateStr] = useState("");
  const [timeStr, setTimeStr] = useState("");
  
  // Toggles
  const [isGenerating, setIsGenerating] = useState(false);
  const [fogOfWar, setFogOfWar] = useState(false);

  // Editor State
  const [selectedFile, setSelectedFile] = useState("");
  const [fileContent, setFileContent] = useState("");

  // Chapter Partitioning & Management
  const [useParts, setUseParts] = useState(false);
  const [selectedManageFiles, setSelectedManageFiles] = useState([]);

  // --- INITIALIZATION ---

  useEffect(() => {
    if (profile) loadProfileData();
  }, [profile]);

  const sortFiles = (fileList) => {
    const chapterRegex = /^(?:Ch|Chapter)[_ ]?(\d+)(?:_Part_(\d+))?/i;
    
    const chapters = [];
    const others = [];

    fileList.forEach(file => {
      const match = file.match(chapterRegex);
      if (match) {
        chapters.push({
          name: file,
          num: parseInt(match[1], 10),
          part: match[2] ? parseInt(match[2], 10) : 0 
        });
      } else {
        others.push(file);
      }
    });

    chapters.sort((a, b) => {
      if (a.num !== b.num) return a.num - b.num;
      if (a.part !== b.part) return a.part - b.part;
      return a.name.localeCompare(b.name);
    });

    return [...chapters.map(c => c.name), ...others];
  };

  const loadProfileData = async () => {
    try {
      const [filesRes, chapterRes, settingsRes, stateRes] = await Promise.all([
        axios.get(`${API_URL}/files/${profile}`),
        axios.get(`${API_URL}/next_chapter/${profile}`),
        axios.get(`${API_URL}/settings/${profile}`),
        axios.get(`${API_URL}/state/${profile}`)
      ]);

      setFiles(sortFiles(filesRes.data));
      setChapter(chapterRes.data.next_chapter);

      const s = settingsRes.data;
      setUseTimeSystem(String(s.use_time_system).toLowerCase() === 'true');
      setUseChapters(String(s.enable_chapters || 'true').toLowerCase() === 'true');
      
      setShowYear(String(s.enable_year || 'true').toLowerCase() === 'true');
      setShowDate(String(s.enable_date || 'true').toLowerCase() === 'true');
      setShowClock(String(s.enable_clock || 'true').toLowerCase() === 'true');
      setAvailableTimelines(stateRes.data.Timelines || []);

    } catch (err) {
      console.error("Initialization Failed:", err);
    }
  };

  const refreshFileList = async () => {
    try {
      const res = await axios.get(`${API_URL}/files/${profile}`);
      setFiles(sortFiles(res.data));
    } catch (err) { console.error("Refresh failed:", err); }
  };

  // --- ACTIONS ---

  const handleGenerate = async () => {
    if (!brief) return toast("Please provide a Scene Brief.", "warning");
    
    setIsGenerating(true);
    try {
      // Fixed API endpoint and payload mapping
      const res = await axios.post(`${API_URL}/scene/generate/${profile}`, {
        chapter: parseInt(chapter) || null,
        title: title,
        year: parseInt(year) || 0,
        date_str: dateStr,
        time_str: timeStr,
        brief: brief,
        context_files: selectedContext,
        fog_of_war: fogOfWar,
        timeline: timeline
      });
      
      await refreshFileList();
      setSelectedFile(res.data.filename);
      setFileContent(res.data.content);
      setActiveTab("read");
      
    } catch (err) {
      console.error(err);
      toast("Generation failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleMergeSelected = async () => {
    if (selectedManageFiles.length < 2) return toast("Select at least 2 files to merge.", "warning");
    
    const sortedFiles = [...selectedManageFiles].sort();
    const ok = await confirm(`Merge these ${sortedFiles.length} files into a single scene?\n\n${sortedFiles.join('\n')}`, { title: "Merge Scenes", confirmLabel: "Merge" });
    if (!ok) return;

    try {
      await axios.post(`${API_URL}/merge/scenes/${profile}`, { filenames: sortedFiles });
      toast("Merged successfully!", "success");
      refreshFileList(); 
      setSelectedManageFiles([]); 
    } catch (err) {
      toast("Merge failed: " + (err.response?.data?.detail || err.message), "error");
    }
  };

  const handleReadFile = async (filename) => {
    setSelectedFile(filename);
    try {
      const res = await axios.get(`${API_URL}/file/${profile}/${filename}`);
      setFileContent(res.data.content);
    } catch (err) { toast("Failed to load file content.", "error"); }
  };

  const handleSaveEdit = async () => {
    try {
      await axios.post(`${API_URL}/scene/save/${profile}`, {
        filename: selectedFile,
        content: fileContent
      });
      toast("File saved successfully.", "success");
    } catch (err) { toast("Save failed.", "error"); }
  };

  const handleDelete = async () => {
    if (selectedManageFiles.length === 0) return toast("No files selected.", "warning");
    
    const fileListString = selectedManageFiles.map(f => `• ${f}`).join('\n');
    const count = selectedManageFiles.length;

    const firstConfirm = window.confirm(
      `You are about to delete ${count} file(s):\n\n${fileListString}\n\nDo you want to proceed?`
    );
    if (!firstConfirm) return;

    const secondConfirm = window.confirm(
      `⚠️ FINAL WARNING ⚠️\n\nThis action is PERMANENT and cannot be undone.\n\nAre you absolutely sure?`
    );
    if (!secondConfirm) return;

    try {
      await axios.post(`${API_URL}/files/bulk_delete/${profile}`, { filenames: selectedManageFiles });
      toast("Files deleted successfully.", "success");
      refreshFileList();
      setSelectedManageFiles([]);
    } catch (err) {
      toast("Delete failed: " + (err.response?.data?.detail || err.message), "error");
    }
  };

  // --- RENDERERS ---

  const renderTabs = () => (
    <div style={styles.tabContainer}>
      {[
        { id: "write", label: "Write", icon: <PenTool size={14} /> },
        { id: "read", label: "Read", icon: <BookOpen size={14} /> },
        { id: "edit", label: "Edit", icon: <Edit size={14} /> },
        { id: "manage", label: "Manage", icon: <FileMinus size={14} /> },
      ].map(tab => (
        <button 
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          style={{
            ...styles.tabButton,
            ...(activeTab === tab.id ? styles.tabButtonActive : {})
          }}
        >
          {tab.icon} {tab.label}
        </button>
      ))}
    </div>
  );

  return (
    <div style={styles.scrollWrapper}>
      <div style={styles.container}>
        
        {/* --- MODULE HEADER --- */}
        <div style={styles.header}>
          <h2 style={styles.title}>
            <PenTool size={28} color="#ef4444" /> Scene Creator
          </h2>
          <p style={styles.subtitle}>
            The primary narrative engine. Draft new scenes, review history, or edit existing prose.
          </p>
        </div>

        {renderTabs()}

        {/* --- WRITE TAB --- */}
        {activeTab === "write" && (
          <div style={styles.formContainer}>
            
            {/* Row 1: Chapter & Title */}
            <div style={styles.row}>
              {useChapters && (
                <div style={{ flex: 1, display: 'flex', gap: '10px' }}>
                  <div style={{ flex: 1 }}>
                    <label style={styles.label}>Chapter</label>
                    <input 
                      type="number" 
                      value={chapter} 
                      onChange={e => setChapter(e.target.value)} 
                      style={styles.input} 
                    />
                  </div>
                  
                  {/* Part System Toggle */}
                  {useParts ? (
                    <div style={{ width: '60px', position: 'relative' }}>
                      <label style={styles.label}>Part</label>
                      <input 
                        type="number" 
                        min="1"
                        value={part} 
                        onChange={e => setPart(e.target.value)} 
                        style={{...styles.input, borderColor: '#3b82f6', color: '#60a5fa'}} 
                      />
                      {/* Close Button to Disable Parts */}
                      <div 
                        onClick={() => setUseParts(false)}
                        style={{position:'absolute', top:'-5px', right:'-5px', background:'#333', borderRadius:'50%', cursor:'pointer', padding:'2px'}}
                      >
                        <X size={10} />
                      </div>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                       <button 
                         onClick={() => setUseParts(true)}
                         style={{...styles.iconButton, height:'42px', fontSize:'11px', padding:'0 10px', gap:'4px'}}
                         title="Split chapter into parts"
                       >
                         <Plus size={12} /> Part
                       </button>
                    </div>
                  )}
                </div>
              )}
              <div style={{ flex: 2 }}>
                <label style={styles.label}>Scene Title</label>
                <input 
                  type="text" 
                  value={title} 
                  onChange={e => setTitle(e.target.value)} 
                  placeholder="Optional (Auto-Generated if empty)" 
                  style={styles.input} 
                />
              </div>
              
              {/* Timeline Input / Dropdown */}
              <div style={{ flex: 1 }}>
                <label style={styles.label}>Timeline (Multiverse)</label>
                {availableTimelines.length > 0 ? (
                  <select 
                    value={timeline} 
                    onChange={e => setTimeline(e.target.value)} 
                    style={{...styles.input, borderColor: '#a855f7'}}
                  >
                    <option value="">Universal (No Timeline)</option>
                    {availableTimelines.map((tl, idx) => (
                      <option key={idx} value={tl.Name}>{tl.Name}</option>
                    ))}
                  </select>
                ) : (
                  <input 
                    type="text" 
                    value={timeline} 
                    onChange={e => setTimeline(e.target.value)} 
                    placeholder="e.g. Prime Earth" 
                    style={{...styles.input, borderColor: '#a855f7'}} 
                  />
                )}
              </div>
            </div>

            {/* Row 2: Chronology (Conditional) */}
            {useTimeSystem ? (
              <div style={styles.row}>
                {showYear && (
                  <div style={{ flex: 1 }}>
                    <label style={styles.label}>Year</label>
                    <input 
                      type="number" 
                      value={year} 
                      onChange={e => setYear(e.target.value)} 
                      placeholder="YYYY" 
                      style={styles.input} 
                    />
                  </div>
                )}
                
                {showDate && (
                  <div style={{ flex: 1 }}>
                    <label style={styles.label}>Date</label>
                    <input 
                      type="text" 
                      value={dateStr} 
                      onChange={e => setDateStr(e.target.value)} 
                      placeholder="e.g. March 6" 
                      style={styles.input} 
                    />
                  </div>
                )}
                
                {showClock && (
                  <div style={{ flex: 1 }}>
                    <label style={styles.label}>Time</label>
                    <input 
                      type="text" 
                      value={timeStr} 
                      onChange={e => setTimeStr(e.target.value)} 
                      placeholder="e.g. 14:00" 
                      style={styles.input} 
                    />
                  </div>
                )}
              </div>
            ) : (
              <div style={styles.disabledBox}>
                <Clock size={14} /> 
                <span>Time System is Disabled (Settings). Chronology will be inferred from context.</span>
              </div>
            )}

            {/* Row 3: Context Selector */}
            <div style={{ zIndex: 10 }}>
              <MultiSelect 
                options={["Auto (Last 3 Scenes)", ...files]} 
                selected={selectedContext}
                onChange={setSelectedContext}
                placeholder="Select context files to guide the AI..."
              />
            </div>

            {/* Row 4: Brief */}
            <div>
              <label style={styles.label}>Scene Brief</label>
              <textarea 
                value={brief} 
                onChange={e => setBrief(e.target.value)} 
                placeholder="Describe key events, conflicts, and outcomes..." 
                style={styles.textarea} 
              />
            </div>

            {/* Row 5: Controls */}
            <div style={styles.checkboxContainer}>
              <input 
                type="checkbox" 
                checked={fogOfWar} 
                onChange={e => setFogOfWar(e.target.checked)} 
                id="fog" 
              />
              <label htmlFor="fog" style={styles.checkboxLabel}>
                Enable Fog of War (Private thoughts are tagged separately)
              </label>
            </div>

            <button 
              onClick={handleGenerate} 
              disabled={isGenerating} 
              style={styles.primaryButton}
            >
              {isGenerating ? "Drafting Scene (This may take a minute)..." : <><Play size={16} /> Generate Scene</>}
            </button>
          </div>
        )}

        {/* --- READ / EDIT / MANAGE TABS --- */}
        {activeTab !== "write" && (
          <div style={styles.formContainer}>

            {/* File Selector - Only visible for Read/Edit modes */}
            {(activeTab === "read" || activeTab === "edit") && (
              <div style={styles.row}>
                <select 
                  value={selectedFile} 
                  onChange={e => handleReadFile(e.target.value)} 
                  style={{ ...styles.input, flex: 1 }}
                >
                  <option value="">-- Select File --</option>
                  {files.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
                <button onClick={refreshFileList} style={styles.iconButton} title="Refresh List">
                  <RefreshCw size={18} />
                </button>
              </div>
            )}

            {activeTab === "read" && fileContent && (
              <div style={styles.readerView}>{fileContent}</div>
            )}

            {activeTab === "edit" && fileContent && (
              <>
                <textarea 
                  value={fileContent} 
                  onChange={e => setFileContent(e.target.value)} 
                  style={{ ...styles.textarea, height: '600px', fontFamily: 'monospace' }} 
                />
                <button onClick={handleSaveEdit} style={styles.primaryButton}>
                  <Save size={16} /> Save Changes
                </button>
              </>
            )}

            {/* --- MANAGE TAB --- */}
            {activeTab === "manage" && (
              <div style={styles.formContainer}>
                
                {/* Header Info */}
                <div style={{ marginBottom: '5px' }}>
                  <span style={{ fontSize: '13px', color: '#a1a1aa' }}>
                    {selectedManageFiles.length} files selected
                  </span>
                </div>

                {/* File List */}
                <div style={{ border: '1px solid #27272a', borderRadius: '6px', maxHeight: '500px', overflowY: 'auto' }}>
                  {files.map(f => (
                    <div 
                      key={f} 
                      style={{
                        display: 'flex', alignItems: 'center', gap: '12px', padding: '10px',
                        borderBottom: '1px solid #27272a',
                        background: selectedManageFiles.includes(f) ? 'rgba(59, 130, 246, 0.05)' : 'transparent'
                      }}
                    >
                      <input 
                        type="checkbox"
                        checked={selectedManageFiles.includes(f)}
                        onChange={(e) => {
                          if (e.target.checked) setSelectedManageFiles([...selectedManageFiles, f]);
                          else setSelectedManageFiles(selectedManageFiles.filter(x => x !== f));
                        }}
                        style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: '#3b82f6' }}
                      />
                      <span style={{ fontSize: '14px', color: '#e4e4e7' }}>{f}</span>
                    </div>
                  ))}
                  {files.length === 0 && <div style={{ padding: '20px', textAlign: 'center', color: '#555' }}>No files found.</div>}
                </div>

                {/* Management Toolbar */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '10px' }}>
                  
                  <button 
                    onClick={handleMergeSelected} 
                    disabled={selectedManageFiles.length < 2}
                    style={{
                      ...styles.primaryButton, 
                      background: selectedManageFiles.length < 2 ? '#27272a' : '#a855f7',
                      opacity: selectedManageFiles.length < 2 ? 0.5 : 1
                    }}
                  >
                    <Merge size={16} /> Merge
                  </button>

                  <button 
                    onClick={handleDelete} 
                    disabled={selectedManageFiles.length === 0}
                    style={{
                      ...styles.primaryButton, 
                      background: selectedManageFiles.length === 0 ? '#27272a' : '#dc2626',
                      opacity: selectedManageFiles.length === 0 ? 0.5 : 1
                    }}
                  >
                    <Trash2 size={16} /> Delete
                  </button>

                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// --- CSS STYLES ---
const styles = {
  scrollWrapper: { 
    height: '100%', width: '100%', overflowY: 'auto', position: 'relative' 
  },
  container: { 
    maxWidth: '1100px', width: '100%', margin: '0 auto', padding: '40px', boxSizing: 'border-box', color: '#e4e4e7' 
  },
  
  // Header
  header: { marginBottom: '30px' },
  title: { margin: 0, display: 'flex', alignItems: 'center', gap: '12px', fontSize: '24px', color: '#e4e4e7' },
  subtitle: { margin: '5px 0 0 0', color: '#64748b', fontSize: '14px', marginLeft: '40px' },

  // Tabs
  tabContainer: { 
    display: 'flex', gap: '10px', marginBottom: '20px', borderBottom: '1px solid #27272a', paddingBottom: '10px' 
  },
  tabButton: { 
    padding: '8px 16px', background: 'transparent', border: 'none', color: '#a1a1aa', 
    cursor: 'pointer', fontWeight: '500', fontSize: '14px', display: 'flex', 
    alignItems: 'center', gap: '8px', borderRadius: '4px', transition: 'all 0.2s' 
  },
  tabButtonActive: { 
    background: '#ef4444', color: '#ffffff', fontWeight: '600' 
  },
  
  // Forms & Inputs
  formContainer: { display: 'flex', flexDirection: 'column', gap: '20px' },
  row: { display: 'flex', gap: '20px' },
  label: { 
    display: 'block', marginBottom: '8px', color: '#a1a1aa', fontSize: '12px', 
    fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' 
  },
  input: { 
    width: '100%', padding: '12px', background: '#18181b', border: '1px solid #3f3f46', 
    color: '#fff', borderRadius: '6px', outline: 'none', fontSize: '14px', boxSizing: 'border-box' 
  },
  textarea: { 
    width: '100%', padding: '12px', background: '#18181b', border: '1px solid #3f3f46', 
    color: '#fff', borderRadius: '6px', outline: 'none', fontSize: '14px', height: '250px', 
    resize: 'vertical', boxSizing: 'border-box', lineHeight: '1.6' 
  },
  
  // MultiSelect Styles
  dropdownMenu: { 
    position: 'absolute', top: '100%', left: 0, width: '100%', background: '#18181b', 
    border: '1px solid #3f3f46', borderRadius: '6px', maxHeight: '250px', overflowY: 'auto', 
    zIndex: 100, marginTop: '4px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' 
  },
  dropdownItem: { 
    padding: '10px 12px', cursor: 'pointer', fontSize: '13px', display: 'flex', 
    justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #27272a',
    transition: 'background 0.1s'
  },
  tag: { 
    background: '#3f3f46', color: '#fff', padding: '4px 8px', borderRadius: '4px', 
    fontSize: '12px', display: 'flex', alignItems: 'center', border: '1px solid #52525b'
  },

  // Controls
  checkboxContainer: { display: 'flex', alignItems: 'center', gap: '10px' },
  checkboxLabel: { color: '#a1a1aa', cursor: 'pointer', fontSize: '14px' },
  primaryButton: { 
    padding: '12px 24px', background: '#ef4444', color: 'white', border: 'none', 
    borderRadius: '6px', cursor: 'pointer', fontWeight: '600', display: 'flex', 
    alignItems: 'center', justifyContent: 'center', gap: '10px', transition: 'background 0.2s' 
  },
  iconButton: { 
    padding: '10px', background: '#27272a', border: '1px solid #3f3f46', color: '#fff', 
    borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' 
  },
  
  // Modes
  readerView: { 
    background: '#18181b', padding: '40px', borderRadius: '8px', border: '1px solid #27272a', 
    whiteSpace: 'pre-wrap', lineHeight: '1.8', fontSize: '16px', fontFamily: 'verdana', color: '#f4f4f5' 
  },
  dangerZone: { 
    padding: '20px', border: '1px solid #7f1d1d', background: '#450a0a', borderRadius: '8px', color: '#fecaca' 
  },
  disabledBox: { 
    display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', background: '#27272a', 
    border: '1px dashed #3f3f46', borderRadius: '6px', color: '#71717a', fontSize: '13px', fontStyle: 'italic' 
  }
};