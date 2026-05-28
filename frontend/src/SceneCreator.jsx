import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Play, Save, Trash2, RefreshCw, 
  PenTool, BookOpen, Edit, FileMinus, 
  Clock, ChevronDown, Check, X, Merge, Plus,
  FlaskConical, FileText, AlertTriangle
} from 'lucide-react';
import { API_URL } from './config';
import { toast, confirm } from './components/Notifications';

/**
 * MultiSelect Component
 */
const MultiSelect = ({ options, selected, onChange, placeholder }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

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
 * Dry Run Modal Component
 */
const DryRunModal = ({ result, onClose, onProceed, onOutlineChange }) => {
  if (!result) return null;

  return (
    <div style={styles.modalOverlay}>
      <div style={styles.modalBox}>

        {/* Header */}
        <div style={styles.modalHeader}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <FlaskConical size={20} color="#22c55e" />
            <h3 style={{ margin: 0, fontSize: '18px', color: '#fff' }}>Dry Run — Planner Output</h3>
          </div>
          <button onClick={onClose} style={styles.modalClose}><X size={18} /></button>
        </div>

        <div style={styles.modalBody}>

          {/* Inferred Chronology */}
          {(result.inferred_year || result.inferred_date || result.inferred_time) && (
            <div style={styles.dryRunSection}>
              <div style={styles.dryRunLabel}>INFERRED CHRONOLOGY</div>
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {result.inferred_year > 0 && (
                  <span style={styles.dryRunChip}>Year: {result.inferred_year}</span>
                )}
                {result.inferred_date && (
                  <span style={styles.dryRunChip}>Date: {result.inferred_date}</span>
                )}
                {result.inferred_time && (
                  <span style={styles.dryRunChip}>Time: {result.inferred_time}</span>
                )}
              </div>
            </div>
          )}

          {/* Retrieved Documents */}
          <div style={styles.dryRunSection}>
            <div style={styles.dryRunLabel}>
              DOCUMENTS RETRIEVED BY LIBRARIAN ({result.retrieved_titles.length})
            </div>
            {result.retrieved_titles.length === 0 ? (
              <div style={{ color: '#52525b', fontSize: '13px', fontStyle: 'italic' }}>
                No documents retrieved. The Librarian found no relevant fragments.
              </div>
            ) : (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {result.retrieved_titles.map((t, i) => (
                  <span key={i} style={styles.logTag}>{t}</span>
                ))}
              </div>
            )}
          </div>

          {/* Active Spoilers */}
          {result.active_spoilers && (
            <div style={styles.dryRunSection}>
              <div style={styles.dryRunLabel}>ACTIVE SPOILER BLOCKS</div>
              <div style={{ color: '#f87171', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.6' }}>
                {result.active_spoilers}
              </div>
            </div>
          )}

          {/* Beat Sheet Outline */}
          <div style={styles.dryRunSection}>
            <div style={styles.dryRunLabel}>PLANNER BEAT SHEET</div>
            <textarea
              value={result.outline}
              onChange={e => onOutlineChange(e.target.value)}
              style={{ ...styles.outlineBox, resize: 'vertical', minHeight: '200px', cursor: 'text', fontFamily: 'inherit' }}
            />
          </div>

          {/* Warning */}
          <div style={styles.dryRunWarning}>
            <AlertTriangle size={14} color="#f59e0b" />
            <span>Review the outline and retrieved documents. If something looks wrong, close and adjust your brief or context before generating.</span>
          </div>

        </div>

        {/* Footer */}
        <div style={styles.modalFooter}>
          <button onClick={onClose} style={styles.modalCancelBtn}>
            Adjust Brief
          </button>
          <button onClick={onProceed} style={styles.modalProceedBtn}>
            <Play size={16} /> Generate Full Scene
          </button>
        </div>

      </div>
    </div>
  );
};

/**
 * Simple line-by-line diff using longest common subsequence.
 * Returns array of {type: 'added'|'removed'|'unchanged', text: string}
 */
function computeLineDiff(original, current) {
  const oldLines = original.split('\n');
  const newLines = current.split('\n');
  
  // Build LCS table
  const m = oldLines.length;
  const n = newLines.length;
  const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (oldLines[i-1] === newLines[j-1]) {
        dp[i][j] = dp[i-1][j-1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
      }
    }
  }
  
  // Backtrack to build diff
  const result = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldLines[i-1] === newLines[j-1]) {
      result.unshift({ type: 'unchanged', text: oldLines[i-1] });
      i--; j--;
    } else if (j > 0 && (i === 0 || dp[i][j-1] >= dp[i-1][j])) {
      result.unshift({ type: 'added', text: newLines[j-1] });
      j--;
    } else {
      result.unshift({ type: 'removed', text: oldLines[i-1] });
      i--;
    }
  }
  return result;
}

/**
 * Scene Creator Module
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
  
  const [useTimeSystem, setUseTimeSystem] = useState(true);
  const [useChapters, setUseChapters] = useState(true);
  const [showYear, setShowYear] = useState(true);
  const [showDate, setShowDate] = useState(true);
  const [showClock, setShowClock] = useState(true);

  const [chapter, setChapter] = useState(1);
  const [part, setPart] = useState(1);
  const [title, setTitle] = useState("");
  const [brief, setBrief] = useState("");
  const [timeline, setTimeline] = useState("");
  const [availableTimelines, setAvailableTimelines] = useState([]);
  const [selectedContext, setSelectedContext] = useState([]);
  const [year, setYear] = useState("");
  const [dateStr, setDateStr] = useState("");
  const [timeStr, setTimeStr] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [fogOfWar, setFogOfWar] = useState(false);

  // Editor State
  const [selectedFile, setSelectedFile] = useState("");
  const [fileContent, setFileContent] = useState("");

  // Chapter Partitioning & Management
  const [useParts, setUseParts] = useState(false);
  const [selectedManageFiles, setSelectedManageFiles] = useState([]);

  // --- GENERATION LOG STATE ---
  const [generationLog, setGenerationLog] = useState(null);
  const [isLoadingLog, setIsLoadingLog] = useState(false);
  const [showLog, setShowLog] = useState(false);

// --- WORLD CONSEQUENCES STATE ---
  const [consequences, setConsequences] = useState(null);
  const [isAnalyzingConsequences, setIsAnalyzingConsequences] = useState(false);
  const [showConsequences, setShowConsequences] = useState(false);

// --- DIFF STATE ---
  const [showDiff, setShowDiff] = useState(false);
  const [diffData, setDiffData] = useState(null);
  const [isLoadingDiff, setIsLoadingDiff] = useState(false);

  // --- DRY RUN STATE ---
  const [isDryRunning, setIsDryRunning] = useState(false);
  const [dryRunResult, setDryRunResult] = useState(null);

  // --- SPOILER WARNING STATE ---
  const [spoilerWarnings, setSpoilerWarnings] = useState([]);

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

  // --- SHARED PAYLOAD BUILDER ---
  const buildPayload = (overrideOutline = "") => ({
    chapter: parseInt(chapter) || null,
    title: title,
    year: parseInt(year) || 0,
    date_str: dateStr,
    time_str: timeStr,
    brief: brief,
    context_files: selectedContext,
    fog_of_war: fogOfWar,
    timeline: timeline,
    override_outline: overrideOutline
  });

  // --- ACTIONS ---

  const handleDryRun = async () => {
    if (!brief) return toast("Please provide a Scene Brief.", "warning");
    setIsDryRunning(true);
    try {
      const res = await axios.post(`${API_URL}/scene/dry_run/${profile}`, buildPayload());
      setDryRunResult(res.data);
    } catch (err) {
      toast("Dry run failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setIsDryRunning(false);
    }
  };

  const handleDryRunOutlineChange = (newOutline) => {
    setDryRunResult(prev => ({ ...prev, outline: newOutline }));
  };

  const handleGenerate = async (overrideOutline = "") => {
    if (!brief) return toast("Please provide a Scene Brief.", "warning");
    setDryRunResult(null);
    setIsGenerating(true);
    // Check for upcoming spoilers near the scene date
    if (year || dateStr) {
      try {
        const warnRes = await axios.post(`${API_URL}/scene/spoiler_check/${profile}`, buildPayload());
        setSpoilerWarnings(warnRes.data.warnings || []);
      } catch (err) {
        setSpoilerWarnings([]);
      }
    }
    try {
      const res = await axios.post(`${API_URL}/scene/generate/${profile}`, buildPayload(overrideOutline));
      await refreshFileList();
      setSelectedFile(res.data.filename);
      setFileContent(res.data.content);
      setActiveTab("read");
      fetchGenerationLog(res.data.filename);
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
    setGenerationLog(null);
    setDiffData(null);
    setShowDiff(false);
    try {
      const res = await axios.get(`${API_URL}/file/${profile}/${filename}`);
      setFileContent(res.data.content);
      fetchGenerationLog(filename);
      setConsequences(null);
      setShowConsequences(false);
      fetchDiff(filename);
    } catch (err) { toast("Failed to load file content.", "error"); }
  };

  const fetchGenerationLog = async (filename) => {
    if (!filename) return;
    setIsLoadingLog(true);
    try {
      const res = await axios.get(`${API_URL}/scene/log/${profile}/${encodeURIComponent(filename)}`);
      setGenerationLog(res.data.log || null);
    } catch (err) {
      console.error("Failed to fetch generation log:", err);
      setGenerationLog(null);
    } finally {
      setIsLoadingLog(false);
    }
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
      for (const filename of selectedManageFiles) {
        await axios.delete(`${API_URL}/scene/${profile}/${filename}`);
      }
      toast("Files deleted successfully.", "success");
      refreshFileList();
      setSelectedManageFiles([]);
    } catch (err) {
      toast("Delete failed: " + (err.response?.data?.detail || err.message), "error");
    }
  };

  const fetchConsequences = async (filename) => {
    if (!filename) return;
    setIsAnalyzingConsequences(true);
    setConsequences(null);
    try {
      const res = await axios.post(
        `${API_URL}/scene/consequences/${profile}/${encodeURIComponent(filename)}`
      );
      setConsequences(res.data.consequences || []);
      setShowConsequences(true);
    } catch (err) {
      console.error("Consequences analysis failed:", err);
      setConsequences([]);
    } finally {
      setIsAnalyzingConsequences(false);
    }
  };

  const fetchDiff = async (filename) => {
    if (!filename) return;
    setIsLoadingDiff(true);
    try {
      const res = await axios.get(`${API_URL}/scene/diff/${profile}/${encodeURIComponent(filename)}`);
      setDiffData(res.data);
    } catch (err) {
      console.error("Failed to fetch diff:", err);
      setDiffData(null);
    } finally {
      setIsLoadingDiff(false);
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

      {/* --- DRY RUN MODAL --- */}
      {dryRunResult && (
        <DryRunModal
          result={dryRunResult}
          onClose={() => setDryRunResult(null)}
          onProceed={() => { const outline = dryRunResult?.outline || ""; setDryRunResult(null); handleGenerate(outline); }}
          onOutlineChange={handleDryRunOutlineChange}
        />
      )}

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
                  
                    {useParts ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <div style={{ width: '60px', position: 'relative' }}>
                        <label style={styles.label}>Part</label>
                        <input 
                          type="number" 
                          min="1"
                          value={part} 
                          onChange={e => setPart(e.target.value)} 
                          style={{...styles.input, borderColor: '#3b82f6', color: '#60a5fa'}} 
                        />
                        <div 
                          onClick={() => setUseParts(false)}
                          style={{position:'absolute', top:'-5px', right:'-5px', background:'#333', borderRadius:'50%', cursor:'pointer', padding:'2px'}}
                        >
                          <X size={10} />
                        </div>
                      </div>
                      {parseInt(part) > 1 && (
                        <div style={{ fontSize: '11px', color: '#60a5fa', background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.2)', borderRadius: '4px', padding: '4px 8px', whiteSpace: 'nowrap' }}>
                          Ch{String(chapter).padStart(2,'0')} Part {parseInt(part) - 1} will be auto-loaded
                        </div>
                      )}
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

            {/* Row 2: Chronology */}
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
                onChange={e => { setBrief(e.target.value); setSpoilerWarnings([]); }} 
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

            {/* Spoiler Proximity Warnings */}
            {spoilerWarnings.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {spoilerWarnings.map((w, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', padding: '10px 14px', background: 'rgba(168,85,247,0.05)', border: '1px solid rgba(168,85,247,0.3)', borderRadius: '6px' }}>
                    <span style={{ fontSize: '14px', flexShrink: 0 }}>⚠️</span>
                    <div>
                      <div style={{ fontSize: '12px', color: '#a855f7', fontWeight: '700', marginBottom: '2px' }}>SPOILER PROXIMITY WARNING</div>
                      <div style={{ fontSize: '12px', color: '#a1a1aa' }}>{w.message}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Row 6: Action Buttons */}
            <div style={{ display: 'flex', gap: '12px' }}>
              <button 
                onClick={handleDryRun} 
                disabled={isDryRunning || isGenerating} 
                style={styles.dryRunButton}
                title="Run only the Planner — see the outline and retrieved documents before spending tokens on the full draft"
              >
                {isDryRunning 
                  ? "Running Planner..." 
                  : <><FlaskConical size={16} /> Dry Run</>
                }
              </button>
              <button 
                onClick={handleGenerate} 
                disabled={isGenerating || isDryRunning} 
                style={{ ...styles.primaryButton, flex: 1 }}
              >
                {isGenerating ? "Drafting Scene (This may take a minute)..." : <><Play size={16} /> Generate Scene</>}
              </button>
            </div>
          </div>
        )}

        {/* --- READ / EDIT / MANAGE TABS --- */}
        {activeTab !== "write" && (
          <div style={styles.formContainer}>

            {/* File Selector */}
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

            {/* READ TAB */}
            {activeTab === "read" && fileContent && (
              <>
                <div style={styles.readerView}>{fileContent}</div>

                {/* --- GENERATION LOG PANEL --- */}
                {selectedFile && (
                  <div style={styles.logBox}>
                    <div onClick={() => setShowLog(!showLog)} style={styles.logHeader}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <BookOpen size={15} color="#60a5fa" />
                        <span style={{ fontWeight: '600', fontSize: '13px' }}>Generation Log</span>
                        {generationLog && !showLog && (
                          <span style={{ fontSize: '11px', color: '#52525b' }}>
                            {generationLog.timestamp}
                          </span>
                        )}
                      </div>
                      <ChevronDown 
                        size={15} 
                        color="#555" 
                        style={{ transform: showLog ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} 
                      />
                    </div>

                    {showLog && (
                      <div style={styles.logBody}>
                        {isLoadingLog ? (
                          <div style={{ color: '#52525b', fontSize: '13px' }}>Loading...</div>
                        ) : !generationLog ? (
                          <div style={{ color: '#52525b', fontSize: '13px', fontStyle: 'italic' }}>
                            No log found for this file. Logs are recorded for scenes generated after this feature was added.
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>

                            {/* Status Row */}
                            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
                              <div style={{
                                ...styles.logBadge,
                                background: generationLog.validator_result === 'PASS' ? 'rgba(34,197,94,0.1)' : 'rgba(245,158,11,0.1)',
                                border: `1px solid ${generationLog.validator_result === 'PASS' ? '#22c55e' : '#f59e0b'}`,
                                color: generationLog.validator_result === 'PASS' ? '#22c55e' : '#f59e0b'
                              }}>
                                {generationLog.validator_result === 'PASS' 
                                  ? <Check size={11} /> 
                                  : <X size={11} />
                                }
                                {generationLog.validator_result === 'PASS' ? 'Validator Passed' : 'Force-Passed (Revision Cap Hit)'}
                              </div>

                              <div style={{ ...styles.logBadge, background: 'rgba(59,130,246,0.1)', border: '1px solid #1d4ed8', color: '#60a5fa' }}>
                                {generationLog.revision_count} Draft{generationLog.revision_count !== 1 ? 's' : ''}
                              </div>

                              {generationLog.timeline && (
                                <div style={{ ...styles.logBadge, background: 'rgba(168,85,247,0.1)', border: '1px solid #7e22ce', color: '#a855f7' }}>
                                  {generationLog.timeline}
                                </div>
                              )}

                              <div style={{ marginLeft: 'auto', fontSize: '11px', color: '#52525b', display: 'flex', alignItems: 'center', gap: '5px' }}>
                                <Clock size={11} /> {generationLog.timestamp}
                              </div>
                            </div>

                            {/* Token Usage */}
                            {generationLog.token_usage && generationLog.token_usage.total > 0 && (
                              <div style={{ width: '100%', marginTop: '4px' }}>
                                <div style={styles.logLabel}>TOKEN USAGE</div>
                                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '6px' }}>
                                  {['planner', 'drafter', 'validator', 'style'].map(role => {
                                    const usage = generationLog.token_usage[role];
                                    if (!usage || usage.total === 0) return null;
                                    return (
                                      <div key={role} style={{ ...styles.logBadge, background: 'rgba(100,116,139,0.1)', border: '1px solid #334155', color: '#94a3b8' }}>
                                        {role.charAt(0).toUpperCase() + role.slice(1)}: {usage.total.toLocaleString()}
                                      </div>
                                    );
                                  })}
                                  <div style={{ ...styles.logBadge, background: 'rgba(100,116,139,0.2)', border: '1px solid #475569', color: '#cbd5e1', fontWeight: '700' }}>
                                    Total: {generationLog.token_usage.total.toLocaleString()}
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Brief */}
                            <div>
                              <div style={styles.logLabel}>BRIEF USED</div>
                              <div style={styles.logValue}>{generationLog.brief}</div>
                            </div>

                            {/* Retrieved Documents */}
                            <div>
                              <div style={styles.logLabel}>
                                DOCUMENTS RETRIEVED BY LIBRARIAN ({generationLog.retrieved_titles.length})
                              </div>
                              {generationLog.retrieved_titles.length === 0 ? (
                                <div style={{ ...styles.logValue, color: '#52525b', fontStyle: 'italic' }}>
                                  None retrieved.
                                </div>
                              ) : (
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '6px' }}>
                                  {generationLog.retrieved_titles.map((t, i) => (
                                    <span key={i} style={styles.logTag}>{t}</span>
                                  ))}
                                </div>
                              )}
                            </div>

                            {/* Active Spoilers */}
                            {generationLog.active_spoilers && (
                              <div>
                                <div style={styles.logLabel}>ACTIVE SPOILER BLOCKS</div>
                                <div style={{ ...styles.logValue, color: '#f87171', fontFamily: 'monospace', fontSize: '12px' }}>
                                  {generationLog.active_spoilers}
                                </div>
                              </div>
                            )}

                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
                {/* --- WORLD CONSEQUENCES PANEL --- */}
                {selectedFile && (
                  <div style={{ ...styles.logBox, borderColor: '#1e3a2f' }}>
                    <div style={{ ...styles.logHeader, background: '#0f1f18' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontSize: '13px', fontWeight: '600', color: '#4ade80' }}>🌐 World Consequences</span>
                        {consequences && consequences.length > 0 && !showConsequences && (
                          <span style={{ fontSize: '11px', color: '#22c55e', background: 'rgba(34,197,94,0.1)', padding: '2px 8px', borderRadius: '10px' }}>
                            {consequences.length} flagged
                          </span>
                        )}
                      </div>
                      <button
                        onClick={() => {
                          if (consequences === null) {
                            fetchConsequences(selectedFile);
                          } else {
                            setShowConsequences(!showConsequences);
                          }
                        }}
                        disabled={isAnalyzingConsequences}
                        style={{ background: 'transparent', border: '1px solid #1e3a2f', color: '#4ade80', padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', fontWeight: '600' }}
                      >
                        {isAnalyzingConsequences ? 'Analyzing...' : consequences === null ? 'Analyze' : showConsequences ? 'Hide' : 'Show'}
                      </button>
                    </div>

                    {showConsequences && consequences !== null && (
                      <div style={{ ...styles.logBody }}>
                        {consequences.length === 0 ? (
                          <div style={{ fontSize: '13px', color: '#52525b', fontStyle: 'italic' }}>
                            No meaningful consequences detected for defined entities.
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                            {consequences.map((c, i) => (
                              <div key={i} style={{ padding: '10px 14px', background: '#0f1f18', border: '1px solid #1e3a2f', borderRadius: '6px' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                                  <span style={{ fontSize: '13px', fontWeight: '700', color: '#4ade80' }}>{c.entity}</span>
                                  <span style={{ fontSize: '10px', color: '#166534', background: 'rgba(34,197,94,0.1)', padding: '2px 6px', borderRadius: '4px' }}>{c.type}</span>
                                </div>
                                <div style={{ fontSize: '12px', color: '#a1a1aa', lineHeight: '1.5' }}>{c.reason}</div>
                              </div>
                            ))}
                            <div style={{ fontSize: '11px', color: '#374151', fontStyle: 'italic', marginTop: '4px' }}>
                              Use the Reaction Tool to generate full responses from these entities.
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}

            {/* EDIT TAB */}
            {activeTab === "edit" && fileContent && (
              <>
                {/* Diff Toggle Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '13px', color: '#52525b' }}>
                    {diffData?.has_diff 
                      ? 'This scene has been edited from the original AI draft.' 
                      : diffData?.original 
                        ? 'No changes from original AI draft.' 
                        : 'No original draft stored for this scene.'}
                  </span>
                  {diffData?.has_diff && (
                    <button
                      onClick={() => setShowDiff(!showDiff)}
                      style={{
                        padding: '6px 14px', background: showDiff ? '#27272a' : 'transparent',
                        border: '1px solid #3f3f46', color: showDiff ? '#fff' : '#a1a1aa',
                        borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600'
                      }}
                    >
                      {showDiff ? 'Hide Diff' : 'Show Diff'}
                    </button>
                  )}
                </div>

                {/* Diff View */}
                {showDiff && diffData?.original && (
                  <div style={{ background: '#09090b', border: '1px solid #27272a', borderRadius: '8px', overflow: 'hidden' }}>
                    <div style={{ padding: '10px 16px', background: '#18181b', borderBottom: '1px solid #27272a', fontSize: '12px', color: '#71717a', display: 'flex', gap: '20px' }}>
                      <span style={{ color: '#f87171' }}>■ Removed</span>
                      <span style={{ color: '#22c55e' }}>■ Added</span>
                      <span style={{ color: '#71717a' }}>■ Unchanged</span>
                    </div>
                    <div style={{ padding: '20px', fontFamily: 'monospace', fontSize: '13px', lineHeight: '1.8', maxHeight: '500px', overflowY: 'auto' }}>
                      {computeLineDiff(diffData.original, diffData.current).map((line, i) => (
                        <div key={i} style={{
                          background: line.type === 'added' ? 'rgba(34,197,94,0.1)' : line.type === 'removed' ? 'rgba(239,68,68,0.1)' : 'transparent',
                          color: line.type === 'added' ? '#22c55e' : line.type === 'removed' ? '#f87171' : '#a1a1aa',
                          padding: '1px 8px',
                          borderLeft: `3px solid ${line.type === 'added' ? '#22c55e' : line.type === 'removed' ? '#ef4444' : 'transparent'}`
                        }}>
                          {line.type === 'added' ? '+ ' : line.type === 'removed' ? '- ' : '  '}
                          {line.text}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

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

            {/* MANAGE TAB */}
            {activeTab === "manage" && (
              <div style={styles.formContainer}>
                
                <div style={{ marginBottom: '5px' }}>
                  <span style={{ fontSize: '13px', color: '#a1a1aa' }}>
                    {selectedManageFiles.length} files selected
                  </span>
                </div>

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
  header: { marginBottom: '30px' },
  title: { margin: 0, display: 'flex', alignItems: 'center', gap: '12px', fontSize: '24px', color: '#e4e4e7' },
  subtitle: { margin: '5px 0 0 0', color: '#64748b', fontSize: '14px', marginLeft: '40px' },

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

  checkboxContainer: { display: 'flex', alignItems: 'center', gap: '10px' },
  checkboxLabel: { color: '#a1a1aa', cursor: 'pointer', fontSize: '14px' },
  primaryButton: { 
    padding: '12px 24px', background: '#ef4444', color: 'white', border: 'none', 
    borderRadius: '6px', cursor: 'pointer', fontWeight: '600', display: 'flex', 
    alignItems: 'center', justifyContent: 'center', gap: '10px', transition: 'background 0.2s' 
  },
  dryRunButton: {
    padding: '12px 20px', background: 'transparent', color: '#22c55e',
    border: '1px solid #22c55e', borderRadius: '6px', cursor: 'pointer',
    fontWeight: '600', display: 'flex', alignItems: 'center',
    justifyContent: 'center', gap: '10px', transition: 'all 0.2s',
    whiteSpace: 'nowrap'
  },
  iconButton: { 
    padding: '10px', background: '#27272a', border: '1px solid #3f3f46', color: '#fff', 
    borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' 
  },
  
  readerView: { 
    background: '#18181b', padding: '40px', borderRadius: '8px', border: '1px solid #27272a', 
    whiteSpace: 'pre-wrap', lineHeight: '1.8', fontSize: '16px', fontFamily: 'verdana', color: '#f4f4f5' 
  },
  disabledBox: { 
    display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', background: '#27272a', 
    border: '1px dashed #3f3f46', borderRadius: '6px', color: '#71717a', fontSize: '13px', fontStyle: 'italic' 
  },

  // Generation Log
  logBox: {
    border: '1px solid #27272a', borderRadius: '8px', overflow: 'hidden', background: '#111'
  },
  logHeader: {
    padding: '12px 16px', background: '#18181b', cursor: 'pointer',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center'
  },
  logBody: {
    padding: '20px', borderTop: '1px solid #27272a'
  },
  logBadge: {
    display: 'inline-flex', alignItems: 'center', gap: '5px',
    padding: '4px 10px', borderRadius: '12px', fontSize: '11px', fontWeight: '700'
  },
  logLabel: {
    fontSize: '10px', color: '#52525b', fontWeight: '700',
    textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '6px'
  },
  logValue: {
    fontSize: '13px', color: '#a1a1aa', lineHeight: '1.5'
  },
  logTag: {
    fontSize: '12px', background: '#1e293b', border: '1px solid #334155',
    color: '#94a3b8', padding: '3px 8px', borderRadius: '4px'
  },

  // Dry Run Modal
  modalOverlay: {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)',
    zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
    backdropFilter: 'blur(2px)'
  },
  modalBox: {
    background: '#111', border: '1px solid #27272a', borderRadius: '10px',
    width: '700px', maxWidth: '90vw', maxHeight: '85vh',
    display: 'flex', flexDirection: 'column', boxShadow: '0 20px 60px rgba(0,0,0,0.8)'
  },
  modalHeader: {
    padding: '20px 24px', borderBottom: '1px solid #27272a',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center'
  },
  modalClose: {
    background: 'transparent', border: 'none', color: '#666', cursor: 'pointer', padding: '4px'
  },
  modalBody: {
    flex: 1, overflowY: 'auto', padding: '24px',
    display: 'flex', flexDirection: 'column', gap: '20px'
  },
  modalFooter: {
    padding: '16px 24px', borderTop: '1px solid #27272a',
    display: 'flex', justifyContent: 'flex-end', gap: '12px'
  },
  modalCancelBtn: {
    padding: '10px 18px', background: 'transparent', border: '1px solid #3f3f46',
    color: '#a1a1aa', borderRadius: '6px', cursor: 'pointer', fontWeight: '600', fontSize: '13px'
  },
  modalProceedBtn: {
    padding: '10px 20px', background: '#ef4444', border: 'none',
    color: '#fff', borderRadius: '6px', cursor: 'pointer', fontWeight: '700',
    fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px'
  },
  dryRunSection: {
    display: 'flex', flexDirection: 'column', gap: '8px'
  },
  dryRunLabel: {
    fontSize: '10px', color: '#52525b', fontWeight: '700',
    textTransform: 'uppercase', letterSpacing: '0.5px'
  },
  dryRunChip: {
    fontSize: '12px', background: '#1e293b', border: '1px solid #334155',
    color: '#94a3b8', padding: '3px 10px', borderRadius: '12px'
  },
  outlineBox: {
    background: '#0a0a0a', border: '1px solid #27272a', borderRadius: '6px',
    padding: '16px', fontSize: '14px', lineHeight: '1.8', color: '#d4d4d8',
    whiteSpace: 'pre-wrap', fontFamily: 'inherit'
  },
  dryRunWarning: {
    display: 'flex', alignItems: 'flex-start', gap: '10px',
    padding: '12px', background: 'rgba(245,158,11,0.05)',
    border: '1px solid rgba(245,158,11,0.2)', borderRadius: '6px',
    fontSize: '12px', color: '#a1a1aa', lineHeight: '1.5'
  }
};
