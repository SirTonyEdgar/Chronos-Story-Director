import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Save, User, Briefcase, Users, Coins, Zap, Globe, FileJson, 
  Sparkles, ChevronDown, ChevronRight, FileText, Check 
} from 'lucide-react';
import { API_URL } from '../config';
import { TimelineDropdown } from '../components/SharedComponents';
import { toast, confirm } from '../components/Notifications';

// Sub-Components
import CharacterManagerTab from './CharacterManagerTab';
import ProjectsTab from './ProjectsTab';
import SkillsTab from './SkillsTab';
import VariablesTab from './VariablesTab';
import JsonTab from './JsonTab';
import RelationsTab from './RelationsTab';
import AssetsTab from './AssetsTab';

export default function WorldStateTracker({ profile }) {
  
  // --- NAVIGATION STATE ---
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem("chronos_tracker_tab") || "protagonist";
  });

  useEffect(() => {
    localStorage.setItem("chronos_tracker_tab", activeTab);
  }, [activeTab]);

  // --- DATA & SYNC STATE ---
  const [state, setState] = useState(null);
  const [isSaving, setIsSaving] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  const isFirstLoad = useRef(true);

  // --- ANALYSIS TOOLS STATE ---
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // Timeline state for the AI Analyzer
  const [analyzeTimeline, setAnalyzeTimeline] = useState("");

  // --- INITIALIZATION ---
  useEffect(() => { 
    if (profile) {
      fetchState(); 
      fetchFiles();
    }
  }, [profile]);

  useEffect(() => {
    if (state) {
      if (isFirstLoad.current) {
        isFirstLoad.current = false;
      } else {
        setHasUnsavedChanges(true);
      }
    }
  }, [state]);

  const fetchState = async () => {
    try {
      const res = await axios.get(`${API_URL}/state/${profile}`);
      setState(res.data);
      setHasUnsavedChanges(false);
      isFirstLoad.current = true;
    } catch (err) { 
      console.error("Error fetching world state:", err); 
    }
  };

  const fetchFiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/files/${profile}`);
      setFiles(res.data || []);
    } catch (err) { 
      console.error("Error fetching file list:", err); 
    }
  };

  const saveState = async () => {
    setIsSaving(true);
    try {
      await axios.post(`${API_URL}/state/save/${profile}`, state);
      setHasUnsavedChanges(false);
    } catch (err) {
      toast("Save failed: " + err.message, "error");
    } finally {
      setIsSaving(false);
    }
  };

  // --- AI ANALYSIS HANDLERS ---

  const runAnalysis = async () => {
    if (selectedFiles.length === 0) return toast("Please select files to analyze.", "warning");
    
    setIsAnalyzing(true);
    try {
      const res = await axios.post(`${API_URL}/state/analyze/${profile}`, {
        filenames: selectedFiles,
        timeline: analyzeTimeline
      });
      setState(res.data);
      toast("Analysis complete. World state has been updated.", "success");
      setShowAnalysis(false);
      setSelectedFiles([]); 
    } catch (err) {
      console.error(err);
      toast("Analysis Error: " + err.message, "error");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleFileSelection = (fname) => {
    if (selectedFiles.includes(fname)) {
      setSelectedFiles(selectedFiles.filter(f => f !== fname));
    } else {
      setSelectedFiles([...selectedFiles, fname]);
    }
  };

  if (!state) return <div style={styles.loadingState}>Loading World State...</div>;

  const availableTimelines = state.Timelines || [];

  const tabs = [
    { id: "protagonist", label: "Cast Roster", icon: <User size={16} /> },
    { id: "projects", label: "Projects", icon: <Briefcase size={16} /> },
    { id: "relations", label: "Relations", icon: <Users size={16} /> },
    { id: "assets", label: "Assets", icon: <Coins size={16} /> },
    { id: "skills", label: "Skills", icon: <Zap size={16} /> },
    { id: "vars", label: "Variables", icon: <Globe size={16} /> },
    { id: "json", label: "JSON", icon: <FileJson size={16} /> },
  ];

  return (
    <div style={styles.scrollWrapper}>
      <div style={styles.container}>
        
        {/* Module Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>
            📊 World State Tracker
          </h2>
        </div>

        {/* AI Batch Analysis Tools (Collapsible) */}
        <div style={styles.analysisBox}>
          <div 
            onClick={() => setShowAnalysis(!showAnalysis)}
            style={styles.analysisHeader}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontWeight: 'bold' }}>
              <Sparkles size={18} color="#a855f7" /> 
              <span>AI Batch Analysis Tools</span>
            </div>
            {showAnalysis ? <ChevronDown size={18} color="#666" /> : <ChevronRight size={18} color="#666" />}
          </div>

          {showAnalysis && (
            <div style={styles.analysisBody}>
              <p style={styles.analysisHint}>
                Select context files (Scenes, Lore, Plans) to auto-extract data and update the tracker.
              </p>
              
              {/* NEW: Sleek Timeline Selector for Analysis */}
              {availableTimelines.length > 0 && (
                <div style={{ marginBottom: '15px' }}>
                  <label style={{ fontSize: '12px', color: '#a855f7', fontWeight: 'bold', display: 'block', marginBottom: '8px' }}>
                    Target Timeline (Isolate Analysis to Universe)
                  </label>
                  <TimelineDropdown 
                    value={analyzeTimeline} 
                    onChange={setAnalyzeTimeline} 
                    timelines={availableTimelines} 
                  />
                </div>
              )}

              <div style={styles.fileGrid}>
                {files.map(f => (
                  <div 
                    key={f} 
                    onClick={() => toggleFileSelection(f)}
                    style={{
                      ...styles.fileItem,
                      background: selectedFiles.includes(f) ? '#a855f730' : '#1a1a1a',
                      border: selectedFiles.includes(f) ? '1px solid #a855f7' : '1px solid #333',
                      color: selectedFiles.includes(f) ? '#fff' : '#888',
                    }}
                  >
                    <FileText size={12} /> {f}
                  </div>
                ))}
              </div>
              <button 
                onClick={runAnalysis} 
                disabled={isAnalyzing}
                style={styles.analyzeBtn}
              >
                {isAnalyzing ? "Processing Context..." : "Analyze & Update State"}
              </button>
            </div>
          )}
        </div>

        {/* Navigation Tabs */}
        <div style={styles.tabContainer}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                ...styles.tabButton,
                background: activeTab === tab.id ? '#262730' : 'transparent',
                borderBottom: activeTab === tab.id ? '2px solid #ff4b4b' : '2px solid transparent',
                color: activeTab === tab.id ? '#fff' : '#888',
                fontWeight: activeTab === tab.id ? 'bold' : 'normal'
              }}
            >
              {tab.icon} {tab.label}
            </button>
          ))}
        </div>

        {/* Main Content Area */}
        <div style={styles.contentArea}>
          {activeTab === 'protagonist' && (
            <CharacterManagerTab 
              state={state} 
              setState={setState} 
              profile={profile} 
            />
          )}
          {activeTab === "projects" && <ProjectsTab state={state} setState={setState} profile={profile} />}
          {activeTab === "relations" && <RelationsTab state={state} setState={setState} />}
          {activeTab === "assets" && <AssetsTab state={state} setState={setState} />}
          {activeTab === "skills" && <SkillsTab state={state} setState={setState} />}
          {activeTab === "vars" && <VariablesTab state={state} setState={setState} />}
          {activeTab === "json" && <JsonTab state={state} setState={setState} />}
        </div>

      </div>

      {/* Floating Action Button (FAB) for Persistence */}
      <button 
        onClick={saveState}
        disabled={isSaving || !hasUnsavedChanges}
        style={{
          ...styles.fab,
          background: hasUnsavedChanges ? '#22c55e' : '#27272a',
          color: hasUnsavedChanges ? '#000' : '#666',
          cursor: (isSaving || !hasUnsavedChanges) ? 'default' : 'pointer',
          width: hasUnsavedChanges ? '160px' : '130px',
          border: hasUnsavedChanges ? '1px solid #16a34a' : '1px solid #333'
        }}
      >
        {isSaving ? (
          "Saving..."
        ) : (
          <>
            {hasUnsavedChanges ? <Save size={18} /> : <Check size={18} />}
            <span>{hasUnsavedChanges ? "Save Changes" : "State Saved"}</span>
          </>
        )}
      </button>

    </div>
  );
}

// --- STYLES ---
const styles = {
  scrollWrapper: { height: '100%', width: '100%', overflowY: 'auto', background: '#0e1117', position: 'relative' },
  container: { padding: '30px', paddingBottom: '100px', maxWidth: '1400px', margin: '0 auto', color: '#fff', width: '100%', boxSizing: 'border-box' },
  loadingState: { padding: '40px', color: '#666', textAlign: 'center', fontStyle: 'italic' },
  header: { marginBottom: '20px' },
  title: { margin: 0, display: 'flex', alignItems: 'center', gap: '12px', fontSize: '24px', fontWeight: '700', letterSpacing: '-0.5px' },
  fab: { position: 'fixed', bottom: '30px', right: '40px', height: '48px', borderRadius: '24px', fontSize: '13px', fontWeight: '700', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', zIndex: 1000, boxShadow: '0 4px 12px rgba(0,0,0,0.3)' },
  analysisBox: { marginBottom: '30px', border: '1px solid #333', borderRadius: '8px', background: '#111', overflow: 'hidden' },
  analysisHeader: { padding: '15px 20px', background: '#1a1a1a', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  analysisBody: { padding: '20px', borderTop: '1px solid #333' },
  analysisHint: { margin: '0 0 15px 0', fontSize: '13px', color: '#aaa' },
  fileGrid: { maxHeight: '200px', overflowY: 'auto', border: '1px solid #333', borderRadius: '6px', padding: '10px', background: '#0e0e0e', marginBottom: '15px', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))', gap: '8px' },
  fileItem: { padding: '8px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '8px', transition: 'background 0.2s' },
  analyzeBtn: { width: '100%', padding: '12px', background: '#a855f7', color: 'white', border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', transition: 'background 0.2s' },
  tabContainer: { display: 'flex', gap: '5px', marginBottom: '0', borderBottom: '1px solid #333' },
  tabButton: { padding: '12px 20px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', borderTop: 'none', borderLeft: 'none', borderRight: 'none', transition: 'all 0.2s', outline: 'none' },
  contentArea: { background: '#151515', padding: '25px', borderRadius: '0 0 8px 8px', minHeight: '500px', border: '1px solid #222', borderTop: 'none' }
};