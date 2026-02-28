import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import { 
  Plus, Trash2, Archive, CheckCircle2, 
  ChevronDown, ChevronRight, X, AlertCircle,
  Briefcase, Users, Box, Search, Link as LinkIcon
} from 'lucide-react';

const API_URL = "http://localhost:8000";

// --- HELPER COMPONENTS ---

/**
 * Streamlit-Style Slider
 * A custom interactive slider for tracking progress (0-100%).
 */
const StreamlitSlider = ({ value, onChange, color }) => {
  const trackRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  const calculateProgress = (clientX) => {
    if (!trackRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.round((x / rect.width) * 100);
    onChange(percent);
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    calculateProgress(e.clientX);
  };

  useEffect(() => {
    const handleMouseMove = (e) => { if (isDragging) calculateProgress(e.clientX); };
    const handleMouseUp = () => setIsDragging(false);

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  return (
    <div 
      ref={trackRef}
      onMouseDown={handleMouseDown}
      style={{ height: '24px', display: 'flex', alignItems: 'center', cursor: 'pointer', position: 'relative', width: '100%' }}
    >
      <div style={{ width: '100%', height: '4px', background: '#27272a', borderRadius: '2px' }} />
      <div style={{ position: 'absolute', left: 0, height: '4px', borderRadius: '2px', width: `${value}%`, background: color }} />
      <div style={{
        position: 'absolute', left: `${value}%`, width: '14px', height: '14px',
        borderRadius: '50%', background: color, transform: 'translateX(-50%)',
        border: '2px solid #18181b', boxShadow: '0 0 0 1px rgba(0,0,0,0.5)',
        transition: isDragging ? 'none' : 'left 0.1s'
      }} />
    </div>
  );
};

// --- MAIN COMPONENT ---

export default function ProjectsTab({ state, setState, profile }) {
  const projects = state.Projects || [];
  const cast = state.Cast || [];

  const [selectedOwnerId, setSelectedOwnerId] = useState("ALL");
  const [searchTerm, setSearchTerm] = useState("");

  // --- DERIVED DATA ---

  const projectCounts = useMemo(() => {
    const counts = { UNASSIGNED: 0, ALL: projects.length };
    cast.forEach(c => counts[c.id] = 0);
    projects.forEach(p => {
      if (p.OwnerId && counts[p.OwnerId] !== undefined) {
        counts[p.OwnerId]++;
      } else {
        counts.UNASSIGNED++;
      }
    });
    return counts;
  }, [projects, cast]);

  const filteredProjects = useMemo(() => {
    let list = projects;
    
    // Owner Filter
    if (selectedOwnerId === "UNASSIGNED") {
      list = list.filter(p => !p.OwnerId || !cast.find(c => c.id === p.OwnerId));
    } else if (selectedOwnerId !== "ALL") {
      list = list.filter(p => p.OwnerId === selectedOwnerId);
    }

    // Search Filter
    if (searchTerm) {
      const lower = searchTerm.toLowerCase();
      list = list.filter(p => 
        (p.Name || "").toLowerCase().includes(lower) || 
        (p.Description || "").toLowerCase().includes(lower)
      );
    }
    return list;
  }, [projects, selectedOwnerId, searchTerm, cast]);

  // --- HANDLERS ---

  const updateProjects = (newList) => {
    setState({ ...state, Projects: newList });
  };

  const handleAdd = () => {
    // Auto-assign owner if a specific character view is active
    const newOwner = (selectedOwnerId !== "ALL" && selectedOwnerId !== "UNASSIGNED") 
      ? selectedOwnerId 
      : "";

    const newProject = {
      Name: "New Strategy",
      Description: "Define objective...",
      Progress: 0,
      Features_Specs: "",
      OwnerId: newOwner
    };

    updateProjects([...projects, newProject]);
  };

  const handleDelete = (indexInFiltered) => {
    if (!confirm("Permanently delete this project?")) return;
    const projectToDelete = filteredProjects[indexInFiltered];
    const updated = projects.filter(p => p !== projectToDelete);
    updateProjects(updated);
  };

  const handleUpdate = (indexInFiltered, field, value) => {
    const projectToUpdate = filteredProjects[indexInFiltered];
    const realIndex = projects.indexOf(projectToUpdate);
    if (realIndex === -1) return;

    const updated = [...projects];
    updated[realIndex] = { ...updated[realIndex], [field]: value };
    updateProjects(updated);
  };

  // Archive Logic (Requires Backend)
  const handleArchive = async (indexInFiltered, summary, targetCategory) => {
    const proj = filteredProjects[indexInFiltered];
    const entryTitle = `Completed Project: ${proj.Name}`;
    const activeProfile = profile || localStorage.getItem("lastProfile");

    try {
      await axios.post(`${API_URL}/knowledge/create/${activeProfile}`, {
        name: entryTitle,
        content: summary,
        category: targetCategory 
      });
      // Remove after archiving
      const updated = projects.filter(p => p !== proj);
      updateProjects(updated);
      alert(`✅ Project archived to ${targetCategory}!`);
    } catch (err) {
      alert("Archive failed: " + err.message);
    }
  };

  const getOwnerName = () => {
    if (selectedOwnerId === "ALL") return "All Active Projects";
    if (selectedOwnerId === "UNASSIGNED") return "Unassigned / Global Goals";
    const char = cast.find(c => c.id === selectedOwnerId);
    return char ? `${char.Name}'s Agenda` : "Unknown Agenda";
  };

  return (
    <div style={styles.container}>
      
      {/* HEADER INFO */}
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <Briefcase size={20} style={{ marginTop: '2px', color: '#60a5fa' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Long-Term Goals & Projects</strong>
            <p style={{ margin: '4px 0 0', color: '#93c5fd', fontSize: '13px', lineHeight: '1.4' }}>
              Track sustained efforts like base-building, research, or political schemes. 
              Assign projects to characters to define their personal agendas.
            </p>
          </div>
        </div>
      </div>

      <div style={styles.layout}>
        
        {/* --- SIDEBAR (FILTERS) --- */}
        <div style={styles.sidebar}>
          <div style={styles.sidebarHeader}>FILTERS</div>
          <div style={styles.filterList}>
            <div 
              onClick={() => setSelectedOwnerId("ALL")}
              style={{...styles.filterItem, background: selectedOwnerId === "ALL" ? '#3b82f6' : 'transparent', color: selectedOwnerId === "ALL" ? '#fff' : '#a1a1aa'}}
            >
              <div style={{display:'flex', gap:'10px', alignItems:'center'}}><Box size={14} /> <span>All Projects</span></div>
              <span style={styles.countBadge}>{projectCounts.ALL}</span>
            </div>

            <div 
              onClick={() => setSelectedOwnerId("UNASSIGNED")}
              style={{...styles.filterItem, background: selectedOwnerId === "UNASSIGNED" ? '#3b82f6' : 'transparent', color: selectedOwnerId === "UNASSIGNED" ? '#fff' : '#a1a1aa'}}
            >
              <div style={{display:'flex', gap:'10px', alignItems:'center'}}><Archive size={14} /> <span>Unassigned</span></div>
              <span style={styles.countBadge}>{projectCounts.UNASSIGNED}</span>
            </div>

            <div style={styles.divider} />
            <div style={styles.sidebarHeader}>BY OWNER</div>

            {cast.map(char => (
              <div 
                key={char.id}
                onClick={() => setSelectedOwnerId(char.id)}
                style={{...styles.filterItem, background: selectedOwnerId === char.id ? '#3b82f6' : 'transparent', color: selectedOwnerId === char.id ? '#fff' : '#a1a1aa'}}
              >
                <div style={{display:'flex', gap:'10px', alignItems:'center', overflow:'hidden'}}>
                  <Users size={14} /> 
                  <span style={{whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{char.Name}</span>
                </div>
                {projectCounts[char.id] > 0 && <span style={styles.countBadge}>{projectCounts[char.id]}</span>}
              </div>
            ))}
          </div>
        </div>

        {/* --- MAIN CONTENT --- */}
        <div style={styles.mainContent}>
          
          {/* Toolbar */}
          <div style={styles.toolbar}>
            <div style={styles.viewTitle}>{getOwnerName()}</div>
            <div style={{display:'flex', gap:'12px', alignItems:'center'}}>
              <div style={styles.searchWrapper}>
                <Search size={14} color="#777" />
                <input 
                  placeholder="Filter projects..." 
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                  style={styles.searchInput}
                />
              </div>
              <button onClick={handleAdd} style={styles.addBtn}>
                <Plus size={16} /> New Project
              </button>
            </div>
          </div>

          {/* Projects List */}
          <div style={styles.listWrapper}>
            {filteredProjects.length === 0 && (
              <div style={styles.emptyState}>
                <AlertCircle size={24} style={{ marginBottom: '10px', opacity: 0.5 }} />
                <p>No active projects in this view.</p>
              </div>
            )}

            {filteredProjects.map((proj, i) => (
              <ProjectCard 
                key={i} 
                index={i} 
                project={proj} 
                cast={cast}
                showOwnerDropdown={selectedOwnerId === "ALL" || selectedOwnerId === "UNASSIGNED"}
                onChange={handleUpdate} 
                onDelete={handleDelete}
                onArchive={handleArchive}
              />
            ))}
          </div>

        </div>
      </div>
    </div>
  );
}

// --- PROJECT CARD COMPONENT ---

const ProjectCard = ({ index, project, cast, showOwnerDropdown, onChange, onDelete, onArchive }) => {
  const [expanded, setExpanded] = useState(false);
  const [showArchive, setShowArchive] = useState(false);
  const [archiveNote, setArchiveNote] = useState("");
  const [archiveTarget, setArchiveTarget] = useState("Lore");

  const progressColor = project.Progress >= 100 ? '#22c55e' : (project.Progress >= 50 ? '#3b82f6' : '#f59e0b');

  const openArchive = () => {
    setArchiveNote(`Final Status: ${project.Progress}% Complete.\n\nSummary:\n${project.Features_Specs || "No technical specs recorded."}`);
    setShowArchive(true);
  };

  return (
    <div style={{...styles.card, borderLeft: `4px solid ${progressColor}`}}>
      
      {/* Collapsed Header */}
      <div 
        onClick={() => setExpanded(!expanded)}
        style={styles.cardHeader}
      >
        <div style={{display:'flex', alignItems:'center', gap:'15px', flex: 1}}>
          <div style={styles.iconBtn}>
            {expanded ? <ChevronDown size={20} color="#fff" /> : <ChevronRight size={20} color="#666" />}
          </div>
          <div style={{flex: 1}}>
            <div style={styles.projectTitle}>
              {project.Name || "Untitled Project"}
            </div>
            <div style={styles.projectDesc}>
              {project.Description || "No objective defined."}
            </div>
          </div>
        </div>

        {/* Progress Mini View */}
        <div style={{ width: '120px', textAlign: 'right', marginRight: '10px' }}>
          <div style={{ fontSize: '11px', fontWeight: 'bold', color: progressColor, marginBottom: '4px' }}>
            {project.Progress}% DONE
          </div>
          <div style={{ height: '4px', background: '#27272a', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ width: `${project.Progress}%`, height: '100%', background: progressColor }} />
          </div>
        </div>
      </div>

      {/* Expanded Editor */}
      {expanded && (
        <div style={styles.expandedPanel}>
          <div style={styles.editGrid}>
            
            {/* LEFT: Metadata */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              <div>
                <label style={styles.label}>PROJECT NAME</label>
                <input 
                  value={project.Name} 
                  onChange={(e) => onChange(index, "Name", e.target.value)}
                  style={styles.input}
                />
              </div>

              {/* Owner Dropdown (Conditional) */}
              {showOwnerDropdown && (
                <div>
                  <label style={styles.label}>PROJECT LEAD / OWNER</label>
                  <div style={{ position: 'relative' }}>
                    <select 
                      value={project.OwnerId || ""} 
                      onChange={(e) => onChange(index, "OwnerId", e.target.value)}
                      style={styles.ownerSelect}
                    >
                      <option value="" style={{background:'#18181b'}}>(Unassigned)</option>
                      {cast.map(c => <option key={c.id} value={c.id} style={{background:'#18181b'}}>{c.Name}</option>)}
                    </select>
                    <LinkIcon size={12} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#60a5fa', pointerEvents: 'none' }} />
                  </div>
                </div>
              )}

              <div>
                <label style={styles.label}>OBJECTIVE / GOAL</label>
                <textarea 
                  value={project.Description} 
                  onChange={(e) => onChange(index, "Description", e.target.value)}
                  style={{ ...styles.textArea, minHeight: '60px' }} 
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <label style={styles.label}>PROGRESS</label>
                  <span style={{ fontSize: '12px', color: progressColor, fontWeight: 'bold' }}>{project.Progress}%</span>
                </div>
                <StreamlitSlider 
                  value={project.Progress} 
                  onChange={(val) => onChange(index, "Progress", val)}
                  color={progressColor}
                />
              </div>
            </div>

            {/* RIGHT: Specs */}
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <label style={styles.label}>TECHNICAL SPECS / LOG</label>
              <textarea 
                value={project.Features_Specs || ""} 
                onChange={(e) => onChange(index, "Features_Specs", e.target.value)}
                style={{ ...styles.textArea, height: '100%', minHeight: '180px' }}
                placeholder="Milestones, resources required, or technical notes..."
              />
            </div>
          </div>

          <div style={styles.actionBar}>
            <button onClick={() => onDelete(index)} style={styles.deleteBtn}>
              <Trash2 size={14} /> Delete
            </button>
            <button onClick={openArchive} style={styles.archiveBtn}>
              <CheckCircle2 size={14} /> Complete & Archive
            </button>
          </div>

          {/* Archive Modal */}
          {showArchive && (
            <div style={styles.archivePanel}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                <h4 style={{ margin: 0, color: '#22c55e', fontSize: '13px' }}>ARCHIVE TO KNOWLEDGE BASE</h4>
                <X size={16} onClick={() => setShowArchive(false)} style={{cursor:'pointer'}} />
              </div>
              <div style={{display:'flex', gap:'15px', marginBottom:'10px'}}>
                <label style={styles.radioLabel}>
                  <input type="radio" checked={archiveTarget === "Lore"} onChange={() => setArchiveTarget("Lore")} /> Lore
                </label>
                <label style={styles.radioLabel}>
                  <input type="radio" checked={archiveTarget === "Fact"} onChange={() => setArchiveTarget("Fact")} /> Fact
                </label>
              </div>
              <textarea 
                value={archiveNote} onChange={(e) => setArchiveNote(e.target.value)}
                style={{ ...styles.textArea, height: '80px', marginBottom: '10px' }}
              />
              <button 
                onClick={() => onArchive(index, archiveNote, archiveTarget)}
                style={{ ...styles.addBtn, width: '100%', background: '#22c55e' }}
              >
                Confirm Archive
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// --- STYLES ---

const styles = {
  container: { padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' },
  infoBox: { background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)', padding: '16px', borderRadius: '8px', marginBottom: '20px' },
  
  layout: { display: 'flex', gap: '20px', flex: 1, minHeight: 0 },
  sidebar: { width: '240px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #27272a', paddingRight: '15px' },
  mainContent: { flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 },

  // Sidebar
  sidebarHeader: { fontSize: '10px', fontWeight: 'bold', color: '#555', marginBottom: '8px', marginTop: '15px' },
  filterList: { display: 'flex', flexDirection: 'column', gap: '4px', overflowY: 'auto' },
  filterItem: { padding: '10px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', transition: 'all 0.2s', fontWeight: '500' },
  countBadge: { fontSize: '10px', background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '10px', color: '#fff' },
  divider: { height: '1px', background: '#333', margin: '15px 0' },

  // Toolbar
  toolbar: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px', background: '#131315', padding: '12px 16px', borderRadius: '8px', border: '1px solid #222' },
  viewTitle: { fontSize: '16px', fontWeight: '700', color: '#fff' },
  searchWrapper: { display: 'flex', alignItems: 'center', background: '#09090b', border: '1px solid #333', borderRadius: '6px', padding: '8px 12px', gap: '8px' },
  searchInput: { background: 'transparent', border: 'none', color: '#fff', outline: 'none', fontSize: '13px', width: '160px' },
  addBtn: { background: '#2563eb', color: '#fff', border: 'none', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: '600' },

  // Cards
  listWrapper: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px' },
  card: { background: '#18181b', border: '1px solid #27272a', borderRadius: '8px', marginBottom: '5px', transition: 'all 0.2s' },
  cardHeader: { padding: '15px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' },
  projectTitle: { fontSize: '15px', fontWeight: '700', color: '#fff', marginBottom: '2px' },
  projectDesc: { fontSize: '13px', color: '#888', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  iconBtn: { display: 'flex', alignItems: 'center', justifyContent: 'center', width: '24px' },
  
  // Expanded Editor
  expandedPanel: { borderTop: '1px solid #27272a', padding: '20px', background: '#131315', borderBottomLeftRadius: '8px', borderBottomRightRadius: '8px' },
  editGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '20px' },
  label: { fontSize: '11px', color: '#666', fontWeight: '700', display: 'block', marginBottom: '6px', letterSpacing: '0.5px' },
  input: { width: '100%', padding: '10px', background: '#09090b', border: '1px solid #333', color: '#fff', borderRadius: '6px', outline: 'none', fontSize: '13px', boxSizing: 'border-box' },
  textArea: { width: '100%', padding: '10px', background: '#09090b', border: '1px solid #333', color: '#ccc', borderRadius: '6px', resize: 'vertical', fontSize: '13px', lineHeight: '1.5', boxSizing: 'border-box', fontFamily: 'monospace' },
  ownerSelect: { width: '100%', padding: '10px 10px 10px 30px', background: '#09090b', border: '1px solid #333', borderRadius: '6px', color: '#60a5fa', fontSize: '13px', outline: 'none', cursor: 'pointer', appearance: 'none' },
  
  // Actions
  actionBar: { display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '10px' },
  deleteBtn: { background: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' },
  archiveBtn: { background: '#22c55e', color: '#fff', border: 'none', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '6px' },
  
  // Archive Modal
  archivePanel: { marginTop: '15px', background: '#000', border: '1px solid #22c55e', borderRadius: '8px', padding: '15px' },
  radioLabel: { fontSize: '13px', color: '#ccc', display: 'flex', gap: '6px', alignItems: 'center', cursor: 'pointer' },
  emptyState: { padding: '40px', textAlign: 'center', color: '#555', fontStyle: 'italic' }
};