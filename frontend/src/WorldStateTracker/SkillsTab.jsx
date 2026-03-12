import React, { useRef, useEffect, useState, useMemo } from 'react';
import { 
  Trash2, Plus, Zap, Edit2, Link as LinkIcon, 
  Search, Users, Box, Archive 
} from 'lucide-react';
import { EditableTextarea } from '../components/SharedComponents';
import { confirm } from '../components/Notifications';

// --- MAIN COMPONENT ---

export default function SkillsTab({ state, setState }) {
  const skills = state.Skills || [];
  const cast = state.Cast || [];
  
  const [selectedOwnerId, setSelectedOwnerId] = useState("ALL");
  const [searchTerm, setSearchTerm] = useState("");

  // --- DERIVED DATA ---

  const skillCounts = useMemo(() => {
    const counts = { UNASSIGNED: 0, ALL: skills.length };
    cast.forEach(c => counts[c.id] = 0);
    skills.forEach(s => {
      if (s.OwnerId && counts[s.OwnerId] !== undefined) {
        counts[s.OwnerId]++;
      } else {
        counts.UNASSIGNED++;
      }
    });
    return counts;
  }, [skills, cast]);

  const filteredSkills = useMemo(() => {
    let list = skills;
    // Owner Filter
    if (selectedOwnerId === "UNASSIGNED") {
      list = list.filter(s => !s.OwnerId || !cast.find(c => c.id === s.OwnerId));
    } else if (selectedOwnerId !== "ALL") {
      list = list.filter(s => s.OwnerId === selectedOwnerId);
    }
    // Search Filter
    if (searchTerm) {
      const lower = searchTerm.toLowerCase();
      list = list.filter(s => 
        (s.Skill || "").toLowerCase().includes(lower) || 
        (s.Description || "").toLowerCase().includes(lower)
      );
    }
    return list;
  }, [skills, selectedOwnerId, searchTerm, cast]);

  // --- HANDLERS ---

  const handleUpdate = (index, field, value) => {
    const realSkill = filteredSkills[index]; 
    const realIndex = skills.indexOf(realSkill);
    if (realIndex === -1) return;

    const newList = [...skills];
    newList[realIndex] = { ...newList[realIndex], [field]: value };
    setState({ ...state, Skills: newList });
  };

  const addSkill = () => {
    const newOwner = (selectedOwnerId !== "ALL" && selectedOwnerId !== "UNASSIGNED") ? selectedOwnerId : "";
    const newSkill = { 
      Skill: "New Ability", 
      Description: "Describe mechanics or permissions...", 
      OwnerId: newOwner 
    };
    setState({ ...state, Skills: [...skills, newSkill] });
  };

  const deleteSkill = async (skillToDelete) => {
    const ok = await confirm("Delete this skill?", { title: "Delete Skill", confirmLabel: "Delete", danger: true });
    if (!ok) return;
    setState({ ...state, Skills: skills.filter(s => s !== skillToDelete) });
  };

  const getOwnerName = () => {
    if (selectedOwnerId === "ALL") return "Global Skillset";
    if (selectedOwnerId === "UNASSIGNED") return "Unassigned Abilities";
    const char = cast.find(c => c.id === selectedOwnerId);
    return char ? `${char.Name}'s Skills` : "Unknown Skillset";
  };

  return (
    <div style={styles.container}>
      
      {/* HEADER */}
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <Zap size={20} style={{ marginTop: '2px', color: '#facc15' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Competence & Ability Modifiers</strong>
            <p style={{ margin: '4px 0 0', color: '#fef08a', fontSize: '13px', lineHeight: '1.4' }}>
              Define narrative permissions. If a character has <b>'Hacking'</b>, the AI unlocks cyber-warfare options. Assign skills to specific characters to track competence.
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
              style={{...styles.filterItem, background: selectedOwnerId === "ALL" ? '#ca8a04' : 'transparent', color: selectedOwnerId === "ALL" ? '#fff' : '#a1a1aa'}}
            >
              <div style={{display:'flex', gap:'10px', alignItems:'center'}}><Box size={14} /> <span>All Skills</span></div>
              <span style={styles.countBadge}>{skillCounts.ALL}</span>
            </div>

            <div 
              onClick={() => setSelectedOwnerId("UNASSIGNED")}
              style={{...styles.filterItem, background: selectedOwnerId === "UNASSIGNED" ? '#ca8a04' : 'transparent', color: selectedOwnerId === "UNASSIGNED" ? '#fff' : '#a1a1aa'}}
            >
              <div style={{display:'flex', gap:'10px', alignItems:'center'}}><Archive size={14} /> <span>Unassigned</span></div>
              <span style={styles.countBadge}>{skillCounts.UNASSIGNED}</span>
            </div>

            <div style={styles.divider} />
            <div style={styles.sidebarHeader}>BY OWNER</div>

            {cast.map(char => (
              <div 
                key={char.id}
                onClick={() => setSelectedOwnerId(char.id)}
                style={{...styles.filterItem, background: selectedOwnerId === char.id ? '#ca8a04' : 'transparent', color: selectedOwnerId === char.id ? '#fff' : '#a1a1aa'}}
              >
                <div style={{display:'flex', gap:'10px', alignItems:'center', overflow:'hidden'}}>
                  <Users size={14} /> 
                  <span style={{whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{char.Name}</span>
                </div>
                {skillCounts[char.id] > 0 && <span style={styles.countBadge}>{skillCounts[char.id]}</span>}
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
                  placeholder="Filter abilities..." 
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                  style={styles.searchInput}
                />
              </div>
              <button onClick={addSkill} style={styles.addBtn}>
                <Plus size={16} /> Add Skill
              </button>
            </div>
          </div>

          {/* Table */}
          <div style={styles.tableWrapper}>
            <table style={styles.table}>
              <thead>
                <tr style={styles.tableHeaderRow}>
                  <th style={{...styles.th, width: '25%'}}>Skill Name</th>
                  {(selectedOwnerId === "ALL" || selectedOwnerId === "UNASSIGNED") && (
                    <th style={{...styles.th, width: '20%'}}>Owner</th>
                  )}
                  <th style={{...styles.th, width: '50%'}}>Mechanics / Description</th>
                  <th style={{...styles.th, width: '5%'}}></th>
                </tr>
              </thead>
              <tbody>
                {filteredSkills.map((skill, i) => (
                  <tr key={i} style={styles.tableRow}>
                    
                    {/* Skill Name */}
                    <td style={styles.td}>
                      <EditableTextarea 
                        value={skill.Skill} 
                        onChange={(e) => handleUpdate(i, "Skill", e.target.value)}
                        placeholder="e.g. Marksmanship" 
                        style={{ fontWeight: '700', color: '#facc15' }} // Amber text
                        highlightFocus="#facc15"
                      />
                    </td>

                    {/* Owner Dropdown */}
                    {(selectedOwnerId === "ALL" || selectedOwnerId === "UNASSIGNED") && (
                      <td style={styles.td}>
                        <div style={{ position: 'relative' }}>
                          <select 
                            value={skill.OwnerId || ""} 
                            onChange={(e) => handleUpdate(i, "OwnerId", e.target.value)}
                            style={styles.ownerSelect}
                          >
                            <option value="">(Unassigned)</option>
                            {cast.map(c => <option key={c.id} value={c.id}>{c.Name}</option>)}
                          </select>
                          <LinkIcon size={10} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#eab308', pointerEvents: 'none' }} />
                        </div>
                      </td>
                    )}
                    
                    {/* Description */}
                    <td style={styles.td}>
                      <EditableTextarea 
                        value={skill.Description} 
                        onChange={(e) => handleUpdate(i, "Description", e.target.value)} 
                        placeholder="Define mechanics..." 
                        style={{ color: '#ccc' }} 
                        highlightFocus="#eab308"
                      />
                    </td>
                    
                    {/* Delete Button */}
                    <td style={{...styles.td, textAlign: 'center', verticalAlign: 'middle'}}>
                      <button onClick={() => deleteSkill(skill)} style={styles.delBtn} title="Delete Skill">
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
                
                {filteredSkills.length === 0 && (
                  <tr>
                    <td colSpan="8" style={styles.emptyState}>No skills found in this view.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

        </div>
      </div>
    </div>
  );
}

// --- STYLES ---

const styles = {
  container: { 
    padding: '10px', 
    height: '100%', 
    display: 'flex', 
    flexDirection: 'column' 
  },
  infoBox: { 
    background: 'rgba(234, 179, 8, 0.1)', 
    border: '1px solid rgba(234, 179, 8, 0.2)', 
    padding: '16px', 
    borderRadius: '8px', 
    marginBottom: '20px' 
  },
  
  layout: { 
    display: 'flex', 
    gap: '20px', 
    flex: 1, 
    minHeight: 0 
  },
  sidebar: { 
    width: '240px', 
    display: 'flex', 
    flexDirection: 'column', 
    borderRight: '1px solid #27272a', 
    paddingRight: '15px' 
  },
  mainContent: { 
    flex: 1, 
    display: 'flex', 
    flexDirection: 'column', 
    minWidth: 0 
  },

  sidebarHeader: { 
    fontSize: '10px', 
    fontWeight: 'bold', 
    color: '#555', 
    marginBottom: '8px', 
    marginTop: '15px' 
  },
  filterList: { 
    display: 'flex', 
    flexDirection: 'column', 
    gap: '4px', 
    overflowY: 'auto' 
  },
  filterItem: { 
    padding: '10px 12px', 
    borderRadius: '6px', 
    cursor: 'pointer', 
    fontSize: '13px', 
    display: 'flex', 
    justifyContent: 'space-between', 
    alignItems: 'center', 
    transition: 'all 0.2s', 
    fontWeight: '500' 
  },
  countBadge: { 
    fontSize: '10px', 
    background: 'rgba(255,255,255,0.1)', 
    padding: '2px 6px', 
    borderRadius: '10px', 
    color: '#fff' 
  },
  divider: { 
    height: '1px', 
    background: '#333', 
    margin: '15px 0' 
  },

  toolbar: { 
    display: 'flex', 
    justifyContent: 'space-between', 
    alignItems: 'center', 
    marginBottom: '15px', 
    background: '#131315', 
    padding: '12px 16px', 
    borderRadius: '8px', 
    border: '1px solid #222' 
  },
  viewTitle: { 
    fontSize: '16px', 
    fontWeight: '700', 
    color: '#fff' 
  },
  searchWrapper: { 
    display: 'flex', 
    alignItems: 'center', 
    background: '#09090b', 
    border: '1px solid #333', 
    borderRadius: '6px', 
    padding: '8px 12px', 
    gap: '8px' 
  },
  searchInput: { 
    background: 'transparent', 
    border: 'none', 
    color: '#fff', 
    outline: 'none', 
    fontSize: '13px', 
    width: '160px' 
  },
  addBtn: { 
    background: '#ca8a04', 
    color: '#fff', 
    border: 'none', 
    padding: '8px 16px', 
    borderRadius: '6px', 
    cursor: 'pointer', 
    display: 'flex', 
    alignItems: 'center', 
    gap: '8px', 
    fontSize: '13px', 
    fontWeight: '600' 
  },

  tableWrapper: { 
    flex: 1, 
    overflow: 'auto', 
    border: '1px solid #27272a', 
    borderRadius: '8px', 
    background: '#111' 
  },
  table: { 
    width: '100%', 
    borderCollapse: 'collapse', 
    fontSize: '14px', 
    minWidth: '700px' 
  },
  tableHeaderRow: { 
    background: '#1a1a1d', 
    textAlign: 'left', 
    color: '#a1a1aa', 
    position: 'sticky', 
    top: 0, 
    zIndex: 10, 
    boxShadow: '0 2px 5px rgba(0,0,0,0.2)' 
  },
  tableRow: { 
    borderBottom: '1px solid #222', 
    background: '#0e0e0e' 
  },
  th: { 
    padding: '14px 12px', 
    borderBottom: '1px solid #333', 
    fontSize: '11px', 
    fontWeight: '700', 
    textTransform: 'uppercase', 
    letterSpacing: '0.5px' 
  },
  td: { 
    padding: '10px 12px', 
    verticalAlign: 'top' 
  },
  
  ownerSelect: { 
    width: '100%', 
    padding: '9px 8px 9px 30px', 
    background: '#18181b',
    border: '1px solid #3f3f46', 
    borderRadius: '6px', 
    color: '#eab308',
    fontSize: '12px', 
    fontWeight: '500',
    outline: 'none', 
    cursor: 'pointer', 
    appearance: 'none' 
  },
  delBtn: { 
    background: 'rgba(239, 68, 68, 0.15)',
    color: '#ef4444', 
    border: '1px solid rgba(239, 68, 68, 0.3)', 
    borderRadius: '4px',
    cursor: 'pointer', 
    padding: '8px', 
    display: 'flex', 
    alignItems: 'center', 
    justifyContent: 'center',
    transition: 'all 0.2s' 
  },
  emptyState: { 
    padding: '60px', 
    textAlign: 'center', 
    color: '#555', 
    fontStyle: 'italic' 
  }
};