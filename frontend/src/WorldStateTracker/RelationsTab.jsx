import React, { useRef, useEffect, useMemo } from 'react';
import { Users, Shield, Swords, Edit2, Link as LinkIcon } from 'lucide-react';

const getIconPath = (iconKey) => {
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

const EditableTextarea = ({ value, onChange, placeholder, style }) => {
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
    <div style={{ position: 'relative', width: '100%' }}>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => { onChange(e); adjustHeight(); }}
        placeholder={placeholder}
        rows={1}
        style={{
          width: '100%', background: 'rgba(0, 0, 0, 0.2)', border: '1px solid #3f3f46',      
          borderRadius: '4px', color: '#eee', outline: 'none', fontFamily: 'inherit', 
          resize: 'none', overflow: 'hidden', minHeight: '32px', lineHeight: '1.5', 
          display: 'block', padding: '6px 8px', fontSize: '13px', transition: 'border-color 0.2s', ...style
        }}
        onFocus={(e) => e.target.style.borderColor = '#60a5fa'} 
        onBlur={(e) => e.target.style.borderColor = '#3f3f46'}
      />
      {!value && <Edit2 size={10} color="#555" style={{ position: 'absolute', right: '8px', top: '10px', pointerEvents: 'none' }} />}
    </div>
  );
};

export default function RelationsTab({ state, setState }) {
  const cast = state.Cast || [];
  
  // Find the POV character ID for linking logic
  const mainPov = useMemo(() => cast.find(c => c.Role === "POV"), [cast]);
  
  const castMap = useMemo(() => {
    return cast.reduce((acc, char) => {
      acc[char.id] = char.Name;
      return acc;
    }, {});
  }, [cast]);

  const relations = cast.filter(c => c.Role !== "POV");

  const updateMember = (id, field, value) => {
    const updatedCast = cast.map(c => {
      if (c.id === id) {
        const updatedChar = { ...c, [field]: value };
        
        // --- SYNC LOGIC: Update the Network Map Link automatically ---
        if (field === "Relation" && mainPov) {
          const currentLinks = c.Links || [];
          // Remove old link to POV
          const otherLinks = currentLinks.filter(l => l.targetId !== mainPov.id);
          // Add new link if value is not empty
          if (value.trim()) {
            updatedChar.Links = [...otherLinks, { targetId: mainPov.id, type: value }];
          } else {
            updatedChar.Links = otherLinks;
          }
        }
        
        return updatedChar;
      }
      return c;
    });
    setState({ ...state, Cast: updatedCast });
  };

  return (
    <div style={styles.container}>
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <Users size={20} style={{ marginTop: '2px' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Relationship Matrix</strong>
            <p style={{ margin: '4px 0 0', color: '#93c5fd', fontSize: '13px', lineHeight: '1.4' }}>
              Define how the supporting cast relates to the active POV. <br/>
              <span style={{ color: '#a1a1aa' }}>* Edits here automatically update the Network Map connections.</span>
            </p>
          </div>
        </div>
      </div>

      <table style={styles.table}>
        <thead>
          <tr style={styles.tableHeaderRow}>
            <th style={{...styles.th, width: '5%'}}></th>
            <th style={{...styles.th, width: '20%'}}>Name / Context</th>
            <th style={{...styles.th, width: '25%'}}>Relation To POV</th>
            <th style={{...styles.th, width: '10%'}}>Loyalty %</th>
            <th style={{...styles.th, width: '40%'}}>Private Notes</th>
          </tr>
        </thead>
        <tbody>
          {relations.map((char) => (
            <tr key={char.id} style={styles.tableRow}>
              <td style={{...styles.td, textAlign: 'center'}}>
                <img src={getIconPath(char.Icon)} alt="avatar" style={{ width: '32px', height: '32px', opacity: 0.9, objectFit: 'contain' }} />
              </td>
              <td style={styles.td}>
                <div style={{ fontWeight: '700', color: '#fff', fontSize: '14px', marginBottom: '4px' }}>{char.Name}</div>
                <div style={{ fontSize: '11px', color: '#a1a1aa', display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '6px' }}>
                  {char.Role === 'Antagonist' ? <Swords size={10} color="#ef4444"/> : <Shield size={10} color="#22c55e"/>}
                  {char.Role}
                </div>
                {char.Orbit && castMap[char.Orbit] && (
                  <div style={styles.orbitBadge}><LinkIcon size={8} /> Orbit: {castMap[char.Orbit]}</div>
                )}
              </td>
              <td style={styles.td}>
                <EditableTextarea 
                  value={char.Relation || ""} 
                  onChange={(e) => updateMember(char.id, "Relation", e.target.value)} 
                  placeholder="e.g. Brother, Rival"
                  style={{ color: '#60a5fa', fontWeight: '500' }}
                />
              </td>
              <td style={styles.td}>
                <input 
                  type="number" 
                  value={char.Loyalty || 0} 
                  onChange={(e) => updateMember(char.id, "Loyalty", parseInt(e.target.value))} 
                  style={{
                    ...styles.numberInput,
                    color: (char.Loyalty < 0) ? '#ef4444' : (char.Loyalty > 75 ? '#22c55e' : '#fff'),
                    borderColor: (char.Loyalty < 0) ? '#7f1d1d' : '#3f3f46'
                  }}
                  className="no-spinner" 
                  placeholder="0"
                />
              </td>
              <td style={styles.td}>
                <EditableTextarea 
                  value={char.Notes || ""} 
                  onChange={(e) => updateMember(char.id, "Notes", e.target.value)} 
                  placeholder="Add context..."
                  style={{ color: '#ffffff', fontStyle: 'normal' }} 
                />
              </td>
            </tr>
          ))}
          {relations.length === 0 && (
            <tr><td colSpan="5" style={styles.emptyState}>No supporting cast found.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

const styles = {
  container: { padding: '10px' },
  infoBox: { background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)', padding: '16px', borderRadius: '8px', marginBottom: '25px' },
  numberInput: { background: 'rgba(0,0,0,0.2)', border: '1px solid #3f3f46', borderRadius: '4px', color: '#fff', width: '60px', margin: '0 auto', outline: 'none', padding: '6px', textAlign: 'center', fontSize: '13px', display: 'block' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: '14px', tableLayout: 'fixed' },
  tableHeaderRow: { background: '#27272a', textAlign: 'left', color: '#a1a1aa' },
  tableRow: { borderBottom: '1px solid #27272a', transition: 'background 0.1s' },
  th: { padding: '12px 15px', borderBottom: '1px solid #3f3f46', verticalAlign: 'middle', fontSize: '12px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' },
  td: { padding: '12px 15px', borderRight: '1px solid #27272a', verticalAlign: 'top' },
  emptyState: { padding: '30px', textAlign: 'center', color: '#555', fontStyle: 'italic', borderBottom: '1px solid #27272a' },
  orbitBadge: { fontSize: '10px', color: '#60a5fa', background: 'rgba(59, 130, 246, 0.1)', padding: '2px 6px', borderRadius: '4px', display: 'inline-flex', alignItems: 'center', gap: '4px', border: '1px solid rgba(59, 130, 246, 0.2)', whiteSpace: 'nowrap' }
};