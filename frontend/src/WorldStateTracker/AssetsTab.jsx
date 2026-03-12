import React, { useRef, useEffect, useState, useMemo } from 'react';
import { 
  Trash2, Plus, Coins, Edit2, Link as LinkIcon, 
  Search, Users, Box, Archive 
} from 'lucide-react';
import { EditableTextarea } from '../components/SharedComponents';
import { confirm } from '../components/Notifications';

// --- CONFIGURATION ---
const ICON_OPTIONS = [
  "Resource/Item", "Vehicle/Transport", "Infrastructure/Base", 
  "Organization/Corp", "Weapon", "Technology", "Magic", 
  "Wealth/Economy", "Investment", "Intel/Secrets", 
  "Security/Defense", "Unknown"
];

// --- HELPER COMPONENTS ---

const IconSelector = ({ value, onChange }) => (
  <select 
    value={value || "Resource/Item"} 
    onChange={onChange}
    style={{
      width: '100%', 
      background: '#18181b', 
      color: '#ccc', 
      border: '1px solid #2f2f35', 
      padding: '9px', 
      borderRadius: '6px', 
      outline: 'none', 
      fontSize: '13px', 
      cursor: 'pointer',
      appearance: 'none'
    }}
  >
    {ICON_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
  </select>
);

// --- MAIN COMPONENT ---

export default function AssetsTab({ state, setState }) {
  const assets = state.Assets || [];
  const cast = state.Cast || [];
  
  const [selectedOwnerId, setSelectedOwnerId] = useState("ALL");
  const [searchTerm, setSearchTerm] = useState("");

  // --- DERIVED DATA ---

  const assetCounts = useMemo(() => {
    const counts = { UNASSIGNED: 0, ALL: assets.length };
    cast.forEach(c => counts[c.id] = 0);
    assets.forEach(a => {
      if (a.OwnerId && counts[a.OwnerId] !== undefined) {
        counts[a.OwnerId]++;
      } else {
        counts.UNASSIGNED++;
      }
    });
    return counts;
  }, [assets, cast]);

  const filteredAssets = useMemo(() => {
    let list = assets;
    if (selectedOwnerId === "UNASSIGNED") {
      list = list.filter(a => !a.OwnerId || !cast.find(c => c.id === a.OwnerId));
    } else if (selectedOwnerId !== "ALL") {
      list = list.filter(a => a.OwnerId === selectedOwnerId);
    }
    if (searchTerm) {
      const lower = searchTerm.toLowerCase();
      list = list.filter(a => 
        (a.Asset || "").toLowerCase().includes(lower) || 
        (a.Type || "").toLowerCase().includes(lower)
      );
    }
    return list;
  }, [assets, selectedOwnerId, searchTerm, cast]);

  // --- HANDLERS ---

  const handleUpdate = (index, field, value) => {
    const realAsset = filteredAssets[index]; 
    const realIndex = assets.indexOf(realAsset);
    if (realIndex === -1) return;

    const newList = [...assets];
    newList[realIndex] = { ...newList[realIndex], [field]: value };
    setState({ ...state, Assets: newList });
  };

  const addItem = () => {
    const newOwner = (selectedOwnerId !== "ALL" && selectedOwnerId !== "UNASSIGNED") ? selectedOwnerId : "";
    const defaultItem = { 
      Asset: "New Item", 
      Type: "Financial", 
      Status: "Active", 
      Value: "Unknown", 
      Icon: "Resource/Item", 
      OwnerId: newOwner 
    };
    setState({ ...state, Assets: [...assets, defaultItem] });
  };

  const deleteItem = async (assetToDelete) => {
    const ok = await confirm("Permanently delete this asset?", { title: "Delete Asset", confirmLabel: "Delete", danger: true });
    if (!ok) return;
    setState({ ...state, Assets: assets.filter(a => a !== assetToDelete) });
  };

  const getOwnerName = () => {
    if (selectedOwnerId === "ALL") return "Global Inventory";
    if (selectedOwnerId === "UNASSIGNED") return "Unassigned Resources";
    const char = cast.find(c => c.id === selectedOwnerId);
    return char ? `${char.Name}'s Inventory` : "Unknown Inventory";
  };

  return (
    <div style={styles.container}>
      
      {/* HEADER */}
      <div style={styles.infoBox}>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
          <Coins size={20} style={{ marginTop: '2px', color: '#fbbf24' }} />
          <div>
            <strong style={{ color: '#fff', fontSize: '14px' }}>Strategic Assets & Inventory</strong>
            <p style={{ margin: '4px 0 0', color: '#93c5fd', fontSize: '13px', lineHeight: '1.4' }}>
              Manage physical items, wealth, and locations. Select an owner from the sidebar to filter the view.
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
              <div style={{display:'flex', gap:'10px', alignItems:'center'}}><Box size={14} /> <span>All Assets</span></div>
              <span style={styles.countBadge}>{assetCounts.ALL}</span>
            </div>

            <div 
              onClick={() => setSelectedOwnerId("UNASSIGNED")}
              style={{...styles.filterItem, background: selectedOwnerId === "UNASSIGNED" ? '#3b82f6' : 'transparent', color: selectedOwnerId === "UNASSIGNED" ? '#fff' : '#a1a1aa'}}
            >
              <div style={{display:'flex', gap:'10px', alignItems:'center'}}><Archive size={14} /> <span>Unassigned</span></div>
              <span style={styles.countBadge}>{assetCounts.UNASSIGNED}</span>
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
                {assetCounts[char.id] > 0 && <span style={styles.countBadge}>{assetCounts[char.id]}</span>}
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
                  placeholder="Filter items..." 
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                  style={styles.searchInput}
                />
              </div>
              <button onClick={addItem} style={styles.addBtn}>
                <Plus size={16} /> Add Asset
              </button>
            </div>
          </div>

          {/* Table */}
          <div style={styles.tableWrapper}>
            <table style={styles.table}>
              <thead>
                <tr style={styles.tableHeaderRow}>
                  <th style={{...styles.th, width: '20%'}}>Asset Name</th>
                  {(selectedOwnerId === "ALL" || selectedOwnerId === "UNASSIGNED") && (
                    <th style={{...styles.th, width: '15%'}}>Owner</th>
                  )}
                  <th style={{...styles.th, width: '12%'}}>Type</th>
                  <th style={{...styles.th, width: '10%'}}>Status</th>
                  <th style={{...styles.th, width: '12%'}}>Value</th>
                  <th style={{...styles.th, width: '12%'}}>Icon</th> 
                  <th style={{...styles.th, width: '19%'}}>Notes</th>
                  <th style={{...styles.th, width: '5%'}}></th>
                </tr>
              </thead>
              <tbody>
                {filteredAssets.map((asset, i) => (
                  <tr key={i} style={styles.tableRow}>
                    
                    {/* Name */}
                    <td style={styles.td}>
                      <EditableTextarea 
                        value={asset.Asset || asset.Name} 
                        onChange={(e) => handleUpdate(i, "Asset", e.target.value)}
                        placeholder="Item Name" 
                        style={{ fontWeight: '700', color: '#fff' }}
                      />
                    </td>

                    {/* Owner Dropdown */}
                    {(selectedOwnerId === "ALL" || selectedOwnerId === "UNASSIGNED") && (
                      <td style={styles.td}>
                        <div style={{ position: 'relative' }}>
                          <select 
                            value={asset.OwnerId || ""} 
                            onChange={(e) => handleUpdate(i, "OwnerId", e.target.value)}
                            style={styles.ownerSelect}
                          >
                            <option value="">(Unassigned)</option>
                            {cast.map(c => <option key={c.id} value={c.id}>{c.Name}</option>)}
                          </select>
                          <LinkIcon size={10} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: '#60a5fa', pointerEvents: 'none' }} />
                        </div>
                      </td>
                    )}
                    
                    {/* Metadata */}
                    <td style={styles.td}>
                      <EditableTextarea value={asset.Type} onChange={(e) => handleUpdate(i, "Type", e.target.value)} placeholder="Type" />
                    </td>
                    <td style={styles.td}>
                      <EditableTextarea value={asset.Status} onChange={(e) => handleUpdate(i, "Status", e.target.value)} placeholder="Active" />
                    </td>
                    <td style={styles.td}>
                      <EditableTextarea value={asset.Value} onChange={(e) => handleUpdate(i, "Value", e.target.value)} placeholder="Value" style={{ color: '#fbbf24' }} highlightFocus="#fbbf24" />
                    </td>
                    <td style={styles.td}>
                      <IconSelector value={asset.Icon} onChange={(e) => handleUpdate(i, "Icon", e.target.value)} />
                    </td>
                    
                    {/* Notes: White Text, Normal Font */}
                    <td style={styles.td}>
                      <EditableTextarea 
                        value={asset.Notes || ""} 
                        onChange={(e) => handleUpdate(i, "Notes", e.target.value)} 
                        placeholder="..." 
                        style={{ color: '#ffffff', fontStyle: 'normal' }} 
                      />
                    </td>
                    
                    {/* Delete Button: Red & Visible */}
                    <td style={{...styles.td, textAlign: 'center', verticalAlign: 'middle'}}>
                      <button onClick={() => deleteItem(asset)} style={styles.delBtn} title="Delete Asset">
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
                
                {filteredAssets.length === 0 && (
                  <tr>
                    <td colSpan="8" style={styles.emptyState}>No assets found in this view.</td>
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
    background: 'rgba(59, 130, 246, 0.1)', 
    border: '1px solid rgba(59, 130, 246, 0.2)', 
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
    marginTop: '15px' },
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
    background: '#2563eb', 
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
    minWidth: '900px' 
  },
  tableHeaderRow: { 
    background: '#1a1a1d', 
    textAlign: 'left', 
    color: '#a1a1aa', 
    position: 'sticky', 
    top: 0, zIndex: 10, 
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
    color: '#60a5fa', 
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