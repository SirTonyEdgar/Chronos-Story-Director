import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Book, Shield, Map, FileText, EyeOff, RefreshCw, CheckCircle2, Users, Network } from 'lucide-react';
import { API_URL } from '../config';
import { toast, confirm } from '../components/Notifications';

// Import Sub-Tabs
import LoreTab from './LoreTab';
import CharactersTab from './CharactersTab';
import FactionTab from './FactionTab';
import RulesTab from './RulesTab';
import PlansTab from './PlansTab';
import FactsTab from './FactsTab';
import SpoilersTab from './SpoilersTab';

export default function KnowledgeBase({ profile }) {
  
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem("chronos_kb_tab") || "Lore";
  });

  const [isRemetadata, setIsRemetadata] = useState(false);
  const [remetadataResult, setRemetadataResult] = useState(null);

  useEffect(() => {
    localStorage.setItem("chronos_kb_tab", activeTab);
  }, [activeTab]);

  const handleBulkRemetadata = async () => {
    const ok = await confirm(
      "This will regenerate Librarian metadata for every document in your Knowledge Base using the current prompt and 32,000 character cap.\n\nAny manual metadata edits you have made will be overwritten.\n\nThis may take several minutes depending on how many documents you have.",
      { title: "Regenerate All Metadata", confirmLabel: "Regenerate", danger: false }
    );
    if (!ok) return;

    setIsRemetadata(true);
    setRemetadataResult(null);
    try {
      const res = await axios.post(`${API_URL}/knowledge/remetadata/${profile}`);
      setRemetadataResult(res.data);
      toast(`Metadata regenerated. ${res.data.success} updated, ${res.data.skipped} skipped, ${res.data.failed} failed.`, "success");
    } catch (err) {
      toast("Bulk re-metadata failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setIsRemetadata(false);
    }
  };

  const tabs = [
    { id: "Lore", icon: <Book size={16} />, label: "Lore", color: "#3b82f6" },
    { id: "Character", icon: <Users size={16} />, label: "Characters", color: "#f97316" },
    { id: "Faction", icon: <Network size={16} />, label: "Factions", color: "#f59e0b" },
    { id: "Rules", icon: <Shield size={16} />, label: "Rules", color: "#ef4444" },
    { id: "Plans", icon: <Map size={16} />, label: "Plans", color: "#f59e0b" },
    { id: "Facts", icon: <FileText size={16} />, label: "Facts", color: "#10b981" },
    { id: "Spoilers", icon: <EyeOff size={16} />, label: "Spoilers", color: "#8b5cf6" },
  ];

  return (
    <div style={styles.container}>
      
      {/* --- HEADER --- */}
      <div style={styles.header}>
        <h2 style={styles.title}>🗄️ Knowledge Base</h2>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>

          {/* Result Badge */}
          {remetadataResult && (
            <div style={styles.resultBadge}>
              <CheckCircle2 size={13} color="#22c55e" />
              <span>{remetadataResult.success} updated · {remetadataResult.skipped} skipped · {remetadataResult.failed} failed</span>
            </div>
          )}

          {/* Bulk Re-metadata Button */}
          <button
            onClick={handleBulkRemetadata}
            disabled={isRemetadata}
            style={{
              ...styles.remetadataBtn,
              opacity: isRemetadata ? 0.6 : 1,
              cursor: isRemetadata ? 'default' : 'pointer'
            }}
            title="Regenerate Librarian metadata for all documents using the current prompt and 32,000 character cap"
          >
            <RefreshCw size={14} style={{ animation: isRemetadata ? 'spin 1s linear infinite' : 'none' }} />
            {isRemetadata ? "Regenerating..." : "Re-generate All Metadata"}
          </button>
        </div>
      </div>

      {/* --- TAB NAVIGATION --- */}
      <div style={styles.tabContainer}>
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            style={{
              ...styles.tabButton,
              borderBottom: activeTab === t.id ? `2px solid ${t.color}` : '2px solid transparent',
              background: activeTab === t.id ? '#262730' : 'transparent',
              color: activeTab === t.id ? '#fff' : '#888',
              fontWeight: activeTab === t.id ? '700' : '500'
            }}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {/* --- CONTENT AREA --- */}
      <div style={styles.contentArea}> 
        {activeTab === "Lore" && <LoreTab profile={profile} />}
        {activeTab === "Character" && <CharactersTab profile={profile} />}
        {activeTab === "Faction" && <FactionTab profile={profile} />}
        {activeTab === "Rules" && <RulesTab profile={profile} />}
        {activeTab === "Plans" && <PlansTab profile={profile} />}
        {activeTab === "Facts" && <FactsTab profile={profile} />}
        {activeTab === "Spoilers" && <SpoilersTab profile={profile} />}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

    </div>
  );
}

const styles = {
  container: {
    padding: '30px',
    maxWidth: '1400px',
    margin: '0 auto',
    color: '#fff',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    boxSizing: 'border-box',
    width: '100%'
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '20px'
  },
  title: {
    margin: 0,
    fontSize: '24px',
    fontWeight: '700'
  },
  tabContainer: {
    display: 'flex',
    gap: '5px',
    marginBottom: '20px',
    borderBottom: '1px solid #333'
  },
  tabButton: {
    padding: '10px 20px',
    borderTop: 'none',
    borderLeft: 'none',
    borderRight: 'none',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '14px',
    transition: 'all 0.2s ease',
    outline: 'none'
  },
  contentArea: {
    flex: 1,
    minHeight: '0',
    position: 'relative'
  },
  remetadataBtn: {
    display: 'flex', alignItems: 'center', gap: '8px',
    padding: '8px 14px', background: 'transparent',
    border: '1px solid #3f3f46', color: '#a1a1aa',
    borderRadius: '6px', fontSize: '12px', fontWeight: '600',
    transition: 'all 0.2s'
  },
  resultBadge: {
    display: 'flex', alignItems: 'center', gap: '6px',
    fontSize: '12px', color: '#22c55e',
    background: 'rgba(34,197,94,0.08)',
    border: '1px solid rgba(34,197,94,0.2)',
    padding: '5px 10px', borderRadius: '6px'
  }
};
