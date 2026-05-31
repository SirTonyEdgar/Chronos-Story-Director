import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Book, Shield, Map, FileText, EyeOff, RefreshCw, CheckCircle2, Users, Network, Search, BookMarked, Ban } from 'lucide-react';
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
import ReservedNamesTab from './ReservedNamesTab';
import ReferenceTab from './ReferenceTab';

export default function KnowledgeBase({ profile }) {
  
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem("chronos_kb_tab") || "Lore";
  });

  const [isRemetadata, setIsRemetadata] = useState(false);
  const [remetadataResult, setRemetadataResult] = useState(null);
  const [remetadataProgress, setRemetadataProgress] = useState(null);
  // { current, total, filename, recentFiles: [{filename, status}] }

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
    setRemetadataProgress({ current: 0, total: 0, filename: '', recentFiles: [] });

    try {
      const eventSource = new EventSource(`${API_URL}/knowledge/remetadata/stream/${profile}`);

      eventSource.onmessage = (e) => {
        const event = JSON.parse(e.data);

        if (event.type === 'start') {
          setRemetadataProgress({ current: 0, total: event.total, filename: 'Starting...', recentFiles: [] });
        } else if (event.type === 'progress') {
          setRemetadataProgress(prev => ({
            current: event.current,
            total: event.total,
            filename: event.filename,
            recentFiles: [
              { filename: event.filename, status: event.status },
              ...(prev?.recentFiles || [])
            ].slice(0, 5)
          }));
        } else if (event.type === 'done') {
          setRemetadataResult(event);
          setRemetadataProgress(null);
          setIsRemetadata(false);
          toast(`Metadata regenerated. ${event.success} updated, ${event.skipped} skipped, ${event.failed} failed.`, "success");
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        toast("Re-metadata stream disconnected.", "error");
        setIsRemetadata(false);
        setRemetadataProgress(null);
        eventSource.close();
      };

    } catch (err) {
      toast("Bulk re-metadata failed: " + err.message, "error");
      setIsRemetadata(false);
      setRemetadataProgress(null);
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
    { id: "ReservedNames", icon: <Ban size={16} />, label: "Reserved Names", color: "#f97316" },
    { id: "Reference", icon: <BookMarked size={16} />, label: "Reference", color: "#06b6d4" },
  ];

  const [globalSearch, setGlobalSearch] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleGlobalSearch = async (query) => {
    setGlobalSearch(query);
    if (!query.trim() || query.length < 2) {
      setSearchResults([]);
      return;
    }
    setIsSearching(true);
    try {
      const res = await axios.get(`${API_URL}/knowledge/search/${profile}?q=${encodeURIComponent(query)}`);
      setSearchResults(res.data || []);
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div style={styles.container}>
      
      {/* --- HEADER --- */}
      <div style={styles.header}>
        <h2 style={styles.title}>🗄️ Knowledge Base</h2>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1, justifyContent: 'flex-end' }}>

          {/* Global Keyword Search */}
          <div style={{ position: 'relative' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', padding: '8px 12px' }}>
              <Search size={14} color="#555" />
              <input
                value={globalSearch}
                onChange={e => handleGlobalSearch(e.target.value)}
                placeholder="Search all knowledge..."
                style={{ background: 'transparent', border: 'none', color: '#fff', outline: 'none', fontSize: '13px', width: '200px' }}
              />
              {isSearching && <span style={{ fontSize: '11px', color: '#555' }}>...</span>}
              {globalSearch && (
                <button onClick={() => { setGlobalSearch(""); setSearchResults([]); }} style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', padding: '0' }}>
                  ×
                </button>
              )}
            </div>

            {searchResults.length > 0 && globalSearch && (
              <div style={{ position: 'absolute', top: '100%', right: 0, width: '480px', background: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', marginTop: '4px', zIndex: 100, maxHeight: '400px', overflowY: 'auto', boxShadow: '0 10px 25px rgba(0,0,0,0.8)' }}>
                <div style={{ padding: '8px 12px', borderBottom: '1px solid #27272a', fontSize: '11px', color: '#555', fontWeight: '700' }}>
                  {searchResults.length} RESULT{searchResults.length !== 1 ? 'S' : ''} FOR "{globalSearch}"
                </div>
                {searchResults.map(r => (
                  <div
                    key={r.id}
                    onClick={() => { setActiveTab(r.type); setGlobalSearch(""); setSearchResults([]); }}
                    style={{ padding: '10px 14px', borderBottom: '1px solid #1a1a1a', cursor: 'pointer', transition: 'background 0.1s' }}
                    onMouseOver={e => e.currentTarget.style.background = '#27272a'}
                    onMouseOut={e => e.currentTarget.style.background = 'transparent'}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3px' }}>
                      <span style={{ fontSize: '13px', color: '#e4e4e7', fontWeight: '600' }}>{r.name}</span>
                      <span style={{ fontSize: '10px', color: '#555', background: '#27272a', padding: '2px 6px', borderRadius: '4px' }}>{r.type}</span>
                    </div>
                    <div style={{ fontSize: '12px', color: '#71717a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {r.content}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {searchResults.length === 0 && globalSearch.length >= 2 && !isSearching && (
              <div style={{ position: 'absolute', top: '100%', right: 0, width: '300px', background: '#18181b', border: '1px solid #3f3f46', borderRadius: '6px', marginTop: '4px', zIndex: 100, padding: '12px', fontSize: '13px', color: '#555', textAlign: 'center' }}>
                No results for "{globalSearch}"
              </div>
            )}
          </div>

          {/* REMETADATA PROGRESS / RESULT */}
        {(isRemetadata || remetadataResult) && (
          <div style={{ margin: '0 20px 16px 20px', background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: '8px', padding: '14px', fontSize: '12px' }}>
            {isRemetadata && remetadataProgress && (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', color: '#71717a' }}>
                  <span>Regenerating metadata...</span>
                  <span>{remetadataProgress.current} / {remetadataProgress.total}</span>
                </div>
                <div style={{ background: '#18181b', borderRadius: '4px', height: '6px', overflow: 'hidden', marginBottom: '10px' }}>
                  <div style={{
                    height: '100%', borderRadius: '4px', background: '#3b82f6',
                    width: remetadataProgress.total > 0 ? `${(remetadataProgress.current / remetadataProgress.total) * 100}%` : '0%',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
                <div style={{ color: '#52525b', marginBottom: '8px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  Processing: {remetadataProgress.filename}
                </div>
                {remetadataProgress.recentFiles.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
                    {remetadataProgress.recentFiles.map((f, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '6px', color: f.status === 'success' ? '#4ade80' : f.status === 'failed' ? '#f87171' : '#52525b', opacity: 1 - i * 0.18 }}>
                        <span>{f.status === 'success' ? '✓' : f.status === 'failed' ? '✗' : '–'}</span>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.filename}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
            {!isRemetadata && remetadataResult && (
              <div style={{ color: '#4ade80' }}>
                ✓ Done — {remetadataResult.success} updated, {remetadataResult.skipped} skipped, {remetadataResult.failed} failed
                <button onClick={() => setRemetadataResult(null)} style={{ marginLeft: '12px', background: 'none', border: 'none', color: '#52525b', cursor: 'pointer', fontSize: '11px' }}>dismiss</button>
              </div>
            )}
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
        {activeTab === "ReservedNames" && <ReservedNamesTab profile={profile} />}
        {activeTab === "Reference" && <ReferenceTab profile={profile} />}
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
