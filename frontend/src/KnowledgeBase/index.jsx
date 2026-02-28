import React, { useState, useEffect } from 'react';
import { Book, Shield, Map, FileText, EyeOff } from 'lucide-react';

// Import Sub-Tabs
import LoreTab from './LoreTab';
import RulesTab from './RulesTab';
import PlansTab from './PlansTab';
import FactsTab from './FactsTab';
import SpoilersTab from './SpoilersTab';

/**
 * Knowledge Base Container
 * ========================
 * The central repository for world-building data.
 * Manages navigation between different knowledge domains (Lore, Rules, etc.)
 * and persists the active view across session refreshes.
 *
 * @param {string} profile - The currently active project profile.
 */
export default function KnowledgeBase({ profile }) {
  
  // --- NAVIGATION STATE (PERSISTENT) ---
  // Lazy initialization: Check localStorage for the last active tab
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem("chronos_kb_tab") || "Lore";
  });

  // Persist tab selection whenever it changes
  useEffect(() => {
    localStorage.setItem("chronos_kb_tab", activeTab);
  }, [activeTab]);

  // Tab Configuration
  const tabs = [
    { id: "Lore", icon: <Book size={16} />, label: "Lore", color: "#3b82f6" },
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
        {activeTab === "Rules" && <RulesTab profile={profile} />}
        {activeTab === "Plans" && <PlansTab profile={profile} />}
        {activeTab === "Facts" && <FactsTab profile={profile} />}
        {activeTab === "Spoilers" && <SpoilersTab profile={profile} />}
      </div>

    </div>
  );
}

/**
 * Component Styles
 */
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
  }
};