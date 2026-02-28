import React from 'react';
import axios from 'axios';
import { 
  PenTool, MessageSquare, Swords, Book, BarChart3, 
  Share2, MessageCircle, FileText, Settings, LogOut, Archive
} from 'lucide-react';

const API_URL = "http://localhost:8000";

/**
 * Sidebar Navigation Component
 * ============================
 * Primary navigation controller for the application.
 * Handles module routing, profile management, and project export operations.
 */
export default function Sidebar({ activeTab, setActiveTab, currentProfile, onSwitchProfile }) {
  
  /**
   * Initiates a full backup of the current profile directory.
   * content is streamed as a ZIP file from the backend.
   */
  const handleExport = async () => {
    if (!confirm(`Create and download backup for "${currentProfile}"?`)) return;
    try {
      const res = await axios.get(`${API_URL}/profiles/export/${currentProfile}`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${currentProfile}_Backup.zip`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Export failed:", err);
      alert("Failed to export profile. Please verify backend connection.");
    }
  };

  // Module Configuration
  const menuItems = [
    { id: "scene", label: "Scene Creator", icon: <PenTool size={16} /> },
    { id: "chat", label: "Co-Author Chat", icon: <MessageSquare size={16} /> },
    { id: "warroom", label: "War Room", icon: <Swords size={16} /> },
    { id: "knowledge", label: "Knowledge Base", icon: <Book size={16} /> },
    { id: "tracker", label: "World State Tracker", icon: <BarChart3 size={16} /> },
    { id: "map", label: "Network Map", icon: <Share2 size={16} /> },
    { id: "reaction", label: "Reaction Tool", icon: <MessageCircle size={16} /> },
    { id: "compiler", label: "Compiler", icon: <FileText size={16} /> },
    { id: "settings", label: "Settings", icon: <Settings size={16} /> },
  ];

  return (
    <div style={styles.container}>
      
      {/* --- BRANDING --- */}
      <div style={styles.header}>
        <div style={{ fontSize: '28px', lineHeight: '1' }}>🕰️</div>
        <div>
          <h1 style={styles.title}>Chronos</h1>
          <h2 style={styles.subtitle}>Story Director</h2>
        </div>
      </div>

      {/* --- PROFILE CONTEXT --- */}
      <div style={styles.profileSection}>
        <div style={styles.profileLabel}>Workspace</div>
        <div style={styles.profileName} title={currentProfile}>
          {currentProfile}
        </div>
        <button 
          onClick={onSwitchProfile}
          style={styles.switchButton}
          title="Switch to a different project profile"
        >
          <LogOut size={12} /> Switch Profile
        </button>
      </div>

      <div style={styles.divider} />

      {/* --- NAVIGATION MODULES --- */}
      <div style={styles.navContainer}>
        {menuItems.map(item => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            style={{
              ...styles.navItem,
              ...(activeTab === item.id ? styles.navItemActive : {})
            }}
          >
            {item.icon} {item.label}
          </button>
        ))}
      </div>

      <div style={styles.divider} />

      {/* --- SYSTEM OPERATIONS --- */}
      <div style={styles.systemContainer}>
        <div style={styles.sectionLabel}>System</div>
        <button 
          onClick={handleExport}
          style={styles.navItem} 
        >
          <Archive size={16} /> Export / Backup
        </button>
      </div>

      {/* --- METADATA --- */}
      <div style={styles.footer}>
        v14.2 - Adaptive Realism
      </div>

    </div>
  );
}

/**
 * Component Styles
 */
const styles = {
  container: {
    display: 'flex', 
    flexDirection: 'column', 
    height: '100%', 
    padding: '20px', 
    background: '#262730', // Primary Dark Surface
    color: '#ffffff',
    overflowY: 'auto',
    borderRight: '1px solid #111',
    boxSizing: 'border-box',
    fontFamily: 'sans-serif'
  },
  header: {
    marginBottom: '25px', 
    display: 'flex', 
    alignItems: 'center', 
    gap: '12px'
  },
  title: {
    margin: 0, 
    fontSize: '20px', 
    fontWeight: '700', 
    color: '#ffffff',
    letterSpacing: '-0.5px'
  },
  subtitle: {
    margin: 0, 
    fontSize: '12px', 
    fontWeight: '400', 
    color: '#a0a0a0'
  },
  profileSection: {
    marginBottom: '10px',
    display: 'flex',
    flexDirection: 'column',
    gap: '6px'
  },
  profileLabel: {
    fontSize: '10px', 
    color: '#808495', 
    fontWeight: '700', 
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  },
  profileName: {
    color: '#ffffff', 
    fontSize: '13px', 
    fontWeight: '600', 
    whiteSpace: 'nowrap', 
    overflow: 'hidden', 
    textOverflow: 'ellipsis',
    paddingBottom: '4px'
  },
  switchButton: {
    width: '100%', 
    padding: '6px 8px', 
    background: 'transparent', 
    border: '1px solid #4a4e57', 
    borderRadius: '4px', 
    color: '#c0c0c0', 
    fontSize: '11px', 
    cursor: 'pointer', 
    display: 'flex', 
    justifyContent: 'center', 
    gap: '6px', 
    alignItems: 'center', 
    transition: 'all 0.15s ease'
  },
  divider: {
    height: '1px', 
    background: '#4a4e57', 
    margin: '15px 0',
    opacity: 0.4
  },
  navContainer: {
    display: 'flex', 
    flexDirection: 'column', 
    gap: '2px'
  },
  navItem: {
    display: 'flex', 
    alignItems: 'center', 
    gap: '10px', 
    padding: '8px 10px',
    borderRadius: '4px', 
    border: 'none', 
    cursor: 'pointer', 
    textAlign: 'left',
    background: 'transparent', 
    color: '#d0d0d0',
    fontSize: '14px', 
    fontWeight: '400',
    transition: 'background 0.15s ease, color 0.15s ease'
  },
  navItemActive: {
    background: '#ff4b4b', // Accent Color
    color: '#ffffff',
    fontWeight: '600'
  },
  systemContainer: {
    display: 'flex', 
    flexDirection: 'column', 
    gap: '5px'
  },
  sectionLabel: {
    fontSize: '10px', 
    color: '#808495', 
    fontWeight: '700', 
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    marginBottom: '4px',
    paddingLeft: '4px'
  },
  footer: {
    marginTop: 'auto', 
    paddingTop: '20px',
    fontSize: '10px', 
    color: '#606470', 
    textAlign: 'center'
  }
};