import React, { useState, useEffect } from 'react';
import { NotificationProvider } from './components/Notifications';

// Core Navigation Components
import Sidebar from './Sidebar';
import ProfileSelector from './ProfileSelector';

// Application Modules
import SceneCreator from './SceneCreator';
import NetworkMap from './NetworkMap';
import WorldStateTracker from './WorldStateTracker';
import KnowledgeBase from './KnowledgeBase';
import WarRoom from './WarRoom';
import ReactionTool from './ReactionTool';
import CoAuthorChat from './CoAuthorChat';
import Compiler from './Compiler';
import Settings from './Settings';

/**
 * App Component
 * =============
 * The root container for the Chronos Story Director.
 * Handles top-level state management, profile authentication/selection,
 * and module routing with state persistence across sessions.
 */
export default function App() {
  
  // --- STATE MANAGEMENT ---

  /**
   * Active Profile State
   * Persisted via "lastProfile" key. Null indicates no profile is selected.
   */
  const [profile, setProfile] = useState(() => {
    return localStorage.getItem("lastProfile") || null;
  });

  /**
   * Active Module/Tab State
   * Persisted via "chronos_active_module" key to restore workflow on refresh.
   * Defaults to "scene" (Scene Creator).
   */
  const [activeTab, setActiveTab] = useState(() => {
    return localStorage.getItem("chronos_active_module") || "scene";
  });

  // --- EFFECTS ---

  /**
   * Persist the active module selection whenever it changes.
   */
  useEffect(() => {
    if (activeTab) {
      localStorage.setItem("chronos_active_module", activeTab);
    }
  }, [activeTab]);

  // --- HANDLERS ---

  /**
   * selectProfile
   * Loads a specific project profile and saves it to local storage.
   * @param {string} profileName - The unique identifier of the project.
   */
  const handleProfileSelect = (profileName) => {
    setProfile(profileName);
    localStorage.setItem("lastProfile", profileName);
  };

  /**
   * switchProfile
   * Clears the current session and returns the user to the Profile Selector.
   */
  const handleSwitchProfile = () => {
    setProfile(null);
    localStorage.removeItem("lastProfile");
  };

  // --- RENDER LOGIC ---

  // 1. Unauthenticated State: Show Profile Selection Screen
  if (!profile) {
    return <ProfileSelector onSelect={handleProfileSelect} />;
  }

  // 2. Authenticated State: Render Main Dashboard
  return (
    <div style={styles.appContainer}>
      <NotificationProvider />
      {/* Primary Sidebar Navigation */}
      <div style={styles.sidebarContainer}>
        <Sidebar 
          activeTab={activeTab} 
          setActiveTab={setActiveTab} 
          currentProfile={profile} 
          onSwitchProfile={handleSwitchProfile} 
        />
      </div>

      {/* Main Content Area */}
      <div style={styles.contentContainer}>
        
        {/* Module Router: Dynamically renders components based on activeTab */}
        {/* The 'profile' prop is passed to all modules to ensure context awareness */}
        {activeTab === "scene" && <SceneCreator profile={profile} />}
        {activeTab === "chat" && <CoAuthorChat profile={profile} />}
        {activeTab === "warroom" && <WarRoom profile={profile} />}
        {activeTab === "knowledge" && <KnowledgeBase profile={profile} />}
        {activeTab === "tracker" && <WorldStateTracker profile={profile} />}
        {activeTab === "map" && <NetworkMap profile={profile} />}
        {activeTab === "reaction" && <ReactionTool profile={profile} />}
        {activeTab === "compiler" && <Compiler profile={profile} />}
        {activeTab === "settings" && <Settings profile={profile} />}

      </div>
    </div>
  );
}

/**
 * Global App Styles
 * Layout configuration for the dual-pane dashboard interface.
 */
const styles = {
  appContainer: {
    display: 'flex',
    width: '100vw',
    height: '100vh',
    background: '#000',
    color: '#fff',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    overflow: 'hidden' // Prevent global body scroll
  },
  sidebarContainer: {
    width: '260px',
    minWidth: '260px',
    borderRight: '1px solid #333',
    overflowY: 'auto',
    background: '#1a1a1a',
    zIndex: 10
  },
  contentContainer: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden', // Modules handle their own scrolling
    position: 'relative',
    background: '#0e1117' // Deep charcoal background for content
  }
};