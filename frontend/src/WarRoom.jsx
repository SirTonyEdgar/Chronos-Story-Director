import React, { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Swords, Save, Play, AlertTriangle, ShieldAlert, FileText, X } from 'lucide-react';
import { API_URL } from './config';
import { toast, confirm } from './components/Notifications';

export default function WarRoom({ profile }) {
  const [scenario, setScenario] = useState("");
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [planName, setPlanName] = useState("");

  const handleRun = async () => {
    if (!scenario) return toast("Please define a scenario first.", "warning");
    setLoading(true);
    setReport(""); 

    try {
      const res = await axios.post(`${API_URL}/simulation/run/${profile}`, {
        scenario: scenario
      });
      setReport(res.data.report);
    } catch (err) {
      console.error(err);
      toast("Simulation Failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setLoading(false);
    }
  };

  const handleSaveClick = () => {
    setPlanName(`Op: ${scenario.slice(0, 20)}...`);
    setShowSaveModal(true);
  };

  const confirmSaveToPlans = async () => {
    if (!planName) return;
    setIsSaving(true);
    try {
      const fullContent = `SCENARIO:\n${scenario}\n\nREPORT:\n${report}`;
      await axios.post(`${API_URL}/knowledge/create/${profile}`, {
        name: planName,
        content: fullContent,
        category: "Plan"
      });
      toast("Saved to Knowledge Base (Plans)!", "success");
      setShowSaveModal(false);
    } catch (err) {
      toast("Save failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    // 1. OUTER CONTAINER: Increased Left Padding to 60px to clear Sidebar overlap
    <div style={{ 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column', 
      padding: '30px 30px 30px 60px',
      boxSizing: 'border-box'
    }}>
      
      {/* HEADER */}
      <div style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
        <h2 style={{ margin: 0, fontSize: '24px', fontWeight: '700', letterSpacing: '-0.5px' }}>
          ⚔️ War Room
        </h2>
      </div>

      {/* 2. THE WINDOW */}
      <div style={{ 
        flex: 1, 
        display: 'flex', 
        border: '1px solid #333', 
        borderRadius: '8px', 
        overflow: 'hidden', 
        background: '#09090b',
        boxShadow: '0 4px 20px rgba(0,0,0,0.4)'
      }}>
        
        {/* --- LEFT PANEL: CONTROLS --- */}
        <div style={{ 
          width: '350px', 
          minWidth: '350px', 
          display: 'flex', 
          flexDirection: 'column', 
          borderRight: '1px solid #333', 
          background: '#111',
          padding: '20px',
          gap: '20px',
          overflowY: 'auto'
        }}>
          
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: '12px', fontWeight: 'bold', color: '#888', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              <ShieldAlert size={14} color="#f87171"/> Strategy Input
            </label>
            <textarea
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              placeholder="Describe your proposed action (e.g., 'Hostile takeover of the kingdom', 'Infiltrate the Gala')..."
              style={{
                flex: 1,
                background: '#18181b', 
                border: '1px solid #333', 
                borderRadius: '6px',
                color: '#fff', 
                padding: '15px', 
                fontSize: '14px', 
                lineHeight: '1.6',
                resize: 'none', 
                outline: 'none',
                fontFamily: 'inherit'
              }}
            />
          </div>

          <button
            onClick={handleRun}
            disabled={loading}
            style={{
              background: loading ? '#27272a' : '#ef4444',
              color: '#fff', 
              border: 'none', 
              padding: '14px', 
              borderRadius: '6px',
              fontSize: '14px', 
              fontWeight: '700', 
              cursor: loading ? 'default' : 'pointer',
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              gap: '8px',
              transition: 'all 0.2s'
            }}
          >
            {loading ? "Simulating..." : <><Play size={18} fill="currentColor" /> RUN SIMULATION</>}
          </button>

          <div style={{ background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.15)', borderRadius: '6px', padding: '15px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#f87171', fontWeight: 'bold', fontSize: '13px', marginBottom: '5px' }}>
              <AlertTriangle size={14} /> System Logic
            </div>
            <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '12px', color: '#a1a1aa', lineHeight: '1.6' }}>
              <li>Cross-references <b>Allies</b> & <b>Assets</b>.</li>
              <li>Checks <b>Skills</b> for competency.</li>
              <li>Calculates <b>Betrayal Risks</b>.</li>
            </ul>
          </div>
        </div>

        {/* --- RIGHT PANEL: OUTPUT --- */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0c0c0c', position: 'relative', overflow: 'hidden' }}>
          
          <div style={{ height: '60px', borderBottom: '1px solid #222', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 25px', background: '#111' }}>
            <h3 style={{ margin: 0, fontSize: '15px', color: '#e4e4e7', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <FileText size={16} color="#71717a"/> Simulation Report
            </h3>
            {/* The old Save button used to be here - it has been removed */}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '40px' }}>
            {report ? (
              <div className="markdown-content" style={{ maxWidth: '800px', margin: '0 auto', fontSize: '15px', lineHeight: '1.8', color: '#d4d4d8' }}>
                <ReactMarkdown>{report}</ReactMarkdown>
              </div>
            ) : (
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#27272a' }}>
                <Swords size={80} style={{ opacity: 0.05, marginBottom: '20px' }} />
                <p style={{ fontSize: '14px', fontWeight: '500', color: '#444' }}>Awaiting Operational Data...</p>
              </div>
            )}
          </div>

          {/* Sticky Action Footer (Only shows if there is a report) */}
          {report && (
            <div style={{ padding: '20px 40px', background: '#111', borderTop: '1px solid #222', display: 'flex', justifyContent: 'flex-end' }}>
              <button 
                onClick={handleSaveClick}
                style={{
                  background: '#3b82f6', color: '#fff', border: 'none', padding: '12px 24px', 
                  borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px',
                  fontSize: '14px', fontWeight: 'bold', boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
                }}
              >
                <Save size={18} /> Save to Plans
              </button>
            </div>
          )}
        </div>

      </div>
      {/* --- CUSTOM SAVE MODAL --- */}
      {showSaveModal && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.75)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', animation: 'fadeIn 0.2s ease-out' }}>
          <div style={{ background: '#111', border: '1px solid #333', borderRadius: '8px', padding: '24px', width: '450px', maxWidth: '90%', boxShadow: '0 10px 30px rgba(0,0,0,0.8)' }}>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
              <h3 style={{ margin: 0, color: '#fff', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '18px' }}>
                <Save size={20} color="#3b82f6" /> Save Operation Plan
              </h3>
              <button onClick={() => setShowSaveModal(false)} style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer' }}>
                <X size={20} />
              </button>
            </div>
            
            <p style={{ fontSize: '14px', color: '#ccc', lineHeight: '1.6', marginBottom: '20px' }}>
              Assign a codename to this operation to archive the scenario and report into your Knowledge Base.
            </p>
            
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 'bold', color: '#888', marginBottom: '8px' }}>OPERATION NAME</label>
            <input 
              value={planName}
              onChange={(e) => setPlanName(e.target.value)}
              autoFocus
              style={{ width: '100%', background: '#1a1a1a', border: '1px solid #333', color: '#fff', padding: '12px', borderRadius: '4px', outline: 'none', fontSize: '14px', marginBottom: '24px', boxSizing: 'border-box' }}
            />

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
              <button 
                onClick={() => setShowSaveModal(false)} 
                style={{ padding: '10px 16px', background: 'transparent', border: '1px solid #444', color: '#ccc', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
              >
                Cancel
              </button>
              <button 
                onClick={confirmSaveToPlans} 
                disabled={isSaving}
                style={{ padding: '10px 16px', background: '#3b82f6', border: 'none', color: '#fff', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '8px' }}
              >
                {isSaving ? "Saving..." : "Confirm & Save"}
              </button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}