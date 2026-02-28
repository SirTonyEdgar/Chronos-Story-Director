import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Clock, FolderPlus, LogIn } from 'lucide-react';

const API_URL = "http://localhost:8000";

export default function ProfileSelector({ onSelect }) {
  const [profiles, setProfiles] = useState([]);
  const [selected, setSelected] = useState("");
  const [newProfileName, setNewProfileName] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfiles();
  }, []);

  const fetchProfiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/profiles/list`);
      setProfiles(res.data);
      if (res.data.length > 0) setSelected(res.data[0]);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!newProfileName) return;
    try {
      const res = await axios.post(`${API_URL}/profiles/create?name=${newProfileName}`);
      alert(`Profile "${res.data.name}" Created!`);
      await fetchProfiles();
      setSelected(res.data.name);
      setNewProfileName("");
    } catch (err) {
      alert("Error: " + err.message);
    }
  };

  if (loading) return <div style={containerStyle}>Loading Profiles...</div>;

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <div style={{ fontSize: '40px', marginBottom: '10px' }}>🕰️</div>
          <h1 style={{ margin: 0, fontSize: '24px', color: '#fff' }}>Chronos Story Director</h1>
          <p style={{ color: '#666', fontSize: '14px' }}>Select a Timeline Profile to begin</p>
        </div>

        {/* SELECT EXISTING */}
        <div style={{ marginBottom: '25px' }}>
          <label style={labelStyle}>Select Profile</label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <select 
              value={selected} 
              onChange={(e) => setSelected(e.target.value)}
              style={selectStyle}
            >
              {profiles.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
            <button 
              onClick={() => onSelect(selected)}
              style={primaryBtnStyle}
            >
              <LogIn size={16} /> Load
            </button>
          </div>
        </div>

        <div style={{ borderTop: '1px solid #333', margin: '20px 0' }} />

        {/* CREATE NEW */}
        <div>
          <label style={labelStyle}>Create New Profile</label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input 
              value={newProfileName}
              onChange={(e) => setNewProfileName(e.target.value)}
              placeholder="e.g. Project_Titan"
              style={inputStyle}
            />
            <button 
              onClick={handleCreate}
              style={secondaryBtnStyle}
            >
              <FolderPlus size={16} /> Create
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}

// STYLES
const containerStyle = { height: '100vh', width: '100vw', background: '#000', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', fontFamily: 'sans-serif' };
const cardStyle = { width: '400px', padding: '40px', background: '#09090b', border: '1px solid #27272a', borderRadius: '12px' };
const labelStyle = { display: 'block', fontSize: '12px', fontWeight: 'bold', color: '#888', marginBottom: '8px' };
const selectStyle = { flex: 1, padding: '10px', background: '#18181b', border: '1px solid #333', color: '#fff', borderRadius: '6px', outline: 'none' };
const inputStyle = { flex: 1, padding: '10px', background: '#18181b', border: '1px solid #333', color: '#fff', borderRadius: '6px', outline: 'none' };
const primaryBtnStyle = { padding: '10px 20px', background: '#ef4444', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' };
const secondaryBtnStyle = { padding: '10px 20px', background: '#333', color: '#fff', border: '1px solid #444', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' };