import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { EyeOff, Trash2, AlertTriangle, Plus } from 'lucide-react';

const API_URL = "http://localhost:8000";

export default function SpoilersTab({ profile }) {
  const [spoilers, setSpoilers] = useState([]);
  const [newBan, setNewBan] = useState("");

  // Reload when profile changes
  useEffect(() => { 
    if (profile) fetchSpoilers(); 
  }, [profile]);

  const fetchSpoilers = async () => {
    try {
      const res = await axios.get(`${API_URL}/knowledge/list/${profile}/Spoiler`);
      setSpoilers(res.data || []);
    } catch (err) { console.error(err); }
  };

  const handleAdd = async () => {
    if (!newBan) return;
    try {
      await axios.post(`${API_URL}/knowledge/create/${profile}`, {
        name: "Spoiler_Alert",
        content: newBan,
        category: "Spoiler"
      });
      setNewBan("");
      fetchSpoilers();
    } catch (err) { alert("Failed to add ban."); }
  };

  const handleDelete = async (id) => {
    try {
      await axios.post(`${API_URL}/knowledge/delete/${profile}`, { id });
      fetchSpoilers();
    } catch (err) { alert("Failed to delete."); }
  };

  return (
    <div style={{ padding: '20px', color: '#eee', maxWidth: '800px', margin: '0 auto' }}>
      
      {/* HEADER */}
      <div style={{ background: '#450a0a', border: '1px solid #991b1b', padding: '20px', borderRadius: '8px', display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '30px' }}>
        <EyeOff size={32} color="#f87171" />
        <div>
          <h3 style={{ margin: 0, color: '#fca5a5', fontSize: '18px' }}>Banned Content (The Anti-Prompt)</h3>
          <p style={{ margin: '5px 0 0 0', fontSize: '13px', color: '#fecaca', lineHeight: '1.5' }}>
            Concepts, twists, or names the AI is explicitly <b>FORBIDDEN</b> from mentioning until you decide it's time.
          </p>
        </div>
      </div>

      {/* INPUT */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '30px' }}>
        <input 
          value={newBan} 
          onChange={e => setNewBan(e.target.value)}
          placeholder="Enter secret to hide (e.g. 'Darth Vader is the father')..."
          style={{ flex: 1, padding: '12px', borderRadius: '6px', border: '1px solid #333', background: '#18181b', color: '#fff', outline: 'none' }}
        />
        <button 
          onClick={handleAdd}
          style={{ background: '#dc2626', color: 'white', border: 'none', padding: '0 24px', borderRadius: '6px', fontWeight: 'bold', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <Plus size={18}/> Ban Term
        </button>
      </div>

      {/* LIST */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {spoilers.map(s => (
          <div key={s.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#18181b', padding: '15px 20px', borderRadius: '6px', border: '1px solid #333' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <AlertTriangle size={18} color="#fbbf24" />
              <span style={{ fontFamily: 'monospace', fontSize: '14px', color: '#e4e4e7', fontWeight: 'bold' }}>STOP: {s.content}</span>
            </div>
            <button onClick={() => handleDelete(s.id)} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#71717a', padding: '5px' }} title="Remove Ban">
              <Trash2 size={18} />
            </button>
          </div>
        ))}
        {spoilers.length === 0 && <div style={{ textAlign: 'center', color: '#555', marginTop: '20px', fontStyle: 'italic' }}>No active spoilers defined.</div>}
      </div>

    </div>
  );
}