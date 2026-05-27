import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { EyeOff, Trash2, AlertTriangle, Plus } from 'lucide-react';
import { API_URL } from '../config';
import { toast } from '../components/Notifications';

export default function SpoilersTab({ profile }) {
  const [spoilers, setSpoilers] = useState([]);
  const [newBan, setNewBan] = useState("");
  const [newRevealDate, setNewRevealDate] = useState("");
  const [isAuditing, setIsAuditing] = useState(false);
  const [auditResults, setAuditResults] = useState(null);

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
        category: "Spoiler",
        reveal_date: newRevealDate
      });
      setNewBan("");
      setNewRevealDate("");
      fetchSpoilers();
    } catch (err) { toast("Failed to add ban.", "error"); }
  };

  const handleDelete = async (id) => {
    try {
      await axios.post(`${API_URL}/knowledge/delete/${profile}`, { id });
      fetchSpoilers();
    } catch (err) { toast("Failed to delete.", "error"); }
  };

  const handleAudit = async () => {
    setIsAuditing(true);
    setAuditResults(null);
    try {
      const res = await axios.post(`${API_URL}/scene/audit_spoilers/${profile}`);
      setAuditResults(res.data);
      if (res.data.total_flagged === 0) {
        toast("Audit complete. No spoiler leaks detected.", "success");
      } else {
        toast(`Audit complete. ${res.data.total_flagged} scene${res.data.total_flagged > 1 ? 's' : ''} flagged.`, "warning");
      }
    } catch (err) {
      toast("Audit failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setIsAuditing(false);
    }
  };

  return (
    <div style={{ padding: '20px', color: '#eee', maxWidth: '800px', margin: '0 auto' }}>
      
      {/* HEADER */}
      <div style={{ background: '#450a0a', border: '1px solid #991b1b', padding: '20px', borderRadius: '8px', display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '30px', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          <EyeOff size={32} color="#f87171" />
          <div>
            <h3 style={{ margin: 0, color: '#fca5a5', fontSize: '18px' }}>Banned Content (The Anti-Prompt)</h3>
            <p style={{ margin: '5px 0 0 0', fontSize: '13px', color: '#fecaca', lineHeight: '1.5' }}>
              Concepts, twists, or names the AI is explicitly <b>FORBIDDEN</b> from mentioning until you decide it's time.
            </p>
          </div>
        </div>
        <button
          onClick={handleAudit}
          disabled={isAuditing || spoilers.length === 0}
          style={{
            padding: '8px 16px', background: 'transparent',
            border: '1px solid #991b1b', color: '#f87171',
            borderRadius: '6px', cursor: isAuditing || spoilers.length === 0 ? 'default' : 'pointer',
            fontSize: '12px', fontWeight: '600', whiteSpace: 'nowrap',
            opacity: spoilers.length === 0 ? 0.4 : 1
          }}
        >
          {isAuditing ? 'Scanning Scenes...' : '🔍 Audit Scenes'}
        </button>
      </div>

      {/* AUDIT RESULTS */}
      {auditResults && (
        <div style={{ marginBottom: '30px', border: '1px solid #333', borderRadius: '8px', overflow: 'hidden', background: '#111' }}>
          <div style={{ padding: '12px 16px', background: '#1a1a1a', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '13px', fontWeight: '700', color: auditResults.total_flagged > 0 ? '#f87171' : '#22c55e' }}>
              {auditResults.total_flagged > 0
                ? `⚠️ ${auditResults.total_flagged} scene${auditResults.total_flagged > 1 ? 's' : ''} with potential spoiler leaks`
                : '✓ All scenes clean — no spoiler leaks detected'}
            </span>
            <button onClick={() => setAuditResults(null)} style={{ background: 'none', border: 'none', color: '#555', cursor: 'pointer', fontSize: '16px' }}>×</button>
          </div>
          {auditResults.total_flagged > 0 && (
            <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {auditResults.flagged.map((f, i) => (
                <div key={i} style={{ padding: '12px', background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '6px' }}>
                  <div style={{ fontSize: '13px', fontWeight: '700', color: '#f87171', marginBottom: '6px' }}>
                    {f.filename}
                  </div>
                  <div style={{ fontSize: '12px', color: '#a1a1aa', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                    {f.violations}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* INPUT */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginBottom: '30px' }}>
        <div style={{ display: 'flex', gap: '10px' }}>
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
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <input 
            value={newRevealDate} 
            onChange={e => setNewRevealDate(e.target.value)}
            placeholder="Reveal Date (optional) — e.g. 1987-06-15 or June 15, 1987 or Day 47 of the Red Moon"
            style={{ flex: 1, padding: '10px 12px', borderRadius: '6px', border: '1px solid #333', background: '#18181b', color: '#a1a1aa', outline: 'none', fontSize: '13px' }}
          />
          <span style={{ fontSize: '12px', color: '#52525b', whiteSpace: 'nowrap' }}>
            Leave blank to suppress permanently
          </span>
        </div>
      </div>

      {/* LIST */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {spoilers.map(s => (
          <div key={s.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: '#18181b', padding: '15px 20px', borderRadius: '6px', border: '1px solid #333' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <AlertTriangle size={18} color="#fbbf24" />
              <div>
                <span style={{ fontFamily: 'monospace', fontSize: '14px', color: '#e4e4e7', fontWeight: 'bold' }}>STOP: {s.content}</span>
                {s.reveal_date ? (
                  <div style={{ fontSize: '11px', color: '#a855f7', marginTop: '4px' }}>
                    Reveals after: {s.reveal_date}
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: '#52525b', marginTop: '4px' }}>
                    Permanently suppressed
                  </div>
                )}
              </div>
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