import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Send, Trash2, Bot, User, Save, FileText, Loader2, Zap } from 'lucide-react';
import { API_URL } from './config';
import { toast, confirm } from './components/Notifications';

export default function CoAuthorChat({ profile }) {
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  
  // --- MULTIVERSE STATE ---
  const [timeline, setTimeline] = useState("");
  const [availableTimelines, setAvailableTimelines] = useState([]);
  
  const scrollRef = useRef(null);

  // 1. Fetch history and timelines whenever the profile changes
  useEffect(() => {
    if (profile) {
      fetchHistory();
      fetchTimelines();
    }
  }, [profile]);

  // 2. Auto-scroll to bottom when history updates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [history, loading]);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_URL}/chat/history/${profile}`);
      setHistory(res.data);
    } catch (err) { console.error(err); }
  };

  const fetchTimelines = async () => {
    try {
      const res = await axios.get(`${API_URL}/state/${profile}`);
      setAvailableTimelines(res.data.Timelines || []);
    } catch (err) { console.error("Failed to fetch timelines:", err); }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: "user", content: input };
    setHistory(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chat/query/${profile}`, {
        prompt: userMsg.content,
        timeline: timeline // <--- Passed to backend!
      });
      const aiMsg = { role: "assistant", content: res.data.response };
      setHistory(prev => [...prev, aiMsg]);
    } catch (err) {
      const errorMsg = { role: "assistant", content: "⚠️ Error: " + (err.response?.data?.detail || err.message) };
      setHistory(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    const ok = await confirm("Clear entire chat history?", { title: "Clear History", confirmLabel: "Clear", danger: true });
    if (!ok) return;
    try {
      await axios.post(`${API_URL}/chat/clear/${profile}`);
      setHistory([]);
    } catch (err) {
      toast("Failed to clear history: " + err.message, "error");
    }
  };

  const handleSaveDraft = async () => {
    const lastMsg = history[history.length - 1];
    if (!lastMsg || lastMsg.role !== "assistant") return;
    
    if (!saveName.trim()) return toast("Please enter a filename.", "warning");

    setIsSaving(true);
    try {
      // Reusing the Scene Edit endpoint to create a new file
      await axios.post(`${API_URL}/scene/save/${profile}`, {
        filename: saveName.endsWith(".txt") ? saveName : `${saveName}.txt`,
        content: lastMsg.content
      });
      toast(`Saved "${saveName}" to Scene Creator!`, "success");
      setSaveName("");
    } catch (err) {
      toast("Save failed: " + err.message, "error");
    } finally {
      setIsSaving(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', maxWidth: '1000px', margin: '0 auto', padding: '20px', boxSizing: 'border-box', width: '100%' }}>
      
      {/* HEADER */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid #333', paddingBottom: '15px' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px', fontSize: '20px', color: '#e4e4e7' }}>
          <Bot size={24} color="#3b82f6" /> Co-Author Chat
        </h2>
        
        {/* HEADER CONTROLS: Timeline Dropdown & Clear Button */}
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          
          {/* Multiverse Dropdown */}
          {availableTimelines.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(168, 85, 247, 0.1)', border: '1px solid #a855f7', padding: '4px 10px', borderRadius: '4px' }}>
              <Zap size={14} color="#a855f7" />
              <select 
                value={timeline}
                onChange={(e) => setTimeline(e.target.value)}
                style={{ background: 'transparent', border: 'none', color: '#e4e4e7', outline: 'none', fontSize: '12px', cursor: 'pointer' }}
              >
                <option value="">Universal (All Timelines)</option>
                {availableTimelines.map((tl, i) => (
                  <option key={i} value={tl.Name}>{tl.Name}</option>
                ))}
              </select>
            </div>
          )}

          <button 
            onClick={handleClear}
            style={{ background: 'transparent', border: '1px solid #333', color: '#a1a1aa', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', transition: 'all 0.2s' }}
            onMouseOver={(e) => { e.currentTarget.style.color = '#ef4444'; e.currentTarget.style.borderColor = '#ef4444'; }}
            onMouseOut={(e) => { e.currentTarget.style.color = '#a1a1aa'; e.currentTarget.style.borderColor = '#333'; }}
          >
            <Trash2 size={14} /> Clear History
          </button>
        </div>
      </div>

      {/* CHAT WINDOW */}
      <div style={{ flex: 1, overflowY: 'auto', paddingRight: '10px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {history.length === 0 && (
          <div style={{ textAlign: 'center', color: '#52525b', marginTop: '50px' }}>
            <Bot size={48} style={{ opacity: 0.2, marginBottom: '10px' }} />
            <p>Ask me about your lore, plot holes, or mechanics...</p>
          </div>
        )}

        {history.map((msg, i) => (
          <div key={i} style={{ display: 'flex', gap: '15px', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
            
            {/* AVATAR */}
            <div style={{ 
              width: '32px', height: '32px', borderRadius: '4px', flexShrink: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: msg.role === 'user' ? '#27272a' : '#3b82f6',
              border: msg.role === 'user' ? '1px solid #3f3f46' : 'none'
            }}>
              {msg.role === 'user' ? <User size={18} color="#e4e4e7" /> : <Bot size={18} color="white" />}
            </div>

            {/* BUBBLE */}
            <div style={{ 
              background: msg.role === 'user' ? '#18181b' : '#09090b', 
              border: '1px solid #27272a',
              borderRadius: '8px', padding: '15px', maxWidth: '80%',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
            }}>
              <div className="markdown-content" style={{ fontSize: '15px', lineHeight: '1.6', color: '#e4e4e7' }}>
                {msg.role === 'user' ? msg.content : <ReactMarkdown>{msg.content}</ReactMarkdown>}
              </div>
            </div>

          </div>
        ))}
        
        {loading && (
          <div style={{ display: 'flex', gap: '15px' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '4px', background: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Bot size={18} color="white" />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', color: '#a1a1aa', fontSize: '14px', gap: '10px' }}>
              <Loader2 className="spin" size={16} /> Consulting Knowledge Base...
            </div>
          </div>
        )}

        <div ref={scrollRef} />
      </div>

      {/* SAVE DRAFT TOOL */}
      {history.length > 0 && history[history.length - 1].role === "assistant" && !loading && (
        <div style={{ margin: '10px 0', padding: '10px', background: '#18181b', border: '1px solid #27272a', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <FileText size={16} color="#3b82f6" />
          <span style={{ fontSize: '13px', color: '#a1a1aa', fontWeight: '600' }}>Save Last Response as Draft:</span>
          <input 
            value={saveName} 
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="Draft_Filename"
            style={{ background: '#09090b', border: '1px solid #3f3f46', color: '#fff', padding: '8px', borderRadius: '4px', fontSize: '13px', flex: 1, outline: 'none' }}
          />
          <button 
            onClick={handleSaveDraft}
            disabled={isSaving}
            style={{ background: '#22c55e', color: '#fff', border: 'none', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px', transition: 'background 0.2s' }}
          >
            <Save size={14} /> {isSaving ? "Saving..." : "Save to Scenes"}
          </button>
        </div>
      )}

      {/* INPUT AREA */}
      <div style={{ marginTop: '20px', position: 'relative' }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={timeline ? `Asking the AI about [${timeline}]...` : "Type your query here... (Shift+Enter for new line)"}
          style={{
            width: '100%', background: '#09090b', border: '1px solid #3f3f46', borderRadius: '8px',
            color: '#fff', padding: '15px', paddingRight: '50px', fontSize: '14px', 
            resize: 'none', height: '60px', outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box'
          }}
        />
        <button 
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            position: 'absolute', right: '10px', top: '10px',
            background: loading ? 'transparent' : '#3b82f6', 
            color: loading ? '#52525b' : 'white', 
            border: 'none', borderRadius: '6px', width: '40px', height: '40px',
            cursor: loading || !input.trim() ? 'default' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            transition: 'background 0.2s'
          }}
        >
          <Send size={18} />
        </button>
      </div>

    </div>
  );
}