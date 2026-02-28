import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Send, Trash2, Bot, User, Save, FileText, Loader2 } from 'lucide-react';

const API_URL = "http://localhost:8000";

export default function CoAuthorChat({ profile }) {
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  
  const scrollRef = useRef(null);

  // 1. Fetch history whenever the profile changes
  useEffect(() => {
    if (profile) {
      fetchHistory();
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

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: "user", content: input };
    setHistory(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chat/query/${profile}`, {
        prompt: userMsg.content
      });
      const aiMsg = { role: "assistant", content: res.data.response };
      setHistory(prev => [...prev, aiMsg]);
    } catch (err) {
      const errorMsg = { role: "assistant", content: "⚠️ Error: " + err.message };
      setHistory(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!confirm("Clear entire chat history?")) return;
    try {
      await axios.post(`${API_URL}/chat/clear/${profile}`);
      setHistory([]);
    } catch (err) {
      alert("Failed to clear history: " + err.message);
    }
  };

  const handleSaveDraft = async () => {
    const lastMsg = history[history.length - 1];
    if (!lastMsg || lastMsg.role !== "assistant") return;
    
    if (!saveName.trim()) return alert("Please enter a filename.");

    setIsSaving(true);
    try {
      // Reusing the Scene Edit endpoint to create a new file
      await axios.post(`${API_URL}/scene/save/${profile}`, {
        filename: saveName.endsWith(".txt") ? saveName : `${saveName}.txt`,
        content: lastMsg.content
      });
      alert(`Saved "${saveName}" to Scene Creator!`);
      setSaveName("");
    } catch (err) {
      alert("Save failed: " + err.message);
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
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '10px', fontSize: '20px' }}>
          <Bot size={24} color="#3b82f6" /> Co-Author Chat
        </h2>
        <button 
          onClick={handleClear}
          style={{ background: 'transparent', border: '1px solid #333', color: '#666', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}
        >
          <Trash2 size={14} /> Clear History
        </button>
      </div>

      {/* CHAT WINDOW */}
      <div style={{ flex: 1, overflowY: 'auto', paddingRight: '10px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {history.length === 0 && (
          <div style={{ textAlign: 'center', color: '#444', marginTop: '50px' }}>
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
              background: msg.role === 'user' ? '#333' : '#3b82f6' 
            }}>
              {msg.role === 'user' ? <User size={18} /> : <Bot size={18} color="white" />}
            </div>

            {/* BUBBLE */}
            <div style={{ 
              background: msg.role === 'user' ? '#1f1f1f' : '#111', 
              border: '1px solid #333',
              borderRadius: '8px', padding: '15px', maxWidth: '80%',
              boxShadow: '0 2px 10px rgba(0,0,0,0.2)'
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
            <div style={{ display: 'flex', alignItems: 'center', color: '#666', fontSize: '14px', gap: '10px' }}>
              <Loader2 className="spin" size={16} /> Consulting Knowledge Base...
            </div>
          </div>
        )}

        <div ref={scrollRef} />
      </div>

      {/* SAVE DRAFT TOOL */}
      {history.length > 0 && history[history.length - 1].role === "assistant" && !loading && (
        <div style={{ margin: '10px 0', padding: '10px', background: '#111', border: '1px solid #333', borderRadius: '6px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <FileText size={16} color="#3b82f6" />
          <span style={{ fontSize: '13px', color: '#888', fontWeight: 'bold' }}>Save Last Response as Draft:</span>
          <input 
            value={saveName} 
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="Draft_Filename"
            style={{ background: '#222', border: '1px solid #444', color: '#fff', padding: '6px', borderRadius: '4px', fontSize: '13px', flex: 1 }}
          />
          <button 
            onClick={handleSaveDraft}
            disabled={isSaving}
            style={{ background: '#22c55e', color: '#fff', border: 'none', padding: '6px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}
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
          placeholder="Type your query here... (Shift+Enter for new line)"
          style={{
            width: '100%', background: '#1a1a1a', border: '1px solid #333', borderRadius: '8px',
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
            color: loading ? '#666' : 'white', 
            border: 'none', borderRadius: '6px', width: '40px', height: '40px',
            cursor: loading ? 'default' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center'
          }}
        >
          <Send size={18} />
        </button>
      </div>

    </div>
  );
}