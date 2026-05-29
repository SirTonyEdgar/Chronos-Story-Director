import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  Send, Trash2, Bot, User, Loader2, Zap, Plus, Lock, Unlock,
  ChevronDown, ChevronRight, FileText, Upload, Search, X,
  BookOpen, Sparkles, ClipboardList, AlertTriangle, Check,
  MoreVertical, Edit2, RefreshCw
} from 'lucide-react';
import { API_URL } from './config';
import { toast, confirm } from './components/Notifications';

const MODES = [
  { id: 'free', label: 'Free Chat', description: 'Open conversation about your story' },
  { id: 'brainstorm', label: 'Brainstorm', description: 'Collaborative ideation with options and canon locking' },
  { id: 'scene_repair', label: 'Scene Repair', description: 'Fix pacing, voice, continuity in a specific scene' },
  { id: 'canon_work', label: 'Canon/Lore Work', description: 'Establish and verify story facts with contradiction checking' },
];

export default function CoAuthorChat({ profile }) {
  // --- SESSION STATE ---
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [newSessionName, setNewSessionName] = useState('');
  const [renamingSession, setRenamingSession] = useState(null);
  const [renameValue, setRenameValue] = useState('');

  // --- CHAT STATE ---
  const [history, setHistory] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('free');
  const [timeline, setTimeline] = useState('');
  const [availableTimelines, setAvailableTimelines] = useState([]);

  // --- ATTACHMENT STATE ---
  const [attachments, setAttachments] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showAttachPanel, setShowAttachPanel] = useState(false);
  const fileInputRef = useRef(null);

  // --- LOCKED CANON STATE ---
  const [lockedItems, setLockedItems] = useState([]);
  const [showLockedPanel, setShowLockedPanel] = useState(false);
  const [lockedMessageIds, setLockedMessageIds] = useState(new Set());

  // --- PROPOSALS STATE ---
  const [proposals, setProposals] = useState([]);
  const [showProposalsPanel, setShowProposalsPanel] = useState(false);
  const [extractingFor, setExtractingFor] = useState(null);
  const [applyModal, setApplyModal] = useState(null);

  // --- CONTRADICTION STATE ---
  const [contradictions, setContradictions] = useState([]);
  const [showContradictions, setShowContradictions] = useState(false);
  const [checkingContradictions, setCheckingContradictions] = useState(false);

  // --- SUMMARY STATE ---
  const [sessionSummary, setSessionSummary] = useState('');
  const [generatingSummary, setGeneratingSummary] = useState(false);
  const [showSummary, setShowSummary] = useState(false);

  const scrollRef = useRef(null);

  useEffect(() => {
    if (profile) {
      loadSessions();
      fetchTimelines();
    }
  }, [profile]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history, loading]);

  useEffect(() => {
    if (activeSession) {
      loadSessionHistory(activeSession.id);
      loadLockedItems(activeSession.id);
      loadProposals(activeSession.id);
    }
  }, [activeSession]);

  // --- DATA LOADERS ---

  const loadSessions = async () => {
    try {
      const res = await axios.get(`${API_URL}/chat/sessions/${profile}`);
      setSessions(res.data || []);
    } catch (err) { console.error(err); }
  };

  const loadSessionHistory = async (sessionId) => {
    try {
      const res = await axios.get(`${API_URL}/chat/history/${profile}/${sessionId}`);
      setHistory(res.data || []);
    } catch (err) { console.error(err); }
  };

  const loadLockedItems = async (sessionId) => {
    try {
      const res = await axios.get(`${API_URL}/chat/locked/${profile}/${sessionId}`);
      const items = res.data || [];
      setLockedItems(items);
      setLockedMessageIds(new Set(items.map(i => i.message_index)));
    } catch (err) { console.error(err); }
  };

  const loadProposals = async (sessionId) => {
    try {
      const res = await axios.get(`${API_URL}/chat/proposals/${profile}/${sessionId}`);
      setProposals(res.data || []);
    } catch (err) { console.error(err); }
  };

  const fetchTimelines = async () => {
    try {
      const res = await axios.get(`${API_URL}/state/${profile}`);
      setAvailableTimelines(res.data.Timelines || []);
    } catch (err) { console.error(err); }
  };

  // --- SESSION MANAGEMENT ---

  const handleCreateSession = async () => {
    const name = newSessionName.trim() || 'New Session';
    try {
      const res = await axios.post(`${API_URL}/chat/sessions/${profile}`, { name, mode });
      const newSession = { id: res.data.id, name, mode, created_at: new Date().toISOString() };
      setSessions(prev => [newSession, ...prev]);
      setActiveSession(newSession);
      setHistory([]);
      setLockedItems([]);
      setProposals([]);
      setContradictions([]);
      setSessionSummary('');
      setNewSessionName('');
      setIsCreatingSession(false);
    } catch (err) {
      toast('Failed to create session.', 'error');
    }
  };

  const handleDeleteSession = async (sessionId) => {
    const ok = await confirm('Delete this session and all its messages?', {
      title: 'Delete Session', confirmLabel: 'Delete', danger: true
    });
    if (!ok) return;
    try {
      await axios.delete(`${API_URL}/chat/sessions/${profile}/${sessionId}`);
      setSessions(prev => prev.filter(s => s.id !== sessionId));
      if (activeSession?.id === sessionId) {
        setActiveSession(null);
        setHistory([]);
      }
    } catch (err) {
      toast('Failed to delete session.', 'error');
    }
  };

  const handleRenameSession = async (sessionId) => {
    if (!renameValue.trim()) return;
    try {
      await axios.post(`${API_URL}/chat/sessions/${profile}/${sessionId}/rename`, { name: renameValue });
      setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, name: renameValue } : s));
      if (activeSession?.id === sessionId) setActiveSession(prev => ({ ...prev, name: renameValue }));
      setRenamingSession(null);
    } catch (err) {
      toast('Failed to rename.', 'error');
    }
  };

  // --- ATTACHMENT MANAGEMENT ---

  const handleSearch = async (q) => {
    setSearchQuery(q);
    if (!q.trim() || q.length < 2) { setSearchResults([]); return; }
    setIsSearching(true);
    try {
      const res = await axios.get(`${API_URL}/knowledge/search/${profile}?q=${encodeURIComponent(q)}`);
      setSearchResults(res.data || []);
    } catch (err) { console.error(err); }
    finally { setIsSearching(false); }
  };

  const addAttachmentFromSearch = (result) => {
    if (attachments.find(a => a.id === result.id)) return;
    setAttachments(prev => [...prev, {
      id: result.id,
      name: result.name,
      type: result.type,
      content: result.content,
      source: 'kb'
    }]);
    setSearchQuery('');
    setSearchResults([]);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    toast(`Loading ${file.name}...`, 'info');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await axios.post(
        `${API_URL}/knowledge/import_file/${profile}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setAttachments(prev => [...prev, {
        id: `upload_${Date.now()}`,
        name: file.name,
        type: 'External',
        content: res.data.text,
        source: 'upload'
      }]);
      toast(`${file.name} attached.`, 'success');
    } catch (err) {
      toast('Upload failed: ' + (err.response?.data?.detail || err.message), 'error');
    }
    e.target.value = null;
  };

  const removeAttachment = (id) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  };

  // --- CHAT ---

  const handleSend = async () => {
    if (!input.trim() || !activeSession) return;
    const userMsg = { role: 'user', content: input, id: Date.now() };
    setHistory(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    const attachedContent = attachments.map(a => `[${a.name}]\n${a.content}`).join('\n\n---\n\n');
    const attachedFilename = attachments.map(a => a.name).join(', ');

    try {
      const res = await axios.post(`${API_URL}/chat/query/${profile}`, {
        prompt: userMsg.content,
        timeline,
        mode,
        session_id: activeSession.id,
        attached_content: attachedContent,
        attached_filename: attachedFilename
      });
      const aiMsg = { role: 'assistant', content: res.data.response, id: Date.now() + 1 };
      setHistory(prev => [...prev, aiMsg]);
    } catch (err) {
      setHistory(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ Error: ' + (err.response?.data?.detail || err.message),
        id: Date.now() + 1
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  // --- LOCK / CANON ---

  const handleLockMessage = async (messageIndex, content) => {
    if (!activeSession) return;
    if (lockedMessageIds.has(messageIndex)) {
      const lockItem = lockedItems.find(l => l.message_index === messageIndex);
      if (lockItem) {
        await axios.delete(`${API_URL}/chat/lock/${profile}/${lockItem.id}`);
        setLockedItems(prev => prev.filter(l => l.message_index !== messageIndex));
        setLockedMessageIds(prev => { const s = new Set(prev); s.delete(messageIndex); return s; });
      }
    } else {
      try {
        const res = await axios.post(`${API_URL}/chat/lock/${profile}`, {
          session_id: activeSession.id,
          message_index: messageIndex,
          content
        });
        const newLock = { id: res.data.id, message_index: messageIndex, content };
        setLockedItems(prev => [...prev, newLock]);
        setLockedMessageIds(prev => new Set([...prev, messageIndex]));
        toast('Locked as canon.', 'success');
      } catch (err) {
        toast('Failed to lock.', 'error');
      }
    }
  };

  // --- PROPOSALS ---

  const handleExtractProposals = async (messageIndex, content) => {
    if (!activeSession) return;
    setExtractingFor(messageIndex);
    try {
      const res = await axios.post(`${API_URL}/chat/proposals/extract/${profile}`, {
        response_text: content,
        session_id: activeSession.id
      });
      const newProposals = res.data.proposals || [];
      if (newProposals.length === 0) {
        toast('No concrete proposals found in this response.', 'info');
      } else {
        setProposals(prev => [...prev, ...newProposals]);
        setShowProposalsPanel(true);
        toast(`${newProposals.length} proposal${newProposals.length > 1 ? 's' : ''} extracted.`, 'success');
      }
    } catch (err) {
      toast('Extraction failed.', 'error');
    } finally {
      setExtractingFor(null);
    }
  };

  const handleApplyProposal = (proposal) => {
    setApplyModal(proposal);
  };

  const handleConfirmApply = async (proposal, targetType, targetTitle) => {
    try {
      if (targetType === 'New') {
        await axios.post(`${API_URL}/knowledge/create/${profile}`, {
          name: targetTitle || 'From Co-Author Session',
          content: proposal.content,
          category: proposal.target_type === 'New' ? 'Fact' : proposal.target_type,
          timeline: ''
        });
        toast('Saved as new KB entry.', 'success');
      }
      await axios.post(`${API_URL}/chat/proposals/${profile}/${proposal.id}/status`, { status: 'applied' });
      setProposals(prev => prev.map(p => p.id === proposal.id ? { ...p, status: 'applied' } : p));
      setApplyModal(null);
    } catch (err) {
      toast('Apply failed: ' + err.message, 'error');
    }
  };

  const handleRejectProposal = async (proposalId) => {
    try {
      await axios.post(`${API_URL}/chat/proposals/${profile}/${proposalId}/status`, { status: 'rejected' });
      setProposals(prev => prev.map(p => p.id === proposalId ? { ...p, status: 'rejected' } : p));
    } catch (err) {
      toast('Failed to reject.', 'error');
    }
  };

  // --- CONTRADICTIONS ---

  const handleCheckContradictions = async () => {
    if (attachments.length === 0) {
      toast('Attach a file first to check for contradictions.', 'warning');
      return;
    }
    setCheckingContradictions(true);
    setContradictions([]);
    try {
      const combined = attachments.map(a => a.content).join('\n\n');
      const filename = attachments.map(a => a.name).join(', ');
      const res = await axios.post(`${API_URL}/chat/contradictions/${profile}`, {
        content: combined,
        filename
      });
      const found = res.data.contradictions || [];
      setContradictions(found);
      setShowContradictions(true);
      if (found.length === 0) toast('No contradictions found.', 'success');
      else toast(`${found.length} potential conflict${found.length > 1 ? 's' : ''} found.`, 'warning');
    } catch (err) {
      toast('Check failed: ' + err.message, 'error');
    } finally {
      setCheckingContradictions(false);
    }
  };

  // --- SUMMARY ---

  const handleGenerateSummary = async () => {
    if (!activeSession) return;
    setGeneratingSummary(true);
    try {
      const res = await axios.post(`${API_URL}/chat/summary/${profile}/${activeSession.id}`);
      setSessionSummary(res.data.summary);
      setShowSummary(true);
    } catch (err) {
      toast('Summary failed.', 'error');
    } finally {
      setGeneratingSummary(false);
    }
  };

  // --- RENDER ---

  const pendingProposals = proposals.filter(p => p.status === 'pending');
  const currentMode = MODES.find(m => m.id === mode);

  return (
    <div style={{ display: 'flex', height: '100%', background: '#09090b' }}>

      {/* --- LEFT SIDEBAR: SESSIONS --- */}
      <div style={{ width: '240px', minWidth: '240px', borderRight: '1px solid #1a1a1a', display: 'flex', flexDirection: 'column', background: '#111' }}>

        <div style={{ padding: '16px', borderBottom: '1px solid #1a1a1a' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
            <span style={{ fontSize: '12px', color: '#71717a', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Sessions</span>
            <button
              onClick={() => setIsCreatingSession(true)}
              style={{ background: 'transparent', border: 'none', color: '#3b82f6', cursor: 'pointer', padding: '2px' }}
              title="New Session"
            >
              <Plus size={16} />
            </button>
          </div>

          {isCreatingSession && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <input
                value={newSessionName}
                onChange={e => setNewSessionName(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleCreateSession(); if (e.key === 'Escape') setIsCreatingSession(false); }}
                placeholder="Session name..."
                autoFocus
                style={{ background: '#18181b', border: '1px solid #3b82f6', color: '#fff', padding: '6px 8px', borderRadius: '4px', fontSize: '12px', outline: 'none', width: '100%', boxSizing: 'border-box' }}
              />
              <div style={{ display: 'flex', gap: '4px' }}>
                <button onClick={handleCreateSession} style={{ flex: 1, background: '#3b82f6', border: 'none', color: '#fff', padding: '5px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' }}>Create</button>
                <button onClick={() => setIsCreatingSession(false)} style={{ flex: 1, background: '#27272a', border: 'none', color: '#a1a1aa', padding: '5px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' }}>Cancel</button>
              </div>
            </div>
          )}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
          {sessions.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: '#3f3f46', fontSize: '12px', fontStyle: 'italic' }}>
              No sessions yet. Create one to start.
            </div>
          )}
          {sessions.map(session => (
            <div
              key={session.id}
              onClick={() => { setActiveSession(session); setMode(session.mode || 'free'); }}
              style={{
                padding: '10px', borderRadius: '6px', cursor: 'pointer', marginBottom: '2px',
                background: activeSession?.id === session.id ? '#1e3a5f' : 'transparent',
                border: activeSession?.id === session.id ? '1px solid #1d4ed8' : '1px solid transparent',
                position: 'relative'
              }}
            >
              {renamingSession === session.id ? (
                <input
                  value={renameValue}
                  onChange={e => setRenameValue(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleRenameSession(session.id); if (e.key === 'Escape') setRenamingSession(null); }}
                  onClick={e => e.stopPropagation()}
                  autoFocus
                  style={{ background: '#18181b', border: '1px solid #3b82f6', color: '#fff', padding: '4px 6px', borderRadius: '4px', fontSize: '12px', outline: 'none', width: '100%', boxSizing: 'border-box' }}
                />
              ) : (
                <>
                  <div style={{ fontSize: '13px', color: activeSession?.id === session.id ? '#93c5fd' : '#e4e4e7', fontWeight: '500', marginBottom: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', paddingRight: '20px' }}>
                    {session.name}
                  </div>
                  <div style={{ fontSize: '10px', color: '#52525b' }}>
                    {MODES.find(m => m.id === session.mode)?.label || 'Free Chat'}
                  </div>
                  <div style={{ position: 'absolute', top: '8px', right: '6px', display: 'flex', gap: '2px' }}>
                    <button
                      onClick={e => { e.stopPropagation(); setRenamingSession(session.id); setRenameValue(session.name); }}
                      style={{ background: 'none', border: 'none', color: '#52525b', cursor: 'pointer', padding: '2px' }}
                    ><Edit2 size={11} /></button>
                    <button
                      onClick={e => { e.stopPropagation(); handleDeleteSession(session.id); }}
                      style={{ background: 'none', border: 'none', color: '#52525b', cursor: 'pointer', padding: '2px' }}
                    ><Trash2 size={11} /></button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* --- MAIN CHAT AREA --- */}
      {!activeSession ? (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '16px', color: '#3f3f46' }}>
          <Bot size={48} style={{ opacity: 0.2 }} />
          <p style={{ margin: 0 }}>Select a session or create a new one</p>
          <button
            onClick={() => setIsCreatingSession(true)}
            style={{ background: '#3b82f6', border: 'none', color: '#fff', padding: '10px 20px', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', fontWeight: '600' }}
          >
            <Plus size={14} style={{ marginRight: '6px' }} />New Session
          </button>
        </div>
      ) : (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* CHAT HEADER */}
          <div style={{ padding: '12px 20px', borderBottom: '1px solid #1a1a1a', background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Bot size={18} color="#3b82f6" />
              <span style={{ fontWeight: '600', color: '#e4e4e7', fontSize: '14px' }}>{activeSession.name}</span>
              <select
                value={mode}
                onChange={e => setMode(e.target.value)}
                style={{ background: '#18181b', border: '1px solid #3f3f46', color: '#a1a1aa', padding: '4px 8px', borderRadius: '4px', fontSize: '11px', outline: 'none', cursor: 'pointer' }}
              >
                {MODES.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
              </select>
              {availableTimelines.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(168,85,247,0.1)', border: '1px solid #a855f7', padding: '3px 8px', borderRadius: '4px' }}>
                  <Zap size={12} color="#a855f7" />
                  <select
                    value={timeline}
                    onChange={e => setTimeline(e.target.value)}
                    style={{ background: 'transparent', border: 'none', color: '#e4e4e7', outline: 'none', fontSize: '11px', cursor: 'pointer' }}
                  >
                    <option value="">Universal</option>
                    {availableTimelines.map((tl, i) => <option key={i} value={tl.Name}>{tl.Name}</option>)}
                  </select>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
              <button
                onClick={() => setShowLockedPanel(!showLockedPanel)}
                style={{ background: lockedItems.length > 0 ? 'rgba(234,179,8,0.1)' : 'transparent', border: '1px solid ' + (lockedItems.length > 0 ? '#eab308' : '#27272a'), color: lockedItems.length > 0 ? '#eab308' : '#52525b', padding: '4px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px' }}
                title="Locked Canon"
              >
                <Lock size={12} /> {lockedItems.length > 0 ? `${lockedItems.length} Locked` : 'Canon'}
              </button>
              <button
                onClick={() => setShowProposalsPanel(!showProposalsPanel)}
                style={{ background: pendingProposals.length > 0 ? 'rgba(59,130,246,0.1)' : 'transparent', border: '1px solid ' + (pendingProposals.length > 0 ? '#3b82f6' : '#27272a'), color: pendingProposals.length > 0 ? '#60a5fa' : '#52525b', padding: '4px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px' }}
                title="Proposed Changes"
              >
                <ClipboardList size={12} /> {pendingProposals.length > 0 ? `${pendingProposals.length} Pending` : 'Proposals'}
              </button>
              <button
                onClick={handleGenerateSummary}
                disabled={generatingSummary}
                style={{ background: 'transparent', border: '1px solid #27272a', color: '#52525b', padding: '4px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px' }}
                title="Generate Session Summary"
              >
                <RefreshCw size={12} style={{ animation: generatingSummary ? 'spin 1s linear infinite' : 'none' }} />
                Summary
              </button>
            </div>
          </div>

          {/* LOCKED CANON PANEL */}
          {showLockedPanel && lockedItems.length > 0 && (
            <div style={{ background: '#1a1400', borderBottom: '1px solid #854d0e', padding: '12px 20px' }}>
              <div style={{ fontSize: '11px', color: '#eab308', fontWeight: '700', textTransform: 'uppercase', marginBottom: '8px' }}>🔒 Locked Canon</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '150px', overflowY: 'auto' }}>
                {lockedItems.map((item, i) => (
                  <div key={item.id} style={{ fontSize: '12px', color: '#fef3c7', background: 'rgba(234,179,8,0.05)', border: '1px solid #78350f', borderRadius: '4px', padding: '6px 10px', lineHeight: '1.5' }}>
                    {item.content.slice(0, 300)}{item.content.length > 300 ? '...' : ''}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SESSION SUMMARY PANEL */}
          {showSummary && sessionSummary && (
            <div style={{ background: '#0f1f0f', borderBottom: '1px solid #166534', padding: '12px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '11px', color: '#4ade80', fontWeight: '700', textTransform: 'uppercase' }}>Session Summary</span>
                <button onClick={() => setShowSummary(false)} style={{ background: 'none', border: 'none', color: '#52525b', cursor: 'pointer' }}><X size={14} /></button>
              </div>
              <div style={{ fontSize: '12px', color: '#a1a1aa', lineHeight: '1.6', maxHeight: '200px', overflowY: 'auto', whiteSpace: 'pre-wrap' }}>
                {sessionSummary}
              </div>
            </div>
          )}

          {/* CONTRADICTIONS PANEL */}
          {showContradictions && contradictions.length > 0 && (
            <div style={{ background: '#1a0a0a', borderBottom: '1px solid #7f1d1d', padding: '12px 20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '11px', color: '#f87171', fontWeight: '700', textTransform: 'uppercase' }}>⚠️ {contradictions.length} Contradiction{contradictions.length > 1 ? 's' : ''} Found</span>
                <button onClick={() => setShowContradictions(false)} style={{ background: 'none', border: 'none', color: '#52525b', cursor: 'pointer' }}><X size={14} /></button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '150px', overflowY: 'auto' }}>
                {contradictions.map((c, i) => (
                  <div key={i} style={{ fontSize: '12px', color: '#fca5a5', background: 'rgba(239,68,68,0.05)', border: '1px solid #7f1d1d', borderRadius: '4px', padding: '6px 10px' }}>
                    <div style={{ fontWeight: '600', marginBottom: '2px' }}>{c.issue}</div>
                    <div style={{ color: '#a1a1aa' }}>Attached says: {c.attached_says}</div>
                    <div style={{ color: '#a1a1aa' }}>KB says: {c.kb_says} [{c.kb_entry}]</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* PROPOSALS PANEL */}
          {showProposalsPanel && proposals.length > 0 && (
            <div style={{ background: '#0a0f1a', borderBottom: '1px solid #1e3a5f', padding: '12px 20px' }}>
              <div style={{ fontSize: '11px', color: '#60a5fa', fontWeight: '700', textTransform: 'uppercase', marginBottom: '8px' }}>📋 Proposed Changes</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '200px', overflowY: 'auto' }}>
                {proposals.map(p => (
                  <div key={p.id} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', padding: '8px', background: p.status === 'applied' ? 'rgba(34,197,94,0.05)' : p.status === 'rejected' ? 'rgba(239,68,68,0.05)' : 'rgba(59,130,246,0.05)', border: '1px solid ' + (p.status === 'applied' ? '#166534' : p.status === 'rejected' ? '#7f1d1d' : '#1e3a5f'), borderRadius: '6px' }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '10px', color: '#52525b', marginBottom: '3px' }}>{p.target_type || 'General'}</div>
                      <div style={{ fontSize: '12px', color: '#a1a1aa', lineHeight: '1.4' }}>{p.content.slice(0, 200)}{p.content.length > 200 ? '...' : ''}</div>
                    </div>
                    {p.status === 'pending' && (
                      <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
                        <button onClick={() => handleApplyProposal(p)} style={{ background: 'rgba(34,197,94,0.1)', border: '1px solid #166534', color: '#4ade80', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' }}>Apply</button>
                        <button onClick={() => handleRejectProposal(p.id)} style={{ background: 'transparent', border: '1px solid #27272a', color: '#52525b', padding: '4px 8px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px' }}>Reject</button>
                      </div>
                    )}
                    {p.status !== 'pending' && (
                      <span style={{ fontSize: '10px', color: p.status === 'applied' ? '#4ade80' : '#f87171', flexShrink: 0 }}>{p.status}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* CHAT WINDOW */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {history.length === 0 && (
              <div style={{ textAlign: 'center', color: '#3f3f46', marginTop: '40px' }}>
                <Bot size={40} style={{ opacity: 0.15, marginBottom: '10px' }} />
                <p style={{ margin: 0, fontSize: '13px' }}>{currentMode?.description}</p>
              </div>
            )}

            {history.map((msg, i) => (
              <div key={msg.id || i} style={{ display: 'flex', gap: '12px', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
                <div style={{ width: '28px', height: '28px', borderRadius: '4px', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: msg.role === 'user' ? '#27272a' : '#3b82f6', border: msg.role === 'user' ? '1px solid #3f3f46' : 'none' }}>
                  {msg.role === 'user' ? <User size={16} color="#e4e4e7" /> : <Bot size={16} color="white" />}
                </div>

                <div style={{ maxWidth: '80%', position: 'relative' }}>
                  <div style={{ background: msg.role === 'user' ? '#18181b' : '#0d0d0d', border: '1px solid ' + (lockedMessageIds.has(i) ? '#854d0e' : '#1a1a1a'), borderRadius: '8px', padding: '12px 14px' }}>
                    <div style={{ fontSize: '14px', lineHeight: '1.6', color: '#e4e4e7' }}>
                      {msg.role === 'user' ? msg.content : <ReactMarkdown>{msg.content}</ReactMarkdown>}
                    </div>
                  </div>

                  {msg.role === 'assistant' && (
                    <div style={{ display: 'flex', gap: '6px', marginTop: '4px', justifyContent: 'flex-start' }}>
                      <button
                        onClick={() => handleLockMessage(i, msg.content)}
                        title={lockedMessageIds.has(i) ? 'Unlock' : 'Lock as Canon'}
                        style={{ background: 'none', border: 'none', color: lockedMessageIds.has(i) ? '#eab308' : '#3f3f46', cursor: 'pointer', padding: '2px 6px', borderRadius: '4px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                      >
                        {lockedMessageIds.has(i) ? <Lock size={12} /> : <Unlock size={12} />}
                        {lockedMessageIds.has(i) ? 'Locked' : 'Lock'}
                      </button>
                      <button
                        onClick={() => handleExtractProposals(i, msg.content)}
                        disabled={extractingFor === i}
                        title="Extract Proposals"
                        style={{ background: 'none', border: 'none', color: '#3f3f46', cursor: 'pointer', padding: '2px 6px', borderRadius: '4px', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '4px' }}
                      >
                        <ClipboardList size={12} />
                        {extractingFor === i ? 'Extracting...' : 'Extract'}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div style={{ display: 'flex', gap: '12px' }}>
                <div style={{ width: '28px', height: '28px', borderRadius: '4px', background: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Bot size={16} color="white" />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', color: '#52525b', fontSize: '13px', gap: '8px' }}>
                  <Loader2 size={14} className="spin" /> Thinking...
                </div>
              </div>
            )}

            <div ref={scrollRef} />
          </div>

          {/* ATTACHMENT BAR */}
          {attachments.length > 0 && (
            <div style={{ padding: '8px 16px', borderTop: '1px solid #1a1a1a', background: '#0d0d0d', display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center' }}>
              <span style={{ fontSize: '11px', color: '#52525b' }}>Attached:</span>
              {attachments.map(a => (
                <span key={a.id} style={{ fontSize: '11px', background: '#18181b', border: '1px solid #27272a', color: '#a1a1aa', padding: '3px 8px', borderRadius: '4px', display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <FileText size={10} />
                  {a.name}
                  <button onClick={() => removeAttachment(a.id)} style={{ background: 'none', border: 'none', color: '#52525b', cursor: 'pointer', padding: '0', lineHeight: 1 }}>×</button>
                </span>
              ))}
              {attachments.length > 0 && (
                <button
                  onClick={handleCheckContradictions}
                  disabled={checkingContradictions}
                  style={{ fontSize: '11px', background: 'transparent', border: '1px solid #7f1d1d', color: '#f87171', padding: '3px 8px', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px' }}
                >
                  <AlertTriangle size={10} />
                  {checkingContradictions ? 'Checking...' : 'Check Contradictions'}
                </button>
              )}
            </div>
          )}

          {/* ATTACH PANEL */}
          {showAttachPanel && (
            <div style={{ padding: '12px 16px', borderTop: '1px solid #1a1a1a', background: '#0d0d0d' }}>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '8px', background: '#18181b', border: '1px solid #27272a', borderRadius: '6px', padding: '6px 10px' }}>
                  <Search size={12} color="#52525b" />
                  <input
                    value={searchQuery}
                    onChange={e => handleSearch(e.target.value)}
                    placeholder="Search KB entries..."
                    style={{ background: 'transparent', border: 'none', color: '#fff', outline: 'none', fontSize: '12px', flex: 1 }}
                  />
                  {isSearching && <Loader2 size={12} color="#52525b" />}
                </div>
                <button onClick={() => fileInputRef.current?.click()} style={{ background: '#18181b', border: '1px solid #27272a', color: '#a1a1aa', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <Upload size={12} /> Upload
                </button>
                <input ref={fileInputRef} type="file" accept=".txt,.md,.pdf,.docx" onChange={handleFileUpload} style={{ display: 'none' }} />
              </div>

              {searchResults.length > 0 && (
                <div style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: '6px', maxHeight: '150px', overflowY: 'auto' }}>
                  {searchResults.slice(0, 8).map(r => (
                    <div key={r.id} onClick={() => addAttachmentFromSearch(r)} style={{ padding: '8px 12px', cursor: 'pointer', borderBottom: '1px solid #1a1a1a', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                      onMouseOver={e => e.currentTarget.style.background = '#27272a'}
                      onMouseOut={e => e.currentTarget.style.background = 'transparent'}
                    >
                      <span style={{ fontSize: '12px', color: '#e4e4e7' }}>{r.name}</span>
                      <span style={{ fontSize: '10px', color: '#52525b', background: '#27272a', padding: '2px 6px', borderRadius: '3px' }}>{r.type}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* INPUT AREA */}
          <div style={{ padding: '12px 16px', borderTop: '1px solid #1a1a1a', background: '#111' }}>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <button
                onClick={() => setShowAttachPanel(!showAttachPanel)}
                style={{ background: showAttachPanel ? 'rgba(59,130,246,0.1)' : 'transparent', border: '1px solid ' + (showAttachPanel ? '#1d4ed8' : '#27272a'), color: showAttachPanel ? '#60a5fa' : '#52525b', padding: '4px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '5px' }}
              >
                <FileText size={12} /> Attach
                {attachments.length > 0 && <span style={{ background: '#3b82f6', color: '#fff', borderRadius: '10px', padding: '0 5px', fontSize: '10px' }}>{attachments.length}</span>}
              </button>
            </div>

            <div style={{ position: 'relative' }}>
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={activeSession ? `${currentMode?.label} mode — Shift+Enter for new line` : 'Select a session first...'}
                disabled={!activeSession}
                style={{ width: '100%', background: '#09090b', border: '1px solid #3f3f46', borderRadius: '8px', color: '#fff', padding: '12px', paddingRight: '50px', fontSize: '14px', resize: 'none', height: '60px', outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box' }}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim() || !activeSession}
                style={{ position: 'absolute', right: '10px', top: '10px', background: loading || !input.trim() ? 'transparent' : '#3b82f6', color: loading || !input.trim() ? '#52525b' : 'white', border: 'none', borderRadius: '6px', width: '36px', height: '36px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* APPLY MODAL */}
      {applyModal && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.8)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: '#111', border: '1px solid #27272a', borderRadius: '10px', width: '500px', maxWidth: '90vw', padding: '24px' }}>
            <h3 style={{ margin: '0 0 16px 0', color: '#fff', fontSize: '16px' }}>Apply Proposal</h3>
            <div style={{ background: '#0d0d0d', border: '1px solid #1a1a1a', borderRadius: '6px', padding: '12px', marginBottom: '16px', fontSize: '13px', color: '#a1a1aa', lineHeight: '1.5', maxHeight: '150px', overflowY: 'auto' }}>
              {applyModal.content}
            </div>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ fontSize: '11px', color: '#71717a', display: 'block', marginBottom: '6px' }}>SAVE AS NEW KB ENTRY — TITLE</label>
              <input
                id="apply-title"
                placeholder="Entry title..."
                defaultValue={applyModal.summary || ''}
                style={{ width: '100%', background: '#18181b', border: '1px solid #27272a', color: '#fff', padding: '8px 10px', borderRadius: '6px', fontSize: '13px', outline: 'none', boxSizing: 'border-box' }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
              <button onClick={() => setApplyModal(null)} style={{ padding: '8px 16px', background: 'transparent', border: '1px solid #3f3f46', color: '#a1a1aa', borderRadius: '6px', cursor: 'pointer' }}>Cancel</button>
              <button
                onClick={() => {
                  const title = document.getElementById('apply-title')?.value || applyModal.summary;
                  handleConfirmApply(applyModal, 'New', title);
                }}
                style={{ padding: '8px 16px', background: '#22c55e', border: 'none', color: '#000', borderRadius: '6px', cursor: 'pointer', fontWeight: '700' }}
              >
                Save to Knowledge Base
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .spin { animation: spin 1s linear infinite; }
      `}</style>
    </div>
  );
}
