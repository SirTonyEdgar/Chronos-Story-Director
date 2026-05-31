import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Plus, Trash2, Wand2, ChevronDown } from 'lucide-react';
import { API_URL } from '../config';
import { toast, confirm } from '../components/Notifications';

export default function ReservedNamesTab({ profile }) {
  const [names, setNames] = useState([]);
  const [nameInput, setNameInput] = useState("");
  const [noteInput, setNoteInput] = useState("");
  const [isExtracting, setIsExtracting] = useState(false);
  const [sceneFiles, setSceneFiles] = useState([]);
  const [extractTarget, setExtractTarget] = useState("all");
  const [extractedNames, setExtractedNames] = useState([]);
  const [selectedForAdd, setSelectedForAdd] = useState([]);
  const [showExtractPanel, setShowExtractPanel] = useState(false);

  useEffect(() => {
    if (profile) {
      fetchNames();
      fetchSceneFiles();
    }
  }, [profile]);

  const fetchNames = async () => {
    try {
      const res = await axios.get(`${API_URL}/reserved_names/${profile}`);
      setNames(res.data || []);
    } catch (err) {
      console.error("Failed to load reserved names:", err);
    }
  };

  const fetchSceneFiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/files/${profile}`);
      setSceneFiles(res.data || []);
    } catch (err) { /* silent */ }
  };

  const handleAdd = async () => {
    if (!nameInput.trim()) return;
    try {
      const res = await axios.post(`${API_URL}/reserved_names/${profile}`, {
        name: nameInput.trim(),
        note: noteInput.trim()
      });
      setNames(prev => [...prev, res.data]);
      setNameInput("");
      setNoteInput("");
    } catch (err) {
      toast("Failed to add name: " + err.message, "error");
    }
  };

  const handleDelete = async (id) => {
    const ok = await confirm("Remove this name from the reserved list?", { title: "Remove Name", confirmLabel: "Remove", danger: true });
    if (!ok) return;
    try {
      await axios.delete(`${API_URL}/reserved_names/${profile}/${id}`);
      setNames(prev => prev.filter(n => n.id !== id));
    } catch (err) {
      toast("Failed to delete: " + err.message, "error");
    }
  };

  const handleNoteBlur = async (id, note) => {
    try {
      await axios.post(`${API_URL}/reserved_names/${profile}/${id}/note`, { note });
      setNames(prev => prev.map(n => n.id === id ? { ...n, note } : n));
    } catch (err) {
      toast("Failed to save note: " + err.message, "error");
    }
  };

  const handleExtract = async () => {
    setIsExtracting(true);
    setExtractedNames([]);
    setSelectedForAdd([]);
    try {
      const filenames = extractTarget === "all"
        ? sceneFiles
        : [extractTarget];
      const res = await axios.post(`${API_URL}/reserved_names/${profile}/extract`, { filenames });
      const existing = new Set(names.map(n => n.name.toLowerCase()));
      const fresh = (res.data.names || []).filter(n => !existing.has(n.toLowerCase()));
      setExtractedNames(fresh);
      if (fresh.length === 0) toast("No new names found.", "info");
    } catch (err) {
      toast("Extraction failed: " + err.message, "error");
    } finally {
      setIsExtracting(false);
    }
  };

  const handleAddSelected = async () => {
    for (const name of selectedForAdd) {
      try {
        const res = await axios.post(`${API_URL}/reserved_names/${profile}`, { name, note: "" });
        setNames(prev => [...prev, res.data]);
      } catch (err) { /* skip */ }
    }
    setExtractedNames([]);
    setSelectedForAdd([]);
    toast(`${selectedForAdd.length} name${selectedForAdd.length !== 1 ? 's' : ''} added.`, "success");
  };

  return (
    <div style={{ display: 'flex', height: '100%', gap: '20px', border: '1px solid #333', borderRadius: '8px', overflow: 'hidden', background: '#09090b' }}>

      {/* LEFT — registry */}
      <div style={{ width: '340px', minWidth: '340px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #27272a', background: '#111' }}>
        <div style={{ padding: '16px', borderBottom: '1px solid #222' }}>
          <div style={{ fontSize: '12px', color: '#f97316', background: 'rgba(249,115,22,0.08)', border: '1px solid rgba(249,115,22,0.2)', padding: '10px', borderRadius: '6px', lineHeight: '1.5', marginBottom: '12px' }}>
            🚫 <strong>Reserved Names</strong> — names and usernames the AI will never assign to new characters. Add manually or extract from your scenes. The AI reads this list before generating and avoids reusing anything on it.
          </div>

          {/* Add input */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <input
              value={nameInput}
              onChange={e => setNameInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleAdd(); }}
              placeholder="Name or username..."
              style={{ background: '#18181b', border: '1px solid #27272a', color: '#e4e4e7', padding: '8px 10px', borderRadius: '6px', fontSize: '13px', outline: 'none' }}
            />
            <div style={{ display: 'flex', gap: '6px' }}>
              <input
                value={noteInput}
                onChange={e => setNoteInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') handleAdd(); }}
                placeholder="Note (optional, e.g. bartender Scene 3)..."
                style={{ flex: 1, background: '#18181b', border: '1px solid #27272a', color: '#e4e4e7', padding: '8px 10px', borderRadius: '6px', fontSize: '12px', outline: 'none' }}
              />
              <button
                onClick={handleAdd}
                style={{ padding: '8px 12px', background: '#f97316', border: 'none', color: '#fff', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: '600' }}
              >
                <Plus size={14} /> Add
              </button>
            </div>
          </div>
        </div>

        {/* Name list */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
          {names.length === 0 && (
            <div style={{ color: '#3f3f46', fontSize: '13px', fontStyle: 'italic', textAlign: 'center', marginTop: '30px' }}>
              No reserved names yet.
            </div>
          )}
          {names.map(n => (
            <div key={n.id} style={{ background: '#18181b', border: '1px solid #27272a', borderRadius: '6px', padding: '8px 10px', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '13px', color: '#e4e4e7', fontWeight: '600', flex: 1 }}>{n.name}</span>
                <button
                  onClick={() => handleDelete(n.id)}
                  style={{ background: 'none', border: 'none', color: '#52525b', cursor: 'pointer', padding: '2px' }}
                >
                  <Trash2 size={13} />
                </button>
              </div>
              <input
                defaultValue={n.note}
                onBlur={e => { if (e.target.value !== n.note) handleNoteBlur(n.id, e.target.value); }}
                placeholder="Add a note..."
                style={{ width: '100%', background: 'transparent', border: 'none', color: '#52525b', fontSize: '11px', outline: 'none', marginTop: '4px', boxSizing: 'border-box' }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* RIGHT — extraction panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '20px', overflowY: 'auto' }}>
        <div style={{ marginBottom: '16px' }}>
          <button
            onClick={() => setShowExtractPanel(!showExtractPanel)}
            style={{ display: 'flex', alignItems: 'center', gap: '8px', background: '#18181b', border: '1px solid #27272a', color: '#e4e4e7', padding: '10px 16px', borderRadius: '6px', cursor: 'pointer', fontSize: '13px', fontWeight: '600' }}
          >
            <Wand2 size={15} color="#a855f7" />
            Extract Names from Scenes
            <ChevronDown size={14} style={{ transform: showExtractPanel ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
          </button>
        </div>

        {showExtractPanel && (
          <div style={{ background: '#111', border: '1px solid #27272a', borderRadius: '8px', padding: '16px', marginBottom: '16px' }}>
            <p style={{ fontSize: '12px', color: '#71717a', margin: '0 0 12px 0', lineHeight: '1.5' }}>
              Pick a scene to extract from, or extract from all scenes at once. Review the results and select which names to add to your reserved list.
            </p>

            <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
              <select
                value={extractTarget}
                onChange={e => setExtractTarget(e.target.value)}
                style={{ flex: 1, background: '#18181b', border: '1px solid #27272a', color: '#e4e4e7', padding: '8px 10px', borderRadius: '6px', fontSize: '12px', outline: 'none' }}
              >
                <option value="all">All Scenes</option>
                {sceneFiles.map((f, i) => (
                  <option key={i} value={f}>{f}</option>
                ))}
              </select>
              <button
                onClick={handleExtract}
                disabled={isExtracting}
                style={{ padding: '8px 16px', background: '#a855f7', border: 'none', color: '#fff', borderRadius: '6px', cursor: isExtracting ? 'default' : 'pointer', fontSize: '12px', fontWeight: '600', opacity: isExtracting ? 0.6 : 1 }}
              >
                {isExtracting ? "Extracting..." : "Extract"}
              </button>
            </div>

            {extractedNames.length > 0 && (
              <>
                <div style={{ fontSize: '11px', color: '#52525b', marginBottom: '8px' }}>
                  {extractedNames.length} new names found — select which to add:
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '12px' }}>
                  {extractedNames.map((name, i) => {
                    const selected = selectedForAdd.includes(name);
                    return (
                      <button
                        key={i}
                        onClick={() => setSelectedForAdd(prev =>
                          selected ? prev.filter(n => n !== name) : [...prev, name]
                        )}
                        style={{
                          padding: '4px 10px', borderRadius: '4px', fontSize: '12px', cursor: 'pointer',
                          background: selected ? 'rgba(168,85,247,0.2)' : '#18181b',
                          border: `1px solid ${selected ? '#a855f7' : '#27272a'}`,
                          color: selected ? '#e4e4e7' : '#71717a',
                          fontWeight: selected ? '600' : '400'
                        }}
                      >
                        {name}
                      </button>
                    );
                  })}
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={() => setSelectedForAdd(extractedNames)}
                    style={{ fontSize: '12px', background: 'none', border: 'none', color: '#52525b', cursor: 'pointer' }}
                  >
                    Select all
                  </button>
                  <button
                    onClick={() => setSelectedForAdd([])}
                    style={{ fontSize: '12px', background: 'none', border: 'none', color: '#52525b', cursor: 'pointer' }}
                  >
                    Clear
                  </button>
                  <button
                    onClick={handleAddSelected}
                    disabled={selectedForAdd.length === 0}
                    style={{ marginLeft: 'auto', padding: '7px 16px', background: '#a855f7', border: 'none', color: '#fff', borderRadius: '6px', cursor: selectedForAdd.length === 0 ? 'default' : 'pointer', fontSize: '12px', fontWeight: '600', opacity: selectedForAdd.length === 0 ? 0.5 : 1 }}
                  >
                    Add {selectedForAdd.length > 0 ? `${selectedForAdd.length} ` : ''}Selected
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        <div style={{ color: '#3f3f46', fontSize: '13px', lineHeight: '1.8' }}>
          <p style={{ marginTop: 0, color: '#52525b' }}>
            Names on this list will <strong style={{ color: '#e4e4e7' }}>never be generated</strong> by the AI for new characters. That's the only thing this does.
          </p>
          <p style={{ color: '#52525b' }}>
            If you want to <strong style={{ color: '#e4e4e7' }}>preserve</strong> a minor character so the AI keeps using them — a recurring bartender, a specific username, a community regular — go to the <strong style={{ color: '#e4e4e7' }}>Reference tab</strong> and add them there instead.
          </p>
          <p style={{ color: '#52525b' }}>
            Use this list for names you want <strong style={{ color: '#e4e4e7' }}>blocked</strong>: names already taken in your story that you don't want accidentally reused for someone new.
          </p>
        </div>
      </div>
    </div>
  );
}