import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  BookOpen, FileText, Download, X, Plus, FileType, 
  CheckCircle2, Settings, ChevronDown, ChevronUp
} from 'lucide-react';
import { API_URL } from './config';
import { toast } from './components/Notifications';

const FONT_OPTIONS = [
  "Times New Roman",
  "Georgia",
  "Garamond",
  "Arial",
  "Calibri",
  "Courier New",
];

export default function Compiler({ profile }) {
  const [availableFiles, setAvailableFiles] = useState([]);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);

  // --- DOCX FORMATTING OPTIONS ---
  const [showDocxOptions, setShowDocxOptions] = useState(false);
  const [fontName, setFontName] = useState("Times New Roman");
  const [fontSize, setFontSize] = useState(12);
  const [marginCm, setMarginCm] = useState(2.54);

  useEffect(() => {
    if (profile) fetchFiles();
  }, [profile]);

  const fetchFiles = async () => {
    try {
      const res = await axios.get(`${API_URL}/files/${profile}`);
      setAvailableFiles(res.data || []);
    } catch (err) { console.error(err); }
  };

  const addToManuscript = (file) => {
    if (!selectedFiles.includes(file)) {
      setSelectedFiles([...selectedFiles, file]);
    }
  };

  const removeFromManuscript = (index) => {
    const newList = [...selectedFiles];
    newList.splice(index, 1);
    setSelectedFiles(newList);
  };

  const handleDownloadText = async () => {
    if (selectedFiles.length === 0) return toast("Select scenes first.", "warning");
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/compiler/compile/${profile}`, { filenames: selectedFiles });
      
      const blob = new Blob([res.data.text], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;

      const now = new Date();
      const timestamp = now.getFullYear() +
        String(now.getMonth() + 1).padStart(2, '0') +
        String(now.getDate()).padStart(2, '0') + "_" +
        String(now.getHours()).padStart(2, '0') +
        String(now.getMinutes()).padStart(2, '0') +
        String(now.getSeconds()).padStart(2, '0');

      link.setAttribute('download', `${profile}_Manuscript_${timestamp}.txt`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      toast("Compile Failed: " + (err.response?.data?.detail || err.message), "error");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadBinary = async (format) => {
    if (selectedFiles.length === 0) return toast("Select scenes first.", "warning");
    setLoading(true);
    try {
      const res = await axios.post(
        `${API_URL}/compiler/export/${profile}/${format}`, 
        { filenames: selectedFiles },
        { responseType: 'blob' } 
      );

      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${profile}_Manuscript.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error(err);
      toast(`Export to ${format.toUpperCase()} failed. Check backend logs for missing libraries.`, "warning");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadDocx = async () => {
    if (selectedFiles.length === 0) return toast("Select scenes first.", "warning");
    setLoading(true);
    try {
      const res = await axios.post(
        `${API_URL}/compiler/export/${profile}/docx`,
        {
          filenames: selectedFiles,
          font_name: fontName,
          font_size: fontSize,
          margin_cm: marginCm,
        },
        { responseType: 'blob' }
      );

      const now = new Date();
      const timestamp = now.getFullYear() +
        String(now.getMonth() + 1).padStart(2, '0') +
        String(now.getDate()).padStart(2, '0') + "_" +
        String(now.getHours()).padStart(2, '0') +
        String(now.getMinutes()).padStart(2, '0') +
        String(now.getSeconds()).padStart(2, '0');

      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `${profile}_Manuscript_${timestamp}.docx`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error(err);
      toast("DOCX export failed. Check backend logs.", "warning");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '30px', boxSizing: 'border-box' }}>
      
      <div style={{ marginBottom: '20px' }}>
        <h2 style={{ margin: 0, fontSize: '24px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <BookOpen size={24} color="#eab308" /> Manuscript Compiler
        </h2>
        <p style={{ margin: '5px 0 0 0', color: '#666', fontSize: '13px' }}>
          Assemble your scenes into distribution-ready formats (TXT, DOCX, PDF, EPUB).
        </p>
      </div>

      <div style={{ flex: 1, display: 'flex', gap: '20px', overflow: 'hidden' }}>
        
        {/* --- LEFT: SOURCE FILES --- */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#111', border: '1px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
          <div style={{ padding: '10px 15px', background: '#1a1a1a', borderBottom: '1px solid #333', fontWeight: 'bold', fontSize: '13px', color: '#888' }}>
            AVAILABLE SCENES ({availableFiles.length})
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
            {availableFiles.map(f => (
              <div 
                key={f} 
                onClick={() => addToManuscript(f)}
                style={{
                  padding: '8px 12px', marginBottom: '4px', borderRadius: '4px', cursor: 'pointer',
                  background: selectedFiles.includes(f) ? '#22c55e20' : 'transparent',
                  color: selectedFiles.includes(f) ? '#22c55e' : '#ccc',
                  fontSize: '13px', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  transition: 'background 0.2s'
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <FileText size={14} /> {f}
                </span>
                {selectedFiles.includes(f) ? <CheckCircle2 size={14} /> : <Plus size={14} style={{ opacity: 0.5 }} />}
              </div>
            ))}
          </div>
        </div>

        {/* --- CENTER DIVIDER --- */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ width: '1px', height: '100%', background: '#333' }} />
        </div>

        {/* --- RIGHT: MANUSCRIPT ORDER --- */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0e0e0e', border: '1px solid #333', borderRadius: '8px', overflow: 'hidden' }}>
          <div style={{ padding: '10px 15px', background: '#1a1a1a', borderBottom: '1px solid #333', fontWeight: 'bold', fontSize: '13px', color: '#eab308' }}>
            MANUSCRIPT ORDER ({selectedFiles.length})
          </div>
          
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
            {selectedFiles.length === 0 ? (
              <div style={{ textAlign: 'center', marginTop: '50px', color: '#444', fontSize: '13px' }}>
                <FileType size={32} style={{ marginBottom: '10px', opacity: 0.2 }} />
                <p>Select scenes from the left to build your book.</p>
              </div>
            ) : (
              selectedFiles.map((f, i) => (
                <div key={`${f}-${i}`} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '8px 12px', background: '#18181b', border: '1px solid #27272a', borderRadius: '4px', marginBottom: '6px' }}>
                  <span style={{ color: '#666', fontSize: '12px', width: '20px' }}>{i + 1}.</span>
                  <span style={{ flex: 1, fontSize: '13px', color: '#eee', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{f}</span>
                  <button onClick={() => removeFromManuscript(i)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#666', padding: '4px' }}>
                    <X size={14} />
                  </button>
                </div>
              ))
            )}
          </div>

          {/* --- DOCX FORMATTING OPTIONS (Collapsible) --- */}
          <div style={{ borderTop: '1px solid #222', background: '#0a0a0a' }}>
            <div 
              onClick={() => setShowDocxOptions(!showDocxOptions)}
              style={{ padding: '10px 15px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#555' }}>
                <Settings size={13} /> DOCX Formatting Options
              </div>
              {showDocxOptions 
                ? <ChevronUp size={14} color="#555" /> 
                : <ChevronDown size={14} color="#555" />
              }
            </div>

            {showDocxOptions && (
              <div style={{ padding: '0 15px 15px 15px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                
                <div>
                  <label style={styles.optionLabel}>Font</label>
                  <select 
                    value={fontName} 
                    onChange={e => setFontName(e.target.value)} 
                    style={styles.optionSelect}
                  >
                    {FONT_OPTIONS.map(f => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>

                <div style={{ display: 'flex', gap: '10px' }}>
                  <div style={{ flex: 1 }}>
                    <label style={styles.optionLabel}>Font Size (pt)</label>
                    <input 
                      type="number" min="8" max="24" 
                      value={fontSize} 
                      onChange={e => setFontSize(Number(e.target.value))} 
                      style={styles.optionInput}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label style={styles.optionLabel}>Margins (cm)</label>
                    <input 
                      type="number" min="1" max="5" step="0.1"
                      value={marginCm} 
                      onChange={e => setMarginCm(Number(e.target.value))} 
                      style={styles.optionInput}
                    />
                  </div>
                </div>

              </div>
            )}
          </div>

          {/* --- EXPORT ACTIONS --- */}
          <div style={{ padding: '15px', borderTop: '1px solid #333', background: '#111', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            <button 
              onClick={handleDownloadText}
              disabled={loading || selectedFiles.length === 0}
              style={{ ...styles.exportBtn, background: '#333', border: '1px solid #444', color: '#fff' }}
            >
              <FileText size={14} /> TXT
            </button>
            <button 
              onClick={handleDownloadDocx}
              disabled={loading || selectedFiles.length === 0}
              style={{ ...styles.exportBtn, background: '#166534', border: '1px solid #15803d', color: '#fff' }}
            >
              <Download size={14} /> DOCX
            </button>
            <button 
              onClick={() => handleDownloadBinary('pdf')}
              disabled={loading || selectedFiles.length === 0}
              style={{ ...styles.exportBtn, background: '#eab308', border: 'none', color: '#000' }}
            >
              <Download size={14} /> PDF
            </button>
            <button 
              onClick={() => handleDownloadBinary('epub')}
              disabled={loading || selectedFiles.length === 0}
              style={{ ...styles.exportBtn, background: '#3b82f6', border: 'none', color: '#fff' }}
            >
              <BookOpen size={14} /> EPUB
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}

const styles = {
  exportBtn: {
    flex: 1, padding: '10px 8px', borderRadius: '6px', cursor: 'pointer',
    fontSize: '12px', fontWeight: 'bold', display: 'flex', 
    justifyContent: 'center', alignItems: 'center', gap: '6px',
    transition: 'opacity 0.2s', minWidth: '60px'
  },
  optionLabel: {
    display: 'block', fontSize: '10px', color: '#555', 
    fontWeight: '700', textTransform: 'uppercase', 
    letterSpacing: '0.5px', marginBottom: '5px'
  },
  optionSelect: {
    width: '100%', padding: '7px', background: '#18181b', 
    border: '1px solid #333', color: '#ccc', borderRadius: '4px', 
    fontSize: '12px', outline: 'none'
  },
  optionInput: {
    width: '100%', padding: '7px', background: '#18181b', 
    border: '1px solid #333', color: '#ccc', borderRadius: '4px', 
    fontSize: '12px', outline: 'none', boxSizing: 'border-box'
  }
};
