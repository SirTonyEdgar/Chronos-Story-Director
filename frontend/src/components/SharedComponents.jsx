import React, { useState, useEffect, useRef } from 'react';
import { Zap, ChevronDown, Edit2 } from 'lucide-react';

export const TimelineDropdown = ({ value, onChange, timelines }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) setIsOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const displayValue = value || "Universal (All Timelines)";

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%' }}>
      <div
        onClick={() => setIsOpen(!isOpen)}
        style={{ 
          display: 'flex', alignItems: 'center', gap: '8px', 
          background: 'rgba(168, 85, 247, 0.1)', border: '1px solid #a855f7', 
          padding: '10px 12px', borderRadius: '6px', cursor: 'pointer', 
          color: '#e4e4e7', fontSize: '13px', transition: 'all 0.2s'
        }}
      >
        <Zap size={14} color="#a855f7" />
        <span style={{ flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {displayValue}
        </span>
        <ChevronDown size={14} color="#a855f7" style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }} />
      </div>

      {isOpen && (
        <div style={{ 
          position: 'absolute', top: '100%', left: 0, right: 0, 
          background: '#18181b', border: '1px solid #a855f7', borderRadius: '6px', 
          marginTop: '4px', zIndex: 100, overflow: 'hidden', 
          boxShadow: '0 10px 25px rgba(0,0,0,0.8)' 
        }}>
          <div
            onClick={() => { onChange(""); setIsOpen(false); }}
            style={{ 
              padding: '10px 12px', cursor: 'pointer', fontSize: '13px', 
              color: value === "" ? '#a855f7' : '#a1a1aa', 
              background: value === "" ? 'rgba(168, 85, 247, 0.1)' : 'transparent',
              fontWeight: value === "" ? 'bold' : 'normal'
            }}
          >
            Universal (All Timelines)
          </div>
          {timelines.map(tl => (
            <div
              key={tl.Name}
              onClick={() => { onChange(tl.Name); setIsOpen(false); }}
              style={{ 
                padding: '10px 12px', cursor: 'pointer', fontSize: '13px', 
                color: value === tl.Name ? '#a855f7' : '#a1a1aa', 
                background: value === tl.Name ? 'rgba(168, 85, 247, 0.1)' : 'transparent', 
                borderTop: '1px solid #27272a',
                fontWeight: value === tl.Name ? 'bold' : 'normal'
              }}
            >
              {tl.Name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export const EditableTextarea = ({ value, onChange, placeholder, style, highlightFocus }) => {
  const textareaRef = useRef(null);
  const adjustHeight = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${el.scrollHeight}px`;
    }
  };
  useEffect(() => { adjustHeight(); }, [value]);

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => { onChange(e); adjustHeight(); }}
        placeholder={placeholder}
        rows={1}
        style={{
          width: '100%', background: 'rgba(0,0,0,0.2)', border: '1px solid #2f2f35',
          borderRadius: '6px', color: '#eee', outline: 'none', fontFamily: 'inherit',
          resize: 'none', overflow: 'hidden', minHeight: '36px', lineHeight: '1.5',
          display: 'block', padding: '8px 10px', fontSize: '13px', transition: 'all 0.2s',
          ...style
        }}
        onFocus={(e) => {
          e.target.style.borderColor = highlightFocus || '#60a5fa';
          e.target.style.background = '#18181b';
        }}
        onBlur={(e) => {
          e.target.style.borderColor = '#2f2f35';
          e.target.style.background = 'rgba(0,0,0,0.2)';
        }}
      />
      {!value && <Edit2 size={10} color="#444" style={{ position: 'absolute', right: '10px', top: '12px', pointerEvents: 'none' }} />}
    </div>
  );
};