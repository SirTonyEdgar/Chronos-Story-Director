import React, { useCallback, useEffect, useMemo } from 'react';
import ReactFlow, { 
  useNodesState, 
  useEdgesState, 
  Background, 
  MarkerType,
  Handle,
  Position,
  useReactFlow,
  ReactFlowProvider,
  BaseEdge,
  getStraightPath,
  getBezierPath,
  EdgeLabelRenderer,
  useNodes
} from 'reactflow';
import 'reactflow/dist/style.css';
import axios from 'axios';
import { Save, ArrowDown, ArrowRight, CircleDot, ZoomIn, ZoomOut, Maximize, Layers } from 'lucide-react';
import dagre from 'dagre';
import { API_URL } from './config';
import { toast, confirm } from './components/Notifications';

// --- CONFIGURATION ---
const NODE_WIDTH = 220;
const NODE_HEIGHT = 240;
const SYSTEM_BUFFER = 2000;

/**
 * MATH UTILITIES
 */
const getBezierPoint = (t, sx, sy, tx, ty, sourcePos, targetPos) => {
  const dist = Math.sqrt(Math.pow(tx - sx, 2) + Math.pow(ty - sy, 2));
  const offset = Math.max(dist * 0.5, 25); 

  let cp1 = { x: sx, y: sy };
  let cp2 = { x: tx, y: ty };

  switch (sourcePos) {
    case Position.Left: cp1.x -= offset; break;
    case Position.Right: cp1.x += offset; break;
    case Position.Top: cp1.y -= offset; break;
    case Position.Bottom: cp1.y += offset; break;
  }

  switch (targetPos) {
    case Position.Left: cp2.x -= offset; break;
    case Position.Right: cp2.x += offset; break;
    case Position.Top: cp2.y -= offset; break;
    case Position.Bottom: cp2.y += offset; break;
  }

  const k = 1 - t;
  const x = (k*k*k * sx) + (3*k*k*t * cp1.x) + (3*k*t*t * cp2.x) + (t*t*t * tx);
  const y = (k*k*k * sy) + (3*k*k*t * cp1.y) + (3*k*t*t * cp2.y) + (t*t*t * ty);

  return { x, y };
};

const findSafeSpot = (pathX, pathY, allNodes) => {
  const HIT_W = (NODE_WIDTH / 2) + 150; 
  const HIT_H = (NODE_HEIGHT / 2) + 150;
  const isHit = (x, y) => {
    return allNodes.some(n => {
      const nx = n.position.x + NODE_WIDTH/2;
      const ny = n.position.y + NODE_HEIGHT/2;
      return Math.abs(x - nx) < HIT_W && Math.abs(y - ny) < HIT_H;
    });
  };
  return !isHit(pathX, pathY);
};

// --- COMPONENTS ---

const CustomNode = ({ data }) => {
  const getIconPath = (key) => {
    if (!key) return "/icons/question-mark.png";
    const map = {
      "Male": "male.png", "Female": "female.png", "Neutral": "neutral.png",
      "Villain": "villain.png", "Leader/Noble": "crown.png", "Official/Diplomat": "diplomat.png",
      "Wizard": "wizard.png", "Tech/Cyborg": "cyborg.png", "Soldier": "soldier.png",
      "Knight": "knight.png", "Child": "child.png", "Student": "student.png", "Cat": "cat.png",
      "Family": "family.png", "Organization/Corp": "briefcase.png",
      "Wealth/Economy": "wealth.png", "Resource/Item": "treasure-chest.png",
      "Asset": "treasure-chest.png"
    };
    const filename = map[key] || "neutral.png";
    return `/icons/${filename}`;
  };

  const getBorderColor = (d) => {
    if (d.role === "POV" || d.category === "Protagonist") return "#ffd700";
    if (d.role === "Antagonist" || d.category === "Enemy") return "#ef4444";
    const key = d.icon;
    if (key === "Enemy" || key === "Target" || key === "Villain") return "#ff4444"; 
    if (key === "Love" || key === "Family") return "#ff69b4"; 
    if (key === "Ally" || key === "Mentor") return "#4CAF50"; 
    return "#555"; 
  };

  return (
    <div style={{ 
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      minWidth: '220px', maxWidth: '450px', width: 'fit-content', height: 'auto',
      padding: '15px',
      border: `3px solid ${getBorderColor(data)}`, 
      borderRadius: '16px', 
      background: '#151515', 
      color: '#fff', 
      textAlign: 'center', 
      boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
      position: 'relative',
      zIndex: 100 
    }}>
      <Handle type="target" position={Position.Top} id="t-top" style={{visibility: 'hidden'}} />
      <Handle type="source" position={Position.Top} id="s-top" style={{visibility: 'hidden'}} />
      <Handle type="target" position={Position.Right} id="t-right" style={{visibility: 'hidden'}} />
      <Handle type="source" position={Position.Right} id="s-right" style={{visibility: 'hidden'}} />
      <Handle type="target" position={Position.Bottom} id="t-bottom" style={{visibility: 'hidden'}} />
      <Handle type="source" position={Position.Bottom} id="s-bottom" style={{visibility: 'hidden'}} />
      <Handle type="target" position={Position.Left} id="t-left" style={{visibility: 'hidden'}} />
      <Handle type="source" position={Position.Left} id="s-left" style={{visibility: 'hidden'}} />
      
      <div style={{ height: '140px', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '10px' }}>
        <img 
          src={getIconPath(data.icon)} 
          alt={data.icon}
          style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain', filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.5))' }} 
        />
      </div>

      <div style={{ fontSize: '24px', fontWeight: '700', color: '#eee', letterSpacing: '0.5px', lineHeight: '1.3', wordWrap: 'break-word', whiteSpace: 'pre-wrap' }}>
        {data.label}
      </div>
      {data.role && data.role !== "Support" && (
        <div style={{
          marginTop: '8px', fontSize: '11px', fontWeight: 'bold', textTransform: 'uppercase',
          padding: '2px 8px', borderRadius: '8px', 
          background: data.role === 'POV' ? 'rgba(255, 215, 0, 0.2)' : 'rgba(255, 255, 255, 0.1)',
          color: data.role === 'POV' ? '#ffd700' : '#aaa'
        }}>
          {data.role}
        </div>
      )}
    </div>
  );
};

const EdgeLabel = ({ label, x, y }) => (
  <EdgeLabelRenderer>
    <div style={{
        position: 'absolute',
        transform: `translate(-50%, -50%) translate(${x}px,${y}px)`,
        background: '#09090b', padding: '6px 12px', borderRadius: '4px',
        fontSize: '24px', fontWeight: 700, color: '#e4e4e7', border: '1px solid #444',
        pointerEvents: 'none', zIndex: 1002, whiteSpace: 'nowrap',
        boxShadow: '0 2px 4px rgba(0,0,0,0.8)'
      }}>
      {label}
    </div>
  </EdgeLabelRenderer>
);

const SmartBezierEdge = ({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, label, style, markerEnd }) => {
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition });
  const nodes = useNodes(); 

  // Default to midpoint for cleaner radial layouts
  let bestX = sourceX + (targetX - sourceX) * 0.5;
  let bestY = sourceY + (targetY - sourceY) * 0.5;
  let found = false;

  const hash = id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const jitter = (hash % 20) * 0.01; 

  // Try to find a spot that doesn't overlap a node
  for (let t = 0.5; t < 0.9; t += 0.05) {
    const tWithJitter = Math.min(0.9, Math.max(0.1, t + jitter));
    const pos = getBezierPoint(tWithJitter, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition);
    if (findSafeSpot(pos.x, pos.y, nodes)) {
      bestX = pos.x;
      bestY = pos.y;
      found = true;
      break;
    }
  }

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />
      {label && <EdgeLabel label={label} x={bestX} y={bestY} />}
    </>
  );
};

const SmartStraightEdge = ({ id, sourceX, sourceY, targetX, targetY, label, style, markerEnd }) => {
  const [edgePath] = getStraightPath({ sourceX, sourceY, targetX, targetY });
  const labelX = sourceX + (targetX - sourceX) * 0.5;
  const labelY = sourceY + (targetY - sourceY) * 0.5;

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />
      {label && <EdgeLabel label={label} x={labelX} y={labelY} />}
    </>
  );
};

// --- LAYOUT ENGINE: ISOLATED SYSTEMS ---

const getIsolatedRippleLayout = (nodes, edges) => {
  // 1. Identify Roles
  const povNodes = nodes.filter(n => n.data.role === 'POV' || n.data.category === 'Protagonist');
  const otherNodes = nodes.filter(n => !povNodes.includes(n));

  // Initialize Systems
  const systems = povNodes.map(pov => ({
    sun: pov,
    satellites: [],
    radius: 0
  }));
  const independents = [];

  // 2. Assign Satellites (Characters AND Assets)
  otherNodes.forEach(node => {
    let assigned = false;

    // A. Explicit Orbit Data
    if (node.data.orbit) {
      const system = systems.find(s => s.sun.id === node.data.orbit);
      if (system) { system.satellites.push(node); assigned = true; }
    }

    // B. Link Discovery (Fallback)
    if (!assigned) {
      const connectedSun = systems.find(s => 
        edges.some(e => 
          (e.source === s.sun.id && e.target === node.id) || 
          (e.target === s.sun.id && e.source === node.id)
        )
      );
      if (connectedSun) { connectedSun.satellites.push(node); assigned = true; }
    }

    if (!assigned) independents.push(node);
  });

  // 3. Layout Each System (Rings)
  let currentSystemX = 0;

  systems.forEach((sys, sysIndex) => {
    const rings = { personal: [], allies: [], assets: [], hostile: [] };
    
    // Sort logic
    const keywords = {
      personal: ["family", "love", "wife", "husband", "son", "daughter", "mother", "father", "spouse"],
      hostile: ["enemy", "rival", "target", "kill", "hates", "nemesis", "villain", "threat"],
      assets: ["wealth", "asset", "money", "base", "hq", "ship", "vehicle", "weapon", "corp", "owns"],
    };

    sys.satellites.forEach(node => {
      const edge = edges.find(e => 
        (e.source === sys.sun.id && e.target === node.id) || 
        (e.target === sys.sun.id && e.source === node.id)
      );
      const label = edge ? (edge.label || "").toLowerCase() : "";
      const cat = (node.data.category || "").toLowerCase();
      const combinedText = `${label} ${cat} ${node.data.icon}`;

      if (keywords.hostile.some(k => combinedText.includes(k))) rings.hostile.push(node);
      else if (keywords.personal.some(k => combinedText.includes(k))) rings.personal.push(node);
      else if (keywords.assets.some(k => combinedText.includes(k))) rings.assets.push(node);
      else if (cat === "asset") rings.assets.push(node);
      else rings.allies.push(node);
    });

    // Determine Radii (Generous Spacing for Labels)
    const ringConfig = [
      { id: 'personal', baseRadius: 800 },
      { id: 'allies',   baseRadius: 1600 },
      { id: 'assets',   baseRadius: 2000 },
      { id: 'hostile',  baseRadius: 3000 }
    ];

    let maxSystemRadius = 1000;

    ringConfig.forEach(config => {
      const group = rings[config.id];
      const count = group.length;
      if (count === 0) return;

      const minCirc = count * (NODE_WIDTH + 500); 
      const actualRadius = Math.max(config.baseRadius, minCirc / (2 * Math.PI));
      
      if (actualRadius > maxSystemRadius) maxSystemRadius = actualRadius;

      const angleStep = (2 * Math.PI) / count;
      
      group.forEach((node, i) => {
        const angle = i * angleStep;
        node.relativePos = {
          x: actualRadius * Math.cos(angle),
          y: actualRadius * Math.sin(angle)
        };
      });
    });

    // Calculate Global Position
    if (sysIndex > 0) {
      const prevSys = systems[sysIndex - 1];
      // Previous Center + Prev Radius + BUFFER + Current Radius
      currentSystemX += (prevSys.radius + maxSystemRadius + SYSTEM_BUFFER);
    }
    
    sys.sun.position = { x: currentSystemX, y: 0 };
    sys.radius = maxSystemRadius;

    sys.satellites.forEach(sat => {
      if (sat.relativePos) {
        sat.position = {
          x: currentSystemX + sat.relativePos.x,
          y: 0 + sat.relativePos.y
        };
      }
    });
  });

  // 4. Place Independents (Grid far below)
  if (independents.length > 0) {
    const startY = 3000;
    const cols = 6;
    independents.forEach((node, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      node.position = { x: col * 400, y: startY + row * 400 };
    });
  }

  return { nodes, edges };
};

const getDagreLayout = (nodes, edges, direction = 'TB') => {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction, nodesep: 150, ranksep: 200 });
  nodes.forEach((node) => { dagreGraph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT }); });
  edges.forEach((edge) => { dagreGraph.setEdge(edge.source, edge.target); });
  dagre.layout(dagreGraph);
  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    return { ...node, position: { x: nodeWithPosition.x - NODE_WIDTH / 2, y: nodeWithPosition.y - NODE_HEIGHT / 2 } };
  });
  return { nodes: layoutedNodes, edges };
};

const getSmartEdges = (nodes, edges, mode) => {
  const nodeMap = {};
  nodes.forEach(n => nodeMap[n.id] = n);
  return edges.map(edge => {
    const source = nodeMap[edge.source];
    const target = nodeMap[edge.target];
    if (!source || !target || !source.position || !target.position) return edge;
    let sourceHandle = 's-bottom';
    let targetHandle = 't-top';
    const dx = target.position.x - source.position.x;
    const dy = target.position.y - source.position.y;
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    if (angle > -45 && angle <= 45) { sourceHandle = 's-right'; targetHandle = 't-left'; }
    else if (angle > 45 && angle <= 135) { sourceHandle = 's-bottom'; targetHandle = 't-top'; }
    else if (angle > 135 || angle <= -135) { sourceHandle = 's-left'; targetHandle = 't-right'; }
    else { sourceHandle = 's-top'; targetHandle = 't-bottom'; }
    return { ...edge, sourceHandle, targetHandle };
  });
};

function GraphEditor({ profile }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { fitView, zoomIn, zoomOut } = useReactFlow();

  const nodeTypes = useMemo(() => ({ customNode: CustomNode }), []);
  const edgeTypes = useMemo(() => ({ 
    smartStraight: SmartStraightEdge,
    smartBezier: SmartBezierEdge
  }), []);

  useEffect(() => { 
    if (profile) fetchData(); 
  }, [profile]);

  async function fetchData() {
    try {
      const res = await axios.get(`${API_URL}/graph/${profile}`);
      const safeNodes = (res.data.nodes || []).map(n => ({
        ...n,
        position: { x: Number(n.position?.x) || 0, y: Number(n.position?.y) || 0 }
      }));

      setNodes(safeNodes);
      
      const arrowEdges = res.data.edges.map(edge => ({
        ...edge,
        type: 'smartBezier', 
        animated: false,
        markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#78909c' },
        style: { strokeWidth: 2, stroke: '#78909c' },
        label: edge.label
      }));
      
      setEdges(arrowEdges);
      
      const hasSavedLayout = safeNodes.some(n => Math.abs(n.position.x) > 1 || Math.abs(n.position.y) > 1);

      if (!hasSavedLayout && safeNodes.length > 0) {
        setTimeout(() => applyLayout('RIPPLE', safeNodes, arrowEdges), 100);
      } else {
        const restoredEdges = getSmartEdges(safeNodes, arrowEdges, 'RIPPLE');
        setEdges(restoredEdges);
        setTimeout(() => fitView({ padding: 0.1, duration: 800 }), 100);
      }

    } catch (err) { console.error("Fetch Error:", err); }
  }

  const applyLayout = useCallback((mode, inputNodes=null, inputEdges=null) => {
    const currentNodes = inputNodes || nodes;
    const currentEdges = inputEdges || edges;
    
    let layouted;
    if (mode === 'RIPPLE') layouted = getIsolatedRippleLayout(currentNodes, currentEdges);
    else layouted = getDagreLayout(currentNodes, currentEdges, mode);

    const cleanNodes = layouted.nodes.map(n => ({
        ...n,
        position: { x: n.position.x || 0, y: n.position.y || 0 }
    }));

    const lineType = (mode === 'RIPPLE') ? 'smartBezier' : 'smartStraight';
    const smartEdges = getSmartEdges(cleanNodes, layouted.edges, mode);
    
    setNodes([...cleanNodes]);
    setEdges(smartEdges.map(e => ({ 
      ...e, 
      type: lineType, 
      markerEnd: { type: MarkerType.ArrowClosed, width: 15, height: 15, color: '#78909c' } 
    })));
    window.requestAnimationFrame(() => fitView({ padding: 0.1, duration: 800 }));
  }, [nodes, edges, fitView, setNodes, setEdges]);

  const onNodeDrag = useCallback((event, node) => {
    setEdges((currentEdges) => getSmartEdges(nodes, currentEdges, 'RIPPLE'));
  }, [nodes, setEdges]);

  const savePositions = async () => {
    const updates = nodes.map(n => ({ id: n.id, position: n.position }));
    await axios.post(`${API_URL}/graph/${profile}`, { updates });
    toast("Layout Saved!", "success");
  };

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#09090b' }}>
      
      {/* Toolbar */}
      <div style={{
        position: 'absolute', top: 20, right: 20, zIndex: 10,
        display: 'flex', gap: '8px', background: '#18181b', padding: '8px', borderRadius: '8px', border: '1px solid #333'
      }}>
        <span style={{color: '#666', fontSize: '12px', alignSelf: 'center', marginRight: '5px'}}>Layout:</span>
        <button onClick={() => applyLayout('RIPPLE')} style={iconBtnStyle} title="Smart Ripple (Isolated)"><Layers size={18} /></button>
        <div style={{width: '1px', background: '#444', margin: '0 5px'}}></div>
        <button onClick={() => applyLayout('TB')} style={iconBtnStyle} title="Vertical Tree"><ArrowDown size={18} /></button>
        <button onClick={() => applyLayout('LR')} style={iconBtnStyle} title="Horizontal Tree"><ArrowRight size={18} /></button>
        
        <div style={{width: '1px', background: '#444', margin: '0 5px'}}></div>
        <button onClick={() => zoomIn()} style={iconBtnStyle} title="Zoom In"><ZoomIn size={18} /></button>
        <button onClick={() => zoomOut()} style={iconBtnStyle} title="Zoom Out"><ZoomOut size={18} /></button>
        <button onClick={() => fitView()} style={iconBtnStyle} title="Fit to Screen"><Maximize size={18} /></button>

        <div style={{width: '1px', background: '#444', margin: '0 5px'}}></div>
        <button onClick={savePositions} style={saveBtnStyle}><Save size={16} /> Save</button>
      </div>

      <ReactFlow 
        nodes={nodes} 
        edges={edges} 
        onNodesChange={onNodesChange} 
        onEdgesChange={onEdgesChange} 
        onNodeDrag={onNodeDrag} 
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        minZoom={0.1}
        maxZoom={4}
        fitView
      >
        <Background color="#333" gap={25} size={1} />
      </ReactFlow>
    </div>
  );
}

export default function NetworkMap({ profile }) {
  return (
    <ReactFlowProvider>
      <GraphEditor profile={profile} />
    </ReactFlowProvider>
  );
}

const iconBtnStyle = { background: '#333', color: '#ccc', border: 'none', padding: '8px', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center' };
const saveBtnStyle = { background: '#2563eb', color: 'white', border: 'none', padding: '8px 16px', borderRadius: '4px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 'bold' };