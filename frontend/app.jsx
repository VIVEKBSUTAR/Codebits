import { useState, useEffect, useRef, useMemo, useCallback } from "react"
import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts"
import { 
  Activity, HardHat, Car, Flame, Ambulance, 
  ChevronRight, AlertTriangle, Droplets, CloudRain, ShieldAlert
} from "lucide-react"

(function () {
  const href = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap"
  if (!document.querySelector(`[href="${href}"]`)) {
    const l = document.createElement("link")
    l.rel = "stylesheet"
    l.href = href
    document.head.appendChild(l)
  }
})()

const T = {
  bg: "#f4f3ee",
  panel: "#ffffff",
  border: "#e2e0d8",
  borderL: "#d1cec1",
  text: "#1d1f24",
  muted: "#666b75",
  label: "#4a4f59",
  green: "#22c55e",
  orange: "#f59e0b",
  red: "#ef4444",
  blue: "#3b82f6",
  navBg: "#ebe9e1",
}

const FF = { fontFamily: "'IBM Plex Sans', sans-serif" }
const FM = { fontFamily: "'IBM Plex Mono', monospace" }

const ZONE_DEFS = [
  { id: 0, name: "Bibwewadi", lat: 18.4582, lng: 73.8620, area: "8.2 km²",  pop: "142,000", radius: 2500 },
  { id: 1, name: "Katraj",    lat: 18.4437, lng: 73.8638, area: "11.5 km²", pop: "198,000", radius: 2800 },
  { id: 2, name: "Hadapsar",  lat: 18.4980, lng: 73.9258, area: "14.1 km²", pop: "310,000", radius: 3200 },
  { id: 3, name: "Kothrud",   lat: 18.5074, lng: 73.8077, area: "9.8 km²",  pop: "225,000", radius: 2600 },
  { id: 4, name: "Warje",     lat: 18.4832, lng: 73.8118, area: "7.3 km²",  pop: "118,000", radius: 2200 },
  { id: 5, name: "Sinhagad",  lat: 18.4326, lng: 73.8095, area: "6.1 km²",  pop: "89,000",  radius: 2000 },
  { id: 6, name: "Pimpri",    lat: 18.6279, lng: 73.8009, area: "12.4 km²", pop: "250,000", radius: 3000 },
  { id: 7, name: "Chinchwad", lat: 18.6261, lng: 73.7828, area: "10.2 km²", pop: "210,000", radius: 2800 },
  { id: 8, name: "Hinjewadi", lat: 18.5913, lng: 73.7389, area: "18.5 km²", pop: "120,000", radius: 3500 },
  { id: 9, name: "Wakad",     lat: 18.5987, lng: 73.7688, area: "8.5 km²",  pop: "150,000", radius: 2400 },
  { id: 10, name: "Baner",    lat: 18.5590, lng: 73.7868, area: "7.8 km²",  pop: "110,000", radius: 2300 },
  { id: 11, name: "Shivajinagar", lat: 18.5314, lng: 73.8446, area: "5.5 km²", pop: "135,000", radius: 2100 }
]

const EVENT_TYPES = ["rainfall", "construction", "accident", "drainage_failure"]
const SEVERITIES  = ["low", "medium", "high"]

const API_BASE = "/api"
const EVENT_TYPE_MAP = { drainage_failure: "flood" }

const TILE_SIZE = 256

function project(lat, lng, zoom) {
  const scale  = TILE_SIZE * Math.pow(2, zoom)
  const x      = (lng + 180) / 360 * scale
  const sinLat = Math.sin(lat * Math.PI / 180)
  const y      = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale
  return { x, y }
}

function pixelToLatLng(px, py, zoom) {
  const scale = TILE_SIZE * Math.pow(2, zoom)
  const lng   = px / scale * 360 - 180
  const n     = Math.PI - (2 * Math.PI * py) / scale
  const lat   = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)))
  return { lat, lng }
}

function latLngToScreen(lat, lng, center, zoom, w, h) {
  const cp = project(center.lat, center.lng, zoom)
  const pp = project(lat, lng, zoom)
  return { x: w / 2 + (pp.x - cp.x), y: h / 2 + (pp.y - cp.y) }
}

function buildTiles(center, zoom, w, h) {
  if (!w || !h) return []
  const cp      = project(center.lat, center.lng, zoom)
  const maxTile = Math.pow(2, zoom)
  const startX  = Math.floor((cp.x - w / 2) / TILE_SIZE)
  const endX    = Math.ceil( (cp.x + w / 2) / TILE_SIZE)
  const startY  = Math.floor((cp.y - h / 2) / TILE_SIZE)
  const endY    = Math.ceil( (cp.y + h / 2) / TILE_SIZE)
  const tiles   = []
  const subdomains = ["a","b","c","d"]
  for (let tx = startX; tx <= endX; tx++) {
    for (let ty = startY; ty <= endY; ty++) {
      const tileX = ((tx % maxTile) + maxTile) % maxTile
      const tileY = ((ty % maxTile) + maxTile) % maxTile
      if (tileY < 0 || tileY >= maxTile) continue
      const sub = subdomains[(Math.abs(tileX + tileY)) % 4]
      tiles.push({
        key:     `${zoom}_${tx}_${ty}`,
        url:     `https://${sub}.basemaps.cartocdn.com/rastertiles/voyager/${zoom}/${tileX}/${tileY}.png`,
        screenX: w / 2 + tx * TILE_SIZE - cp.x,
        screenY: h / 2 + ty * TILE_SIZE - cp.y,
      })
    }
  }
  return tiles
}

function computeRisks(events = [], interventions = []) {
  const has = (t, s) => events.some(e => e.type === t && (!s || e.severity === s))
  const rainfall = has("rainfall","high") ? 0.85 : has("rainfall") ? 0.45 : 0.08
  const constr   = has("construction")    ? 0.72 : 0.18
  const accident = has("accident")        ? 0.60 : 0.10
  const drainage = has("drainage_failure")? 0.80 : 0.28
  
  let flood     = Math.min(0.95, rainfall * 0.58 + drainage * 0.42 + Math.random() * 0.03)
  let traffic   = Math.min(0.95, flood * 0.44 + constr * 0.34 + accident * 0.22 + Math.random() * 0.03)
  let emergency = Math.min(0.95, traffic * 0.54 + flood * 0.36 + Math.random() * 0.03)
  
  return { flood, traffic, emergency, overall: flood * 0.35 + traffic * 0.40 + emergency * 0.25 }
}

function getRisks(zone, interventionRisks = null) {
  if (interventionRisks) {
    const overall = interventionRisks.flooding * 0.35 + interventionRisks.traffic * 0.40 + interventionRisks.emergency_delay * 0.25
    return {
      flood: interventionRisks.flooding,
      traffic: interventionRisks.traffic,
      emergency: interventionRisks.emergency_delay,
      overall
    }
  }
  return zone.apiRisks || computeRisks(zone.events, zone.interventions)
}

function generatePolygonPoints(lat, lng, radiusMeter, seed) {
  const points = []
  const numPoints = 5 
  const rLat = (radiusMeter / 1000) / 110.574
  const rLng = (radiusMeter / 1000) / (111.320 * Math.cos(lat * Math.PI / 180))
  for(let i=0; i<numPoints; i++) {
    const angle = (i * 2 * Math.PI) / numPoints + (seed * 0.5)
    const rand = Math.abs(Math.sin(seed * 12.9898 + i * 78.233)) * 0.5 + 0.7
    points.push({ lat: lat + rLat * Math.sin(angle) * rand, lng: lng + rLng * Math.cos(angle) * rand })
  }
  return points
}

const pct       = v => Math.round(v * 100)
const hms       = d => d.toTimeString().slice(0, 8)
const riskColor = v => v > 0.68 ? T.red : v > 0.42 ? T.orange : T.green
const riskLabel = v => v > 0.68 ? "CRITICAL" : v > 0.42 ? "ELEVATED" : "OPERATIONAL"

function seedZones() {
  return ZONE_DEFS.map((z, idx) => ({
    ...z,
    polygon: generatePolygonPoints(z.lat, z.lng, z.radius, idx + 10),
    events: Math.random() > 0.6
      ? [{ type: EVENT_TYPES[Math.floor(Math.random() * 4)], severity: SEVERITIES[Math.floor(Math.random() * 3)], id: Math.random() }]
      : [],
    interventions: [],
    apiRisks: null,
    history: Array.from({ length: 24 }, (_, i) => ({
      t: i, label: `${String(i).padStart(2, "0")}:00`,
      overall:   +(0.12 + Math.random() * 0.50).toFixed(3),
      flood:     +(0.08 + Math.random() * 0.45).toFixed(3),
      traffic:   +(0.15 + Math.random() * 0.55).toFixed(3),
      yesterday: +(0.10 + Math.random() * 0.40).toFixed(3),
    })),
  }))
}

const CT = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{ background: T.panel, border: `1px solid ${T.border}`, padding: "8px 12px", borderRadius: 4, fontFamily: "IBM Plex Mono", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}>
      <div style={{ fontSize: 9, color: T.muted, marginBottom: 4 }}>{label}</div>
      {payload.map(p => <div key={p.dataKey} style={{ fontSize: 11, color: p.stroke }}>{p.name}: {pct(p.value)}%</div>)}
    </div>
  )
}

function TileMap({ zones, selected, onSelect, mapLayer }) {
  const containerRef = useRef(null)
  const [size,    setSize]   = useState({ w: 0, h: 0 })
  const [center,  setCenter] = useState({ lat: 18.53, lng: 73.82 })
  const [zoom,    setZoom]   = useState(12)
  const [tooltip, setTooltip]= useState(null)
  const dragRef = useRef(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      setSize({ w: el.offsetWidth, h: el.offsetHeight })
    })
    ro.observe(el)
    setSize({ w: el.offsetWidth, h: el.offsetHeight })
    return () => ro.disconnect()
  }, [])

  const tiles = useMemo(() => buildTiles(center, zoom, size.w, size.h), [center, zoom, size])

  const onMouseDown = e => {
    dragRef.current = { x: e.clientX, y: e.clientY, center: { ...center } }
  }
  const onMouseMove = e => {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.x
    const dy = e.clientY - dragRef.current.y
    const cp = project(dragRef.current.center.lat, dragRef.current.center.lng, zoom)
    const { lat, lng } = pixelToLatLng(cp.x - dx, cp.y - dy, zoom)
    setCenter({ lat, lng })
  }
  const onMouseUp = () => { dragRef.current = null }

  const zoomIn  = e => { e.stopPropagation(); setZoom(z => Math.min(18, z + 1)) }
  const zoomOut = e => { e.stopPropagation(); setZoom(z => Math.max(10, z - 1)) }

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const handler = e => {
      e.preventDefault()
      setZoom(z => e.deltaY < 0 ? Math.min(18, z + 1) : Math.max(10, z - 1))
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [])

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", position: "relative", overflow: "hidden", background: "#e5e3df", cursor: dragRef.current ? "grabbing" : "grab", userSelect: "none" }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      {tiles.map(t => (
        <img
          key={t.key}
          src={t.url}
          alt=""
          draggable={false}
          style={{
            position: "absolute",
            left: Math.round(t.screenX),
            top:  Math.round(t.screenY),
            width:  TILE_SIZE,
            height: TILE_SIZE,
            display: "block",
            pointerEvents: "none",
            filter: "brightness(0.97) contrast(1.05)"
          }}
        />
      ))}

      {size.w > 0 && (
        <svg style={{ position: "absolute", inset: 0 }} width={size.w} height={size.h}>
          {zones.map((z, i) => {
            const r = getRisks(z)
            const targetVal = r[mapLayer]
            const color = riskColor(targetVal)
            const isSel = i === selected
            
            const pointsString = z.polygon.map(p => {
              const pt = latLngToScreen(p.lat, p.lng, center, zoom, size.w, size.h)
              return `${pt.x},${pt.y}`
            }).join(" ")

            const pos = latLngToScreen(z.lat, z.lng, center, zoom, size.w, size.h)

            return (
              <g key={z.id}>
                <polygon
                  points={pointsString}
                  fill={color}
                  fillOpacity={isSel ? 0.6 : 0.35}
                  stroke={isSel ? "#111111" : "#ffffff"}
                  strokeWidth={isSel ? 3 : 1.5}
                  strokeDasharray={isSel ? "none" : "8 6"}
                  style={{ cursor: "pointer", transition: "all 0.3s" }}
                  onClick={e => { e.stopPropagation(); onSelect(i) }}
                  onMouseEnter={() => setTooltip({ i, x: pos.x, y: pos.y })}
                  onMouseLeave={() => setTooltip(null)}
                />
                
                <text
                  x={pos.x}
                  y={pos.y}
                  fill="#111111"
                  fontSize={isSel ? 15 : 12}
                  fontWeight="700"
                  textAnchor="middle"
                  alignmentBaseline="middle"
                  style={{ pointerEvents: "none", fontFamily: "'IBM Plex Sans', sans-serif", textShadow: "0px 1px 4px rgba(255,255,255,1)" }}
                >
                  {z.name}
                </text>
              </g>
            )
          })}
        </svg>
      )}

      {/* Hover tooltip */}
      {tooltip !== null && size.w > 0 && (() => {
        const z   = zones[tooltip.i]
        const r   = getRisks(z)
        const col = riskColor(r[mapLayer])
        const tx  = Math.min(tooltip.x + 10, size.w - 170)
        const ty  = Math.max(tooltip.y - 110, 6)
        return (
          <div style={{
            position:   "absolute",
            left:       tx,
            top:        ty,
            background: T.panel,
            border:     `1px solid ${T.borderL}`,
            borderRadius: 6,
            padding:    "9px 13px",
            zIndex:     500,
            pointerEvents: "none",
            minWidth:   160,
            boxShadow:  "0 8px 24px rgba(0,0,0,0.15)",
          }}>
            <div style={{ fontWeight: 700, fontSize: 13, color: col, marginBottom: 5, fontFamily: "IBM Plex Sans" }}>{z.name}</div>
            <div style={{ fontSize: 10, color: T.text, fontFamily: "IBM Plex Mono", lineHeight: 1.7 }}>
              <div style={{ fontWeight: mapLayer === 'overall' ? 700 : 400 }}>Overall  <span style={{ color: riskColor(r.overall), fontWeight: 700 }}>{pct(r.overall)}%</span></div>
              <div style={{ fontWeight: mapLayer === 'flood' ? 700 : 400 }}>Flood    <span style={{ color: riskColor(r.flood) }}>{pct(r.flood)}%</span></div>
              <div style={{ fontWeight: mapLayer === 'traffic' ? 700 : 400 }}>Traffic  <span style={{ color: riskColor(r.traffic) }}>{pct(r.traffic)}%</span></div>
              <div style={{ fontWeight: mapLayer === 'emergency' ? 700 : 400 }}>Emergency <span style={{ color: riskColor(r.emergency) }}>{pct(r.emergency)}%</span></div>
            </div>
          </div>
        )
      })()}

      <div style={{ position: "absolute", bottom: 20, right: 14, zIndex: 300, display: "flex", flexDirection: "column", gap: 4 }}>
        {[["+", zoomIn], ["−", zoomOut]].map(([label, fn]) => (
          <button key={label} onClick={fn} style={{
            width: 36, height: 36, borderRadius: 8,
            border: `1px solid ${T.borderL}`, background: T.panel,
            color: T.text, cursor: "pointer", fontSize: 20,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontFamily: "IBM Plex Mono", boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
          }}>{label}</button>
        ))}
      </div>
    </div>
  )
}

// ── Dynamic Live Causal Graph ──────────────────────────────────────────────────
function DynamicCausalGraph({ zones, onNodeClick }) {
  const agg = useMemo(() => {
    let hasRain = false, hasConst = false, hasDrainage = false
    let maxFlood = 0, maxTraffic = 0, maxEmergency = 0

    zones.forEach(z => {
      if (z.events.some(e => e.type === "rainfall")) hasRain = true
      if (z.events.some(e => e.type === "construction")) hasConst = true
      if (z.events.some(e => e.type === "drainage_failure")) hasDrainage = true
      
      const r = getRisks(z)
      if (r.flood > maxFlood) maxFlood = r.flood
      if (r.traffic > maxTraffic) maxTraffic = r.traffic
      if (r.emergency > maxEmergency) maxEmergency = r.emergency
    })

    return { hasRain, hasConst, hasDrainage, maxFlood, maxTraffic, maxEmergency }
  }, [zones])

  const chain = []
  if (agg.hasRain || agg.hasDrainage) {
    chain.push({ id: "n1", label: "HEAVY RAINFALL", icon: CloudRain, metric: "overall", isRoot: true, active: true })
    chain.push({ id: "n2", label: "SURFACE FLOODING", icon: Droplets, metric: "flood", active: agg.maxFlood > 0.4 })
    chain.push({ id: "n3", label: "TRAFFIC CONGESTION", icon: Car, metric: "traffic", active: agg.maxTraffic > 0.4 })
    chain.push({ id: "n4", label: "EMERGENCY DELAY", icon: Ambulance, metric: "emergency", active: agg.maxEmergency > 0.4 })
  } else {
    chain.push({ id: "n1", label: "CONSTRUCTION WORK", icon: HardHat, metric: "overall", isRoot: true, active: agg.hasConst })
    chain.push({ id: "n2", label: "TRAFFIC GRIDLOCK", icon: Car, metric: "traffic", active: agg.maxTraffic > 0.4 })
    chain.push({ id: "n3", label: "ACCIDENT RISK", icon: Flame, metric: "overall", active: agg.maxTraffic > 0.6 })
    chain.push({ id: "n4", label: "EMERGENCY DELAY", icon: Ambulance, metric: "emergency", active: agg.maxEmergency > 0.4 })
  }

  const getProb = (metric) => {
    if (metric === 'flood') return pct(agg.maxFlood)
    if (metric === 'traffic') return pct(agg.maxTraffic)
    if (metric === 'emergency') return pct(agg.maxEmergency)
    return Math.floor(Math.random() * 20 + 60)
  }

  return (
    <div style={{
      width: "100%", height: "260px", background: T.panel,
      border: `1px solid ${T.border}`, borderRadius: 8, position: "relative",
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "0 60px", backgroundImage: `radial-gradient(${T.borderL} 1.5px, transparent 1.5px)`,
      backgroundSize: "24px 24px"
    }}>
      {chain.map((node, i) => {
        const Icon = node.icon
        const val = node.metric === 'overall' ? 0.8 : (node.metric === 'flood' ? agg.maxFlood : node.metric === 'traffic' ? agg.maxTraffic : agg.maxEmergency)
        const nodeColor = !node.active ? T.borderL : riskColor(val)
        const nextProb = i < chain.length - 1 ? getProb(chain[i+1].metric) : 0
        const edgeColor = nextProb > 68 ? T.red : nextProb > 42 ? T.orange : T.green

        return (
          <div key={node.id} style={{ display: "flex", alignItems: "center", flex: i === chain.length - 1 ? "none" : 1 }}>
            
            <div 
              onClick={() => onNodeClick(node.metric)}
              style={{
                width: 100, height: 100, borderRadius: "50%", background: T.panel,
                border: `3px solid ${nodeColor}`,
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                boxShadow: node.active && node.isRoot ? `0 0 0 8px ${nodeColor}15` : "0 4px 16px rgba(0,0,0,0.06)",
                zIndex: 10, flexShrink: 0, cursor: "pointer", transition: "all 0.2s transform"
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = "scale(1.05)"}
              onMouseLeave={(e) => e.currentTarget.style.transform = "scale(1)"}
            >
              <Icon size={28} color={T.text} style={{ marginBottom: 6 }} />
              <div style={{ fontSize: 10, fontWeight: 700, color: T.text, textAlign: "center", lineHeight: 1.2, width: "80%" }}>
                {node.label}
              </div>
            </div>

            {i < chain.length - 1 && (
              <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <div style={{ 
                  position: "absolute", left: 0, right: 0, top: "50%", 
                  borderTop: `2px ${nextProb > 68 ? "dashed" : "solid"} ${edgeColor}`, 
                  zIndex: 1, opacity: node.active ? 1 : 0.3
                }} />
                
                {node.active && (
                  <div style={{ 
                    position: "relative", zIndex: 2, background: edgeColor, color: "#fff", 
                    padding: "4px 12px", borderRadius: 12, fontSize: 13, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace" 
                  }}>
                    0.{nextProb}
                  </div>
                )}
                
                <div style={{ position: "absolute", right: -5, top: "50%", transform: "translateY(-50%)", color: edgeColor, zIndex: 1, opacity: node.active ? 1 : 0.3 }}>
                  <ChevronRight size={20} />
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function SidebarZonesList({ zones, sel, setSel, mapLayer }) {
  return (
    <div style={{ padding: "24px", flex: 1, overflowY: "auto", background: T.panel, borderRight: `1px solid ${T.border}` }}>
      <div style={{ fontSize: 11, color: T.muted, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 16, fontWeight: 600 }}>Municipal Zones</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {zones.map((z, i) => {
          const r = getRisks(z)
          const activeRisk = r[mapLayer]
          return (
            <button key={z.id} onClick={() => setSel(i)} style={{
              display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderRadius: 8,
              background: i === sel ? T.bg : "transparent",
              border: `1px solid ${i === sel ? T.borderL : "transparent"}`,
              cursor: "pointer", textAlign: "left", ...FF, transition: "all 0.2s"
            }}>
              <div style={{ width: 12, height: 12, borderRadius: "50%", background: riskColor(activeRisk), flexShrink: 0 }} />
              <span style={{ fontSize: 14, fontWeight: i === sel ? 600 : 500, color: T.text, flex: 1 }}>{z.name}</span>
              <span style={{ fontSize: 13, ...FM, color: riskColor(activeRisk), fontWeight: 700 }}>{pct(activeRisk)}%</span>
              {z.events.length > 0 && mapLayer === 'overall' && <span style={{ fontSize: 10, color: T.orange, background: T.orange + "15", border: `1px solid ${T.orange}40`, padding: "2px 6px", borderRadius: 4 }}>{z.events.length} ev</span>}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Resource Optimization Panel ─────────────────────────────────────────────────
function ResourceOptimizationPanel({ zoneName }) {
  const [deployment, setDeployment] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchOptimalDeployment = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/optimal-deployment?resources=pumps:2,ambulances:1,traffic_units:2`)
      if (res.ok) {
        const data = await res.json()
        setDeployment(data)
      }
    } catch (_) {}
    setLoading(false)
  }, [])

  useEffect(() => {
    if (zoneName) fetchOptimalDeployment()
  }, [zoneName, fetchOptimalDeployment])

  if (loading) return <div style={{...T.P, padding: 24, textAlign: "center"}}>Optimizing deployment...</div>

  if (!deployment) return <div style={{...T.P, padding: 24}}>No deployment data available.</div>

  return (
    <div style={{...T.P, margin: 24}}>
      <h3 style={{fontSize: 16, fontWeight: 700, marginBottom: 20, color: T.text}}>Optimal Resource Deployment</h3>
      <div style={{fontSize: 13, color: T.muted, marginBottom: 16}}>
        Expected citywide risk reduction: <span style={{color: T.green, fontWeight: 700}}>{deployment.expected_citywide_risk_reduction}%</span>
      </div>
      <div style={{display: "flex", flexDirection: "column", gap: 12}}>
        {deployment.plan?.map((item, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 12, padding: "12px 16px",
            background: T.bg, borderRadius: 6, border: `1px solid ${T.border}`
          }}>
            <div style={{
              padding: "6px 10px", background: T.blue + "20", color: T.blue,
              borderRadius: 4, fontSize: 11, fontWeight: 700, textTransform: "uppercase"
            }}>{item.resource}</div>
            <span style={{fontSize: 14, color: T.text, flex: 1}}>{item.zone}</span>
            <span style={{fontSize: 12, color: T.green, fontWeight: 600}}>+{item.benefit_expected}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Intervention Controls ──────────────────────────────────────────────────────
function InterventionControls({ zoneName, onRiskUpdate }) {
  const [activeInterventions, setActiveInterventions] = useState(new Set())
  const [simulationResults, setSimulationResults] = useState(null)

  const INTERVENTIONS = [
    { id: "deploy_pump", label: "Deploy Pump", icon: "🔧", color: T.blue },
    { id: "close_road", label: "Close Road", icon: "🚫", color: T.orange },
    { id: "dispatch_ambulance", label: "Dispatch Ambulance", icon: "🚑", color: T.red }
  ]

  const toggleIntervention = useCallback(async (interventionId) => {
    const newActive = new Set(activeInterventions)
    const isActivating = !newActive.has(interventionId)

    if (isActivating) {
      newActive.add(interventionId)
      try {
        const res = await fetch(`${API_BASE}/simulate-intervention`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ zone: zoneName, intervention: interventionId })
        })
        if (res.ok) {
          const data = await res.json()
          setSimulationResults(data)
          onRiskUpdate?.(data.after_intervention)
        }
      } catch (_) {}
    } else {
      newActive.delete(interventionId)
      setSimulationResults(null)
      onRiskUpdate?.(null)
    }

    setActiveInterventions(newActive)
  }, [activeInterventions, zoneName, onRiskUpdate])

  return (
    <div style={{...T.P, margin: 24}}>
      <h3 style={{fontSize: 16, fontWeight: 700, marginBottom: 20, color: T.text}}>Impact Simulation</h3>

      <div style={{display: "flex", flexDirection: "column", gap: 12, marginBottom: 20}}>
        {INTERVENTIONS.map(intervention => (
          <button
            key={intervention.id}
            onClick={() => toggleIntervention(intervention.id)}
            style={{
              display: "flex", alignItems: "center", gap: 12, padding: "12px 16px",
              background: activeInterventions.has(intervention.id) ? intervention.color + "20" : T.bg,
              border: `2px solid ${activeInterventions.has(intervention.id) ? intervention.color : T.border}`,
              borderRadius: 8, cursor: "pointer", transition: "all 0.2s",
              color: activeInterventions.has(intervention.id) ? intervention.color : T.text
            }}
          >
            <span style={{fontSize: 18}}>{intervention.icon}</span>
            <span style={{fontSize: 14, fontWeight: 600, flex: 1}}>{intervention.label}</span>
            <span style={{fontSize: 12, fontWeight: 700}}>
              {activeInterventions.has(intervention.id) ? "ON" : "OFF"}
            </span>
          </button>
        ))}
      </div>

      {simulationResults && (
        <div style={{padding: 16, background: T.green + "10", borderRadius: 6, border: `1px solid ${T.green}40`}}>
          <div style={{fontSize: 13, fontWeight: 700, color: T.green, marginBottom: 8}}>Risk Reduction Impact:</div>
          {Object.entries(simulationResults.benefit).map(([key, value]) => (
            <div key={key} style={{fontSize: 12, color: T.text, display: "flex", justifyContent: "space-between"}}>
              <span>{key.replace('_', ' ')}</span>
              <span style={{color: T.green, fontWeight: 600}}>-{value}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Event Timeline ─────────────────────────────────────────────────────────────
function EventTimeline({ zoneName }) {
  const [timeline, setTimeline] = useState(null)
  const [loading, setLoading] = useState(false)

  const fetchTimeline = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/zone-timeline?zone=${encodeURIComponent(zoneName)}`)
      if (res.ok) {
        const data = await res.json()
        setTimeline(data)
      }
    } catch (_) {}
    setLoading(false)
  }, [zoneName])

  useEffect(() => {
    if (zoneName) fetchTimeline()
  }, [zoneName, fetchTimeline])

  if (loading) return <div style={{...T.P, padding: 24, textAlign: "center"}}>Loading timeline...</div>

  if (!timeline || !timeline.predicted_events?.length) {
    return <div style={{...T.P, padding: 24, textAlign: "center", color: T.muted}}>No timeline events predicted</div>
  }

  return (
    <div style={{...T.P, margin: 24}}>
      <div style={{display: "flex", alignItems: "center", gap: 12, marginBottom: 20}}>
        <h3 style={{fontSize: 16, fontWeight: 700, color: T.text}}>Event Timeline</h3>
        <div style={{
          padding: "4px 8px", borderRadius: 4, fontSize: 11, fontWeight: 700,
          background: timeline.escalation_risk_score === "HIGH" ? T.red + "20" :
                     timeline.escalation_risk_score === "MEDIUM" ? T.orange + "20" : T.green + "20",
          color: timeline.escalation_risk_score === "HIGH" ? T.red :
                timeline.escalation_risk_score === "MEDIUM" ? T.orange : T.green
        }}>
          {timeline.escalation_risk_score} RISK
        </div>
      </div>

      <div style={{display: "flex", flexDirection: "column", gap: 8}}>
        {timeline.predicted_events.map((event, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 16, padding: "12px 16px",
            background: T.bg, borderRadius: 6, border: `1px solid ${T.border}`
          }}>
            <div style={{
              padding: "6px 10px", background: T.blue + "20", color: T.blue,
              borderRadius: 4, fontSize: 12, fontWeight: 700, fontFamily: "'IBM Plex Mono', monospace"
            }}>{event.predicted_time}</div>
            <span style={{fontSize: 14, color: T.text, flex: 1}}>{event.event_name}</span>
            <span style={{fontSize: 13, color: riskColor(event.probability), fontWeight: 600}}>
              {Math.round(event.probability * 100)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main Dashboard App ───────────────────────────────────────────────────────
export default function App() {
  const [zones,       setZones]       = useState(seedZones)
  const [sel,         setSel]         = useState(0)
  const [tab,         setTab]         = useState("Overview")
  const [mapLayer,    setMapLayer]    = useState("overall")
  const [now,         setNow]         = useState(new Date())
  const [showInject,  setShowInject]  = useState(false)
  const [newEv,       setNewEv]       = useState({ type: "rainfall", severity: "high" })
  const [interventionRisks, setInterventionRisks] = useState(null)

  const zone  = zones[sel]
  const risks = getRisks(zone, interventionRisks)

  const handleInterventionUpdate = useCallback((newRisks) => {
    setInterventionRisks(newRisks)
  }, [])

  const handleGraphNodeClick = useCallback((metric) => {
    setMapLayer(metric)
    setTab("Overview")

    // Find zone with highest risk for this metric
    let maxRisk = 0
    let targetZoneIndex = sel
    zones.forEach((z, i) => {
      const zoneRisk = getRisks(z, interventionRisks)[metric]
      if (zoneRisk > maxRisk) {
        maxRisk = zoneRisk
        targetZoneIndex = i
      }
    })

    // Switch to the zone with highest risk for this metric
    if (targetZoneIndex !== sel) {
      setSel(targetZoneIndex)
    }
  }, [zones, sel, interventionRisks])

  const fetchZoneRisk = useCallback(async (zoneName, zoneIdx) => {
    try {
      const res = await fetch(`${API_BASE}/zone-risk?zone=${encodeURIComponent(zoneName)}`)
      if (!res.ok) return
      const data = await res.json()
      const flood     = data.predictions.find(p => p.event === "Flooding")?.probability ?? null
      const traffic   = data.predictions.find(p => p.event === "TrafficCongestion")?.probability ?? null
      const emergency = data.predictions.find(p => p.event === "EmergencyDelay")?.probability ?? null
      if (flood === null || traffic === null || emergency === null) return
      const overall = flood * 0.35 + traffic * 0.40 + emergency * 0.25
      setZones(prev => prev.map((z, i) => i !== zoneIdx ? z : { ...z, apiRisks: { flood, traffic, emergency, overall } }))
    } catch (_) {}
  }, [])

  useEffect(() => {
    fetchZoneRisk(zones[sel].name, sel)
  }, [sel, fetchZoneRisk])

  const stableCount   = zones.filter(z => getRisks(z).overall < 0.43).length
  const opPct         = Math.round((stableCount / zones.length) * 100)
  const criticalCount = zones.filter(z => getRisks(z).overall > 0.68).length

  useEffect(() => {
    const id = setInterval(() => {
      setNow(new Date())
      setZones(prev => prev.map(z => {
        const r  = getRisks(z)
        const pt = {
          t: z.history.length, label: new Date().toTimeString().slice(0, 5),
          overall:   +(r.overall   + (Math.random() - 0.5) * 0.04).toFixed(3),
          flood:     +(r.flood     + (Math.random() - 0.5) * 0.04).toFixed(3),
          traffic:   +(r.traffic   + (Math.random() - 0.5) * 0.04).toFixed(3),
          yesterday: +(0.15 + Math.random() * 0.35).toFixed(3),
        }
        return { ...z, history: [...z.history.slice(-24), pt] }
      }))
    }, 3000)
    return () => clearInterval(id)
  }, [])

  const injectEvent = useCallback(async () => {
    const zoneName = zones[sel].name
    setZones(prev => prev.map((z, i) => i !== sel ? z : { ...z, events: [...z.events, { ...newEv, id: Math.random() }] }))
    setShowInject(false)
    try {
      await fetch(`${API_BASE}/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_type: EVENT_TYPE_MAP[newEv.type] || newEv.type,
          zone: zoneName,
          severity: newEv.severity,
          timestamp: new Date().toISOString(),
          source: "frontend"
        })
      })
    } catch (_) {}
    fetchZoneRisk(zoneName, sel)
  }, [sel, newEv, zones, fetchZoneRisk])

  const TABS = ["Overview", "Analysis", "Resources", "Reports"]
  const FF   = { fontFamily: "'IBM Plex Sans', sans-serif" }
  const FM   = { fontFamily: "'IBM Plex Mono', monospace" }
  const P    = { background: T.panel, border: `1px solid ${T.border}`, borderRadius: 8, ...FF }

  return (
    <div style={{ ...FF, background: T.bg, height: "100vh", color: T.text, display: "flex", flexDirection: "column", overflow: "hidden" }}>

      {/* ── GLOBAL HEADER ─────────────────────────────────────────────────── */}
      <div style={{ height: 64, flexShrink: 0, background: T.panel, borderBottom: `1px solid ${T.border}`, display: "flex", alignItems: "stretch", zIndex: 100 }}>
        <div style={{ padding: "0 28px", display: "flex", alignItems: "center", borderRight: `1px solid ${T.border}`, minWidth: 220 }}>
          <Activity color={T.blue} size={20} style={{ marginRight: 8 }} />
          <span style={{ fontSize: 18, fontWeight: 700, letterSpacing: 1, color: T.text }}>Pune Nexus</span>
        </div>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: "0 28px", border: "none", cursor: "pointer",
            background: tab === t ? T.bg : "transparent",
            borderBottom: tab === t ? `3px solid ${T.blue}` : "3px solid transparent",
            color: tab === t ? T.text : T.muted, fontSize: 15,
            fontWeight: tab === t ? 600 : 500, ...FF,
            borderRight: `1px solid ${T.border}`, transition: "all .15s",
          }}>
            {t}
          </button>
        ))}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 24, padding: "0 28px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 10, height: 10, borderRadius: "50%", background: T.green, boxShadow: `0 0 8px ${T.green}`, animation: "pulse 2s infinite" }} />
            <span style={{ fontSize: 12, color: T.green, ...FM, fontWeight: 600 }}>LIVE</span>
          </div>
          <span style={{ fontSize: 12, color: T.muted, ...FM }}>{now.toLocaleTimeString()}</span>
        </div>
      </div>

      {/* ── TAB CONTENT ───────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        
        {/* 1. OVERVIEW TAB */}
        {tab === "Overview" && (
          <>
            <div style={{ width: 340, display: "flex", flexDirection: "column", flexShrink: 0 }}>
              <div style={{ padding: "24px", background: T.panel, borderBottom: `1px solid ${T.border}`, borderRight: `1px solid ${T.border}` }}>
                <div style={{ fontSize: 12, color: T.muted, letterSpacing: 1, textTransform: "uppercase", marginBottom: 6, fontWeight: 600 }}>Network Integrity</div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                  <div style={{ fontSize: 36, fontWeight: 700, color: opPct > 75 ? T.green : T.orange }}>{opPct}%</div>
                  <div style={{ fontSize: 14, color: T.label, fontWeight: 500 }}>Operational</div>
                </div>
                <div style={{ fontSize: 12, color: T.muted, marginTop: 8 }}>{zones.length - stableCount} elevated · {criticalCount} critical zones</div>
              </div>
              <SidebarZonesList zones={zones} sel={sel} setSel={setSel} mapLayer={mapLayer} />
            </div>
            <div style={{ flex: 1, position: "relative", display: "flex", flexDirection: "column" }}>
              
              <div style={{ height: 60, background: T.panel, borderBottom: `1px solid ${T.border}`, display: "flex", alignItems: "center", padding: "0 24px", gap: 20 }}>
                 <span style={{ fontSize: 13, fontWeight: 600, color: T.label }}>Active Casualty Layer:</span>
                 <div style={{ display: "flex", gap: 8 }}>
                    {[
                      { id: "overall", label: "Overall System" },
                      { id: "flood", label: "Surface Flooding" },
                      { id: "traffic", label: "Mobility Grid" },
                      { id: "emergency", label: "Response Delay" }
                    ].map(layer => (
                      <button 
                        key={layer.id} 
                        onClick={() => setMapLayer(layer.id)}
                        style={{
                          padding: "6px 12px", borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: "pointer",
                          border: `1px solid ${mapLayer === layer.id ? T.blue : T.border}`,
                          background: mapLayer === layer.id ? T.blue + "15" : T.bg,
                          color: mapLayer === layer.id ? T.blue : T.muted,
                          transition: "all 0.2s"
                        }}
                      >
                        {layer.label}
                      </button>
                    ))}
                 </div>
              </div>

              <div style={{ flex: 1, position: "relative" }}>
                <TileMap zones={zones} selected={sel} onSelect={setSel} mapLayer={mapLayer} />
                
                <div style={{ position: "absolute", top: 20, left: 20, zIndex: 400, background: T.panel, padding: "16px 20px", borderRadius: 8, border: `1px solid ${T.border}`, boxShadow: "0 4px 16px rgba(0,0,0,0.08)" }}>
                  <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 10, color: T.label, letterSpacing: 1, textTransform: "uppercase" }}>{mapLayer} THREAT</div>
                  <div style={{ display: "flex", gap: 16 }}>
                    {[["Stable", T.green], ["Elevated", T.orange], ["Critical", T.red]].map(([l, c]) => (
                      <div key={l} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <div style={{ width: 14, height: 14, background: c, borderRadius: 3, opacity: 0.6, border: `1px solid ${c}` }} />
                        <span style={{ fontSize: 13, color: T.text, fontWeight: 500 }}>{l}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* 2. ANALYSIS TAB */}
        {tab === "Analysis" && (
          <>
            <div style={{ width: 340, display: "flex", flexDirection: "column", flexShrink: 0 }}>
              <SidebarZonesList zones={zones} sel={sel} setSel={setSel} mapLayer={mapLayer} />
            </div>
            <div style={{ flex: 1, overflowY: "auto", padding: "40px" }}>
              <div style={{ maxWidth: 900, margin: "0 auto" }}>
                
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 32 }}>
                  <div>
                    <h1 style={{ fontSize: 36, fontWeight: 700, color: T.text, marginBottom: 12 }}>{zone.name} Diagnostics</h1>
                    <div style={{ display: "flex", gap: 20, fontSize: 14, color: T.muted }}>
                      <span>Population: <strong style={{color: T.text}}>{zone.pop}</strong></span>
                      <span>Area: <strong style={{color: T.text}}>{zone.area}</strong></span>
                      <span>Confidence: <strong style={{color: T.text}}>{72 + pct(risks.overall * 0.2)}%</strong></span>
                    </div>
                  </div>
                  <div style={{ padding: "8px 20px", borderRadius: 6, fontSize: 16, fontWeight: 700, background: riskColor(risks.overall) + "15", border: `1px solid ${riskColor(risks.overall)}40`, color: riskColor(risks.overall) }}>
                    {riskLabel(risks.overall)}: {pct(risks.overall)}%
                  </div>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 24, marginBottom: 32 }}>
                  {[
                    { label: "Flood Risk", v: pct(risks.flood), c: T.blue },
                    { label: "Traffic Risk", v: pct(risks.traffic), c: T.orange },
                    { label: "Emergency Delay", v: pct(risks.emergency), c: T.red },
                  ].map(s => (
                    <div key={s.label} style={{ ...P, padding: "24px", boxShadow: "0 2px 10px rgba(0,0,0,0.02)" }}>
                      <div style={{ fontSize: 12, color: T.muted, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10, fontWeight: 600 }}>{s.label}</div>
                      <div style={{ fontSize: 42, fontWeight: 700, ...FM, color: s.c, lineHeight: 1 }}>{s.v}%</div>
                      <div style={{ height: 6, background: T.bg, borderRadius: 3, marginTop: 16 }}>
                        <div style={{ height: "100%", width: `${s.v}%`, background: s.c, borderRadius: 3 }} />
                      </div>
                    </div>
                  ))}
                </div>

                <div style={{ ...P, padding: "28px", marginBottom: 32, boxShadow: "0 2px 10px rgba(0,0,0,0.02)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                    <span style={{ fontSize: 18, fontWeight: 700 }}>Risk Propagation Timeline</span>
                    <div style={{ display: "flex", gap: 16 }}>
                      {[["Yesterday", T.muted], ["Overall", T.red], ["Flood", T.blue], ["Traffic", T.orange]].map(([l, c]) => (
                        <div key={l} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <div style={{ width: 14, height: 4, background: c, borderRadius: 2 }} />
                          <span style={{ fontSize: 13, color: T.label, fontWeight: 500 }}>{l}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={zone.history} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                      <CartesianGrid stroke={T.border} strokeDasharray="4 4" vertical={false} />
                      <XAxis dataKey="label" tick={{ fill: T.muted, fontSize: 12, fontFamily: "IBM Plex Mono" }} interval={3} />
                      <YAxis domain={[0, 1]} tickFormatter={v => `${pct(v)}%`} tick={{ fill: T.muted, fontSize: 12, fontFamily: "IBM Plex Mono" }} />
                      <Tooltip content={<CT />} />
                      <Line dataKey="yesterday" name="Yesterday" stroke={T.muted}  strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                      <Line dataKey="overall"   name="Overall"   stroke={T.red}    strokeWidth={3} dot={false} isAnimationActive={false} />
                      <Line dataKey="flood"     name="Flood"     stroke={T.blue}   strokeWidth={2} dot={false} isAnimationActive={false} />
                      <Line dataKey="traffic"   name="Traffic"   stroke={T.orange} strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ ...P, padding: "28px", boxShadow: "0 2px 10px rgba(0,0,0,0.02)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
                    <span style={{ fontSize: 18, fontWeight: 700 }}>Live Causal Triggers</span>
                    <button onClick={() => setShowInject(true)} style={{ padding: "8px 20px", borderRadius: 6, border: "none", background: T.text, color: "#fff", fontSize: 14, cursor: "pointer", fontWeight: 600, transition: "all 0.2s" }}>+ Inject Event</button>
                  </div>
                  {zone.events.length === 0 ? (
                    <div style={{ padding: "40px", textAlign: "center", background: T.bg, borderRadius: 8, color: T.muted, fontSize: 15, border: `1px dashed ${T.borderL}` }}>
                      No active triggers. Causal engine confirms zone stability.
                    </div>
                  ) : (
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      {zone.events.map(ev => (
                        <div key={ev.id} style={{ display: "flex", alignItems: "center", gap: 20, padding: "16px 20px", border: `1px solid ${T.border}`, borderRadius: 8, background: T.bg }}>
                          <div style={{ width: 14, height: 14, borderRadius: "50%", background: ev.severity === "high" ? T.red : ev.severity === "medium" ? T.orange : T.green, flexShrink: 0 }} />
                          <div style={{ flex: 1 }}>
                            <div style={{ fontSize: 15, fontWeight: 700, textTransform: "capitalize", marginBottom: 4, color: T.text }}>{ev.type.replace('_', ' ')}</div>
                            <div style={{ fontSize: 13, color: T.muted }}>Detected via sensor grid. Downstream impacts currently propagating.</div>
                          </div>
                          <span style={{ fontSize: 12, color: T.text, background: T.panel, border: `1px solid ${T.borderL}`, padding: "6px 12px", borderRadius: 6, textTransform: "uppercase", fontWeight: 700 }}>{ev.severity}</span>
                          <button onClick={() => setZones(prev => prev.map((z, i) => i !== sel ? z : { ...z, events: z.events.filter(e => e.id !== ev.id) }))}
                            style={{ background: "none", border: "none", color: T.muted, cursor: "pointer", fontSize: 18, padding: "4px" }}>✕</button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

              </div>
            </div>
          </>
        )}

        {/* 3. RESOURCES TAB */}
        {tab === "Resources" && (
          <div style={{ flex: 1, display: "flex" }}>
            <SidebarZonesList zones={zones} sel={sel} setSel={setSel} mapLayer={mapLayer} />
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflowY: "auto" }}>
              <div style={{ padding: "32px 32px 0", borderBottom: `1px solid ${T.border}` }}>
                <h1 style={{ fontSize: 24, fontWeight: 700, color: T.text, marginBottom: 8 }}>Resource Management</h1>
                <p style={{ fontSize: 14, color: T.muted, marginBottom: 24 }}>
                  Optimize resource deployment and simulate intervention impacts for {zone.name}
                </p>
              </div>

              <div style={{ flex: 1, padding: "0 32px 32px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, alignItems: "start" }}>
                <div>
                  <ResourceOptimizationPanel zoneName={zone.name} />
                  <EventTimeline zoneName={zone.name} />
                </div>
                <div>
                  <InterventionControls zoneName={zone.name} onRiskUpdate={handleInterventionUpdate} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 4. REPORTS TAB */}
        {tab === "Reports" && (
          <div style={{ flex: 1, overflowY: "auto", padding: "40px" }}>
            <div style={{ maxWidth: 1000, margin: "0 auto" }}>
              <h1 style={{ fontSize: 32, fontWeight: 700, color: T.text, marginBottom: 32 }}>System Reports & Causal Mapping</h1>
              
              <div style={{ marginBottom: 32 }}>
                <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, color: T.label, textTransform: "uppercase", letterSpacing: 1 }}>Live Causal Chain Trajectory</h2>
                <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 24 }}>
                  <ShieldAlert size={20} color={T.blue} style={{ marginTop: 2 }}/>
                  <p style={{ fontSize: 14, color: T.muted, lineHeight: 1.6, flex: 1 }}>
                    This graph is strictly dynamic. It aggregates real-time events across all zones to trace the dominant cause-and-effect reaction occurring in the city right now. <strong>Click any node</strong> to instantly switch to the Overview Map and isolate that specific casualty layer geographically.
                  </p>
                </div>
                <DynamicCausalGraph zones={zones} onNodeClick={handleGraphNodeClick} />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
                {/* Inventory Card */}
                <div style={{ ...P, padding: "28px", boxShadow: "0 2px 10px rgba(0,0,0,0.02)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                    <span style={{ fontSize: 18, fontWeight: 700 }}>Active Sensor Inventory</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                    {[[T.blue, "Flood Sensors", "1,842"], [T.orange, "Traffic Probes", "2,215"], [T.green, "Control Units", "348"], ["#805ad5", "Weather Stations", "62"]].map(([c, l, v]) => (
                      <div key={l} style={{ padding: "20px", background: T.bg, borderRadius: 8, border: `1px solid ${T.border}` }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                          <div style={{ width: 12, height: 12, borderRadius: 3, background: c }} />
                          <div style={{ fontSize: 13, color: T.label, fontWeight: 600 }}>{l}</div>
                        </div>
                        <div style={{ fontSize: 28, fontWeight: 700, ...FM, color: T.text }}>{v}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Alerts Log */}
                <div style={{ ...P, padding: "28px", boxShadow: "0 2px 10px rgba(0,0,0,0.02)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
                    <span style={{ fontSize: 18, fontWeight: 700 }}>Global Alerts Log</span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    {[
                      { type: "critical", text: "Flood risk elevated — Katraj", time: "10:42, 11 Mar" },
                      { type: "critical", text: "Drainage failure detected — Warje", time: "10:38, 11 Mar" },
                      { type: "minor", text: "Construction delay update — Kothrud", time: "10:30, 11 Mar" },
                      { type: "minor", text: "Traffic sensor offline — Hadapsar", time: "10:15, 11 Mar" }
                    ].map((a, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 16, padding: "16px", background: T.bg, borderRadius: 8, border: `1px solid ${T.border}` }}>
                        {a.type === "critical"
                          ? <AlertTriangle size={18} color={T.red} style={{ flexShrink: 0, marginTop: 2 }} />
                          : <div style={{ width: 12, height: 12, background: T.borderL, borderRadius: "50%", flexShrink: 0, marginTop: 5 }} />}
                        <div>
                          <div style={{ fontSize: 14, color: T.text, fontWeight: 600, marginBottom: 6 }}>{a.text}</div>
                          <div style={{ fontSize: 12, color: T.muted, ...FM }}>{a.time}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

      </div>

      {/* ── INJECT MODAL ──────────────────────────────────────────────────── */}
      {showInject && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 2000 }}
          onClick={() => setShowInject(false)}>
          <div style={{ background: T.panel, border: `1px solid ${T.borderL}`, borderRadius: 12, padding: 32, width: 440, boxShadow: "0 20px 40px rgba(0,0,0,0.15)" }}
            onClick={e => e.stopPropagation()}>
            <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 28, color: T.text }}>
              Inject Causal Event
              <div style={{ fontSize: 14, color: T.muted, fontWeight: 500, marginTop: 6 }}>Targeting Zone: <strong style={{color: T.text}}>{zone.name}</strong></div>
            </div>
            
            <div style={{ fontSize: 12, color: T.label, fontWeight: 700, letterSpacing: 1, marginBottom: 12 }}>EVENT CLASSIFICATION</div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 28 }}>
              {EVENT_TYPES.map(t => (
                <button key={t} onClick={() => setNewEv(p => ({ ...p, type: t }))} style={{
                  ...FF, padding: "10px 14px", borderRadius: 6, fontSize: 13, cursor: "pointer", fontWeight: 600, textTransform: "capitalize",
                  border: `2px solid ${newEv.type === t ? T.blue : T.border}`,
                  background: newEv.type === t ? T.blue + "15" : T.bg,
                  color: newEv.type === t ? T.blue : T.label,
                  transition: "all 0.15s"
                }}>{t.replace('_', ' ')}</button>
              ))}
            </div>
            
            <div style={{ fontSize: 12, color: T.label, fontWeight: 700, letterSpacing: 1, marginBottom: 12 }}>THREAT SEVERITY</div>
            <div style={{ display: "flex", gap: 12, marginBottom: 36 }}>
              {SEVERITIES.map(s => {
                const c = s === "high" ? T.red : s === "medium" ? T.orange : T.green
                return (
                  <button key={s} onClick={() => setNewEv(p => ({ ...p, severity: s }))} style={{
                    ...FF, flex: 1, padding: "12px", borderRadius: 6, fontSize: 14, fontWeight: 700,
                    cursor: "pointer", textTransform: "uppercase",
                    border: `2px solid ${newEv.severity === s ? c : T.border}`,
                    background: newEv.severity === s ? c + "15" : T.bg,
                    color: newEv.severity === s ? c : T.label,
                    transition: "all 0.15s"
                  }}>{s}</button>
                )
              })}
            </div>
            
            <div style={{ display: "flex", gap: 16 }}>
              <button onClick={() => setShowInject(false)} style={{ ...FF, flex: 1, padding: "14px", borderRadius: 8, border: `1px solid ${T.borderL}`, background: T.bg, color: T.text, fontSize: 15, cursor: "pointer", fontWeight: 600 }}>Cancel</button>
              <button onClick={injectEvent} style={{ ...FF, flex: 1, padding: "14px", borderRadius: 8, border: "none", background: T.text, color: "#fff", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>Confirm Injection</button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        button:focus, input:focus { outline: none; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${T.borderL}; border-radius: 4px; border: 2px solid ${T.bg}; }
      `}</style>
    </div>
  )
}