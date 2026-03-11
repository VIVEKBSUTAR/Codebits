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
const P  = { background: T.panel, border: `1px solid ${T.border}`, borderRadius: 8, ...FF }

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

// Sub-zone precise locations for targeted focusing (ready for real-time API coordinates)
const ZONE_SUB_LOCATIONS = {
  "Bibwewadi": {
    flood: { lat: 18.4582, lng: 73.8620, zoom: 19, label: "Karve Road Drainage", type: "flood" },
    traffic: { lat: 18.4585, lng: 73.8625, zoom: 19, label: "Bibwewadi Junction", type: "traffic" },
    emergency: { lat: 18.4580, lng: 73.8618, zoom: 19, label: "Sahakarnagar Hospital Route", type: "emergency" },
    overall: { lat: 18.4582, lng: 73.8620, zoom: 17, label: "Bibwewadi Center", type: "overall" }
  },
  "Katraj": {
    flood: { lat: 18.4440, lng: 73.8635, zoom: 19, label: "Katraj Lake Overflow", type: "flood" },
    traffic: { lat: 18.4435, lng: 73.8640, zoom: 19, label: "Pune-Satara Road", type: "traffic" },
    emergency: { lat: 18.4445, lng: 73.8630, zoom: 19, label: "Katraj Hospital Access", type: "emergency" },
    overall: { lat: 18.4437, lng: 73.8638, zoom: 17, label: "Katraj Center", type: "overall" }
  },
  "Hadapsar": {
    flood: { lat: 18.4975, lng: 73.9260, zoom: 19, label: "Mutha River Banks", type: "flood" },
    traffic: { lat: 18.4985, lng: 73.9255, zoom: 19, label: "Hadapsar IT Hub", type: "traffic" },
    emergency: { lat: 18.4970, lng: 73.9265, zoom: 19, label: "Magarpatta Emergency Route", type: "emergency" },
    overall: { lat: 18.4980, lng: 73.9258, zoom: 17, label: "Hadapsar Center", type: "overall" }
  },
  "Kothrud": {
    flood: { lat: 18.5070, lng: 73.8080, zoom: 19, label: "Kothrud Drainage Canal", type: "flood" },
    traffic: { lat: 18.5078, lng: 73.8085, zoom: 19, label: "Mumbai-Pune Highway", type: "traffic" },
    emergency: { lat: 18.5065, lng: 73.8075, zoom: 19, label: "Kothrud Hospital Junction", type: "emergency" },
    overall: { lat: 18.5074, lng: 73.8077, zoom: 17, label: "Kothrud Center", type: "overall" }
  },
  "Warje": {
    flood: { lat: 18.4830, lng: 73.8120, zoom: 19, label: "Warje Bridge Underpass", type: "flood" },
    traffic: { lat: 18.4835, lng: 73.8115, zoom: 19, label: "Warje Circle", type: "traffic" },
    emergency: { lat: 18.4825, lng: 73.8125, zoom: 19, label: "Warje Emergency Access", type: "emergency" },
    overall: { lat: 18.4832, lng: 73.8118, zoom: 17, label: "Warje Center", type: "overall" }
  },
  "Sinhagad": {
    flood: { lat: 18.4320, lng: 73.8100, zoom: 19, label: "Sinhagad Road Drainage", type: "flood" },
    traffic: { lat: 18.4330, lng: 73.8090, zoom: 19, label: "Sinhagad Road Main", type: "traffic" },
    emergency: { lat: 18.4315, lng: 73.8105, zoom: 19, label: "Sinhagad Medical Access", type: "emergency" },
    overall: { lat: 18.4326, lng: 73.8095, zoom: 17, label: "Sinhagad Center", type: "overall" }
  },
  "Pimpri": {
    flood: { lat: 18.6285, lng: 73.8005, zoom: 19, label: "Pimpri Creek Area", type: "flood" },
    traffic: { lat: 18.6275, lng: 73.8015, zoom: 19, label: "Pimpri PCMC Junction", type: "traffic" },
    emergency: { lat: 18.6290, lng: 73.8000, zoom: 19, label: "Pimpri Hospital Route", type: "emergency" },
    overall: { lat: 18.6279, lng: 73.8009, zoom: 17, label: "Pimpri Center", type: "overall" }
  }
}

// Real-time API integration utilities (ready for live data feeds)
const REAL_TIME_API = {
  // Example structure for real-time incident data
  processIncidentUpdate: (incidentData) => {
    // incidentData structure for real-time APIs:
    // {
    //   id: "INC_12345",
    //   type: "flood" | "traffic" | "emergency",
    //   zone: "Bibwewadi",
    //   coordinates: { lat: 18.4582, lng: 73.8620 },
    //   severity: "high" | "medium" | "low",
    //   timestamp: "2024-01-15T09:30:00Z",
    //   status: "active" | "resolved"
    // }

    // Map incident to sub-location for precise focusing
    const incident = {
      zone: incidentData.zone,
      metric: incidentData.type,
      location: {
        lat: incidentData.coordinates.lat,
        lng: incidentData.coordinates.lng,
        zoom: 19,
        label: `${incidentData.type.toUpperCase()} - ${incidentData.zone}`,
        type: incidentData.type,
        timestamp: Date.now()
      }
    }

    return incident
  },

  // WebSocket connection structure for real-time updates
  connectToRealTimeUpdates: (onUpdate) => {
    // Example WebSocket integration:
    // const ws = new WebSocket('ws://your-api/live-events')
    // ws.onmessage = (event) => {
    //   const incidentData = JSON.parse(event.data)
    //   const processedIncident = REAL_TIME_API.processIncidentUpdate(incidentData)
    //   onUpdate(processedIncident)
    // }
    // return ws

    console.log('Real-time API connection ready. Connect to your WebSocket/SSE endpoint here.')
  }
}

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

function TileMap({ zones, selected, onSelect, mapLayer, externalFocus }) {
  const containerRef = useRef(null)
  const [size,    setSize]   = useState({ w: 0, h: 0 })
  const [center,  setCenter] = useState({ lat: 18.53, lng: 73.82 })
  const [zoom,    setZoom]   = useState(12)
  const [tooltip, setTooltip]= useState(null)
  const [isTransitioning, setIsTransitioning] = useState(false)
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

  // Handle external focus updates with smooth transitions
  useEffect(() => {
    if (!externalFocus) return

    setIsTransitioning(true)

    // Smooth transition animation
    const startCenter = { ...center }
    const startZoom = zoom
    const targetCenter = { lat: externalFocus.lat, lng: externalFocus.lng }
    const targetZoom = externalFocus.zoom || 17

    const duration = 800 // ms
    const startTime = Date.now()

    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)

      // Easing function for smooth animation
      const easeProgress = 1 - Math.pow(1 - progress, 3)

      // Interpolate center coordinates
      const newLat = startCenter.lat + (targetCenter.lat - startCenter.lat) * easeProgress
      const newLng = startCenter.lng + (targetCenter.lng - startCenter.lng) * easeProgress
      const newZoom = startZoom + (targetZoom - startZoom) * easeProgress

      setCenter({ lat: newLat, lng: newLng })
      setZoom(newZoom)

      if (progress < 1) {
        requestAnimationFrame(animate)
      } else {
        setIsTransitioning(false)
      }
    }

    requestAnimationFrame(animate)
  }, [externalFocus])

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

  const zoomIn  = e => { e.stopPropagation(); setZoom(z => Math.min(20, z + 1)) }
  const zoomOut = e => { e.stopPropagation(); setZoom(z => Math.max(10, z - 1)) }

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const handler = e => {
      e.preventDefault()
      setZoom(z => e.deltaY < 0 ? Math.min(20, z + 1) : Math.max(10, z - 1))
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

          {/* GPS-style location marker for focused point */}
          {externalFocus && zoom >= 16 && (() => {
            const markerPos = latLngToScreen(externalFocus.lat, externalFocus.lng, center, zoom, size.w, size.h)

            // Only show marker if it's visible on screen
            if (markerPos.x < 0 || markerPos.x > size.w || markerPos.y < 0 || markerPos.y > size.h) {
              return null
            }

            const markerColor = externalFocus.type === 'flood' ? T.blue :
                               externalFocus.type === 'traffic' ? T.orange :
                               externalFocus.type === 'emergency' ? T.red : T.green

            return (
              <g key="location-marker">
                {/* Crosshair indicator */}
                <g style={{ opacity: isTransitioning ? 0.6 : 1, transition: "all 0.3s" }}>
                  <line x1={markerPos.x - 20} y1={markerPos.y} x2={markerPos.x + 20} y2={markerPos.y}
                        stroke={markerColor} strokeWidth="2" strokeDasharray="4 2" />
                  <line x1={markerPos.x} y1={markerPos.y - 20} x2={markerPos.x} y2={markerPos.y + 20}
                        stroke={markerColor} strokeWidth="2" strokeDasharray="4 2" />
                </g>

                {/* Pulsing outer circle */}
                <circle cx={markerPos.x} cy={markerPos.y} r="25" fill={markerColor} fillOpacity="0.2"
                        style={{ animation: "pulse 2s infinite" }} />

                {/* Main location pin */}
                <g transform={`translate(${markerPos.x}, ${markerPos.y - 12})`}>
                  {/* Pin shadow */}
                  <path d="M 2 14 L 0 12 C 0 5.4 5.4 0 12 0 S 24 5.4 24 12 L 22 14 L 12 28 L 2 14 Z"
                        fill="rgba(0,0,0,0.3)" />
                  {/* Pin body */}
                  <path d="M 0 12 C 0 5.4 5.4 0 12 0 S 24 5.4 24 12 L 12 26 L 0 12 Z"
                        fill={markerColor} stroke="#fff" strokeWidth="2" />
                  {/* Pin dot */}
                  <circle cx="12" cy="12" r="6" fill="#fff" />
                  <circle cx="12" cy="12" r="3" fill={markerColor} />
                </g>

                {/* Accuracy circle */}
                <circle cx={markerPos.x} cy={markerPos.y} r="15" fill="none"
                        stroke={markerColor} strokeWidth="1" strokeOpacity="0.5" strokeDasharray="2 3" />
              </g>
            )
          })()}
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

      {/* GPS-style location targeting panel */}
      {externalFocus && !isTransitioning && (() => {
        const focusColor = externalFocus.type === 'flood' ? T.blue :
                          externalFocus.type === 'traffic' ? T.orange :
                          externalFocus.type === 'emergency' ? T.red : T.green

        const typeIcon = externalFocus.type === 'flood' ? '💧' :
                        externalFocus.type === 'traffic' ? '🚗' :
                        externalFocus.type === 'emergency' ? '🚑' : '📍'

        return (
          <div style={{
            position: "absolute", top: 20, left: 20, zIndex: 400,
            background: "rgba(255,255,255,0.95)", backdropFilter: "blur(10px)",
            border: `2px solid ${focusColor}`, borderRadius: 12,
            boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
            animation: "fadeIn 0.3s ease-in-out", minWidth: 280
          }}>
            {/* Header */}
            <div style={{
              background: focusColor, color: "#fff", padding: "8px 16px",
              borderRadius: "10px 10px 0 0", fontSize: 12, fontWeight: 700,
              display: "flex", alignItems: "center", gap: 8
            }}>
              <span style={{ fontSize: 16 }}>{typeIcon}</span>
              <span>TARGET LOCATION</span>
            </div>

            {/* Location details */}
            <div style={{ padding: "12px 16px" }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: T.text, marginBottom: 8 }}>
                {externalFocus.label}
              </div>

              <div style={{ fontSize: 11, color: T.muted, fontFamily: "'IBM Plex Mono', monospace", lineHeight: 1.4 }}>
                <div>📍 {externalFocus.lat.toFixed(6)}, {externalFocus.lng.toFixed(6)}</div>
                <div>🔍 Zoom Level: {externalFocus.zoom}</div>
                <div>⏰ {new Date().toLocaleTimeString()}</div>
              </div>

              {/* Accuracy indicator */}
              <div style={{
                marginTop: 8, padding: "6px 10px", background: focusColor + "15",
                borderRadius: 6, border: `1px solid ${focusColor}40`,
                fontSize: 11, fontWeight: 600, color: focusColor
              }}>
                🎯 HIGH PRECISION TARGETING
              </div>
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

// ── AI-Powered Predictive Analytics Panel ──────────────────────────────────────
function PredictiveAnalyticsPanel({ zoneName, selectedMetric = "overall" }) {
  const [forecast, setForecast] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeMetric, setActiveMetric] = useState(selectedMetric)

  const METRICS = [
    { key: 'overall', label: 'Overall Risk', color: T.blue, icon: '📊' },
    { key: 'flood', label: 'Flood Risk', color: T.blue, icon: '💧' },
    { key: 'traffic', label: 'Traffic Risk', color: T.orange, icon: '🚗' },
    { key: 'emergency', label: 'Emergency Risk', color: T.red, icon: '🚑' }
  ]

  const fetchPredictiveForecast = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/predictive-forecast?zone=${encodeURIComponent(zoneName)}&metric=${activeMetric}&hours=24`)
      if (res.ok) {
        const data = await res.json()
        setForecast(data)
      }
    } catch (e) {
      console.error('Forecast error:', e)
    }
    setLoading(false)
  }, [zoneName, activeMetric])

  useEffect(() => {
    if (zoneName) fetchPredictiveForecast()
  }, [zoneName, activeMetric, fetchPredictiveForecast])

  if (loading) return (
    <div style={{...P, margin: 24, padding: 24, textAlign: "center"}}>
      <div style={{fontSize: 14, color: T.muted, marginBottom: 12}}>🤖 AI Analyzing Patterns...</div>
      <div style={{color: T.blue}}>Running Prophet + Anomaly Detection</div>
    </div>
  )

  if (!forecast) return (
    <div style={{...P, margin: 24, padding: 24, textAlign: "center", color: T.muted}}>
      No predictive data available
    </div>
  )

  const hasAnomalies = forecast.anomaly_detection?.upcoming_anomalies?.length > 0

  return (
    <div style={{...P, margin: 24}}>
      <div style={{display: "flex", alignItems: "center", gap: 12, marginBottom: 20}}>
        <h3 style={{fontSize: 16, fontWeight: 700, color: T.text, margin: 0}}>
          🚀 AI Predictive Analytics
        </h3>
        <div style={{
          padding: "4px 8px", borderRadius: 4, fontSize: 10, fontWeight: 700,
          background: T.green + "20", color: T.green
        }}>
          PROPHET + ANOMALY AI
        </div>
      </div>

      {/* Metric Selector */}
      <div style={{display: "flex", gap: 8, marginBottom: 16}}>
        {METRICS.map(metric => (
          <button
            key={metric.key}
            onClick={() => setActiveMetric(metric.key)}
            style={{
              padding: "6px 10px", borderRadius: 6, fontSize: 11, fontWeight: 600,
              background: activeMetric === metric.key ? metric.color + "20" : T.bg,
              color: activeMetric === metric.key ? metric.color : T.muted,
              border: `1px solid ${activeMetric === metric.key ? metric.color : T.border}`,
              cursor: "pointer", transition: "all 0.2s"
            }}
          >
            {metric.icon} {metric.label}
          </button>
        ))}
      </div>

      {/* Anomaly Alerts */}
      {hasAnomalies && (
        <div style={{
          padding: "12px 16px", marginBottom: 16, borderRadius: 6,
          background: T.red + "10", border: `1px solid ${T.red}40`
        }}>
          <div style={{fontSize: 12, fontWeight: 700, color: T.red, marginBottom: 8, display: "flex", alignItems: "center", gap: 6}}>
            🚨 ANOMALY DETECTED - IMMEDIATE ATTENTION REQUIRED
          </div>

          {forecast.anomaly_detection.upcoming_anomalies.slice(0, 3).map((anomaly, i) => (
            <div key={i} style={{
              fontSize: 11, color: T.text, marginBottom: 4,
              display: "flex", justifyContent: "space-between"
            }}>
              <span>⏰ {anomaly.timestamp}</span>
              <span>📈 {(anomaly.predicted_value * 100).toFixed(1)}%</span>
              <span style={{
                color: anomaly.severity === 'high' ? T.red : T.orange,
                fontWeight: 700, textTransform: "uppercase"
              }}>
                {anomaly.severity}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Forecast Chart */}
      <div style={{marginBottom: 16}}>
        <div style={{fontSize: 12, color: T.muted, marginBottom: 8}}>
          📊 24-Hour Risk Prediction with Confidence Intervals
        </div>

        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={forecast.predictions.slice(0, 12)}>
            <CartesianGrid strokeDasharray="3 3" stroke={T.borderL} />
            <XAxis dataKey="timestamp" fontSize={10} stroke={T.muted} />
            <YAxis domain={[0, 1]} tickFormatter={pct} fontSize={10} stroke={T.muted} />
            <Tooltip content={({ active, payload, label }) => {
              if (!active || !payload?.length) return null
              const data = payload[0].payload
              return (
                <div style={{
                  background: T.panel, border: `1px solid ${T.border}`, padding: "8px 12px",
                  borderRadius: 4, fontSize: 11, boxShadow: "0 4px 12px rgba(0,0,0,0.1)"
                }}>
                  <div style={{fontWeight: 700, marginBottom: 4}}>Time: {label}</div>
                  <div>Predicted: {pct(data.predicted_value)}%</div>
                  <div>Range: {pct(data.lower_bound)}% - {pct(data.upper_bound)}%</div>
                  <div>Confidence: ±{pct(data.confidence_interval)}%</div>
                </div>
              )
            }} />

            {/* Confidence band */}
            <Line
              dataKey="upper_bound"
              stroke={activeMetric === 'flood' ? T.blue : activeMetric === 'traffic' ? T.orange : T.red}
              strokeOpacity={0.3}
              strokeDasharray="2 3"
              dot={false}
            />
            <Line
              dataKey="lower_bound"
              stroke={activeMetric === 'flood' ? T.blue : activeMetric === 'traffic' ? T.orange : T.red}
              strokeOpacity={0.3}
              strokeDasharray="2 3"
              dot={false}
            />

            {/* Main prediction line */}
            <Line
              dataKey="predicted_value"
              stroke={activeMetric === 'flood' ? T.blue : activeMetric === 'traffic' ? T.orange : T.red}
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Model Performance */}
      <div style={{
        display: "flex", justifyContent: "space-between", fontSize: 11,
        color: T.muted, padding: "8px 12px", background: T.bg, borderRadius: 4
      }}>
        <span>📈 Training Samples: {forecast.model_performance.training_samples}</span>
        <span>🎯 Avg Uncertainty: ±{pct(forecast.model_performance.forecast_uncertainty)}%</span>
        <span>🔬 Method: {forecast.anomaly_detection.anomaly_method.toUpperCase()}</span>
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

// ── Computer Vision Infrastructure Assessment ──────────────────────────────────
function ComputerVisionPanel({ zoneName }) {
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)

  const analyzeImage = useCallback(async (file) => {
    setLoading(true)
    setAnalysis(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('zone', zoneName)

      const res = await fetch(`${API_BASE}/analyze-infrastructure-image?zone=${encodeURIComponent(zoneName)}`, {
        method: 'POST',
        body: formData
      })

      if (res.ok) {
        const data = await res.json()
        setAnalysis(data)
      } else {
        throw new Error('Analysis failed')
      }
    } catch (e) {
      console.error('CV analysis error:', e)
      setAnalysis({ analysis_status: 'error', error_message: 'Analysis failed' })
    }
    setLoading(false)
  }, [zoneName])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    const imageFile = files.find(f => f.type.startsWith('image/'))

    if (imageFile) {
      analyzeImage(imageFile)
    }
  }, [analyzeImage])

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0]
    if (file && file.type.startsWith('image/')) {
      analyzeImage(file)
    }
  }, [analyzeImage])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
  }, [])

  if (loading) {
    return (
      <div style={{...P, margin: 24, padding: 24, textAlign: "center"}}>
        <div style={{fontSize: 14, color: T.muted, marginBottom: 12}}>🤖 Analyzing Infrastructure...</div>
        <div style={{color: T.blue}}>Running YOLO + Damage Classification AI</div>
        <div style={{marginTop: 12, padding: "8px 12px", background: T.blue + "15", borderRadius: 4, fontSize: 11, color: T.blue, fontWeight: 600}}>
          Computer Vision Processing...
        </div>
      </div>
    )
  }

  return (
    <div style={{...P, margin: 24}}>
      <div style={{display: "flex", alignItems: "center", gap: 12, marginBottom: 20}}>
        <h3 style={{fontSize: 16, fontWeight: 700, color: T.text, margin: 0}}>
          📱 Computer Vision Assessment
        </h3>
        <div style={{
          padding: "4px 8px", borderRadius: 4, fontSize: 10, fontWeight: 700,
          background: T.orange + "20", color: T.orange
        }}>
          YOLO + AI DAMAGE ANALYSIS
        </div>
      </div>

      {!analysis && (
        <div
          style={{
            border: `2px dashed ${dragOver ? T.blue : T.border}`,
            borderRadius: 8, padding: "32px 24px", textAlign: "center",
            background: dragOver ? T.blue + "05" : T.bg,
            cursor: "pointer", transition: "all 0.3s"
          }}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => document.getElementById('file-input').click()}
        >
          <input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{display: "none"}}
          />

          <div style={{fontSize: 48, marginBottom: 16}}>📸</div>
          <div style={{fontSize: 14, fontWeight: 600, color: T.text, marginBottom: 8}}>
            Upload Infrastructure Image
          </div>
          <div style={{fontSize: 12, color: T.muted, lineHeight: 1.4}}>
            Drag & drop an image or click to select<br/>
            Supports: JPG, PNG, WEBP • Max 10MB<br/>
            AI will detect damage automatically
          </div>

          {dragOver && (
            <div style={{
              marginTop: 12, padding: "8px 12px", background: T.blue + "20",
              borderRadius: 4, fontSize: 11, color: T.blue, fontWeight: 600
            }}>
              Drop image to analyze with YOLO AI
            </div>
          )}
        </div>
      )}

      {analysis && analysis.analysis_status === 'success' && (
        <div>
          {/* Overall Assessment */}
          <div style={{
            padding: "16px", marginBottom: 16, borderRadius: 6,
            background: analysis.overall_assessment.risk_level === 'CRITICAL' ? T.red + "10" :
                       analysis.overall_assessment.risk_level === 'HIGH' ? T.orange + "10" :
                       T.green + "10",
            border: `1px solid ${
              analysis.overall_assessment.risk_level === 'CRITICAL' ? T.red + "40" :
              analysis.overall_assessment.risk_level === 'HIGH' ? T.orange + "40" :
              T.green + "40"
            }`
          }}>
            <div style={{display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12}}>
              <div style={{fontSize: 14, fontWeight: 700, color: T.text}}>
                📊 Overall Infrastructure Assessment
              </div>
              <div style={{
                padding: "4px 8px", borderRadius: 4, fontSize: 11, fontWeight: 700,
                background: analysis.overall_assessment.risk_level === 'CRITICAL' ? T.red + "20" :
                           analysis.overall_assessment.risk_level === 'HIGH' ? T.orange + "20" :
                           T.green + "20",
                color: analysis.overall_assessment.risk_level === 'CRITICAL' ? T.red :
                       analysis.overall_assessment.risk_level === 'HIGH' ? T.orange :
                       T.green
              }}>
                {analysis.overall_assessment.risk_level} RISK
              </div>
            </div>

            <div style={{fontSize: 11, color: T.muted, marginBottom: 8}}>
              Damage Score: {(analysis.overall_assessment.damage_score * 100).toFixed(1)}% •
              Critical Issues: {analysis.overall_assessment.critical_issues} •
              Total Detections: {analysis.overall_assessment.total_detections}
            </div>

            <div style={{fontSize: 12, fontWeight: 600, color: T.text}}>
              Priority: {analysis.overall_assessment.priority.replace(/_/g, ' ')}
            </div>
          </div>

          {/* Annotated Image */}
          {analysis.annotated_image && (
            <div style={{marginBottom: 16}}>
              <div style={{fontSize: 12, color: T.muted, marginBottom: 8}}>
                🖼️ AI-Annotated Image with Damage Detection
              </div>
              <img
                src={analysis.annotated_image}
                alt="Annotated infrastructure"
                style={{
                  width: "100%", maxHeight: 300, objectFit: "contain",
                  borderRadius: 6, border: `1px solid ${T.border}`
                }}
              />
            </div>
          )}

          {/* Damage Detections */}
          {analysis.damage_detections?.length > 0 && (
            <div style={{marginBottom: 16}}>
              <div style={{fontSize: 12, color: T.muted, marginBottom: 8}}>
                🔍 Detected Infrastructure Issues ({analysis.damage_detections.length})
              </div>

              {analysis.damage_detections.slice(0, 3).map((detection, i) => (
                <div key={i} style={{
                  padding: "8px 12px", marginBottom: 8, borderRadius: 4,
                  background: T.bg, border: `1px solid ${T.border}`,
                  display: "flex", justifyContent: "space-between", alignItems: "center"
                }}>
                  <div>
                    <div style={{fontSize: 12, fontWeight: 600, color: T.text}}>
                      {detection.object_type.replace(/_/g, ' ')} - {detection.damage_type.replace(/_/g, ' ')}
                    </div>
                    <div style={{fontSize: 10, color: T.muted}}>
                      {detection.damage_description}
                    </div>
                  </div>
                  <div style={{textAlign: "right"}}>
                    <div style={{fontSize: 11, color: T.orange, fontWeight: 600}}>
                      {detection.estimated_repair_cost}
                    </div>
                    <div style={{fontSize: 10, color: T.muted}}>
                      {(detection.severity_score * 100).toFixed(0)}% severity
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Recommendations */}
          {analysis.recommendations?.length > 0 && (
            <div>
              <div style={{fontSize: 12, color: T.muted, marginBottom: 8}}>
                💡 AI Recommendations
              </div>

              {analysis.recommendations.slice(0, 4).map((rec, i) => (
                <div key={i} style={{
                  fontSize: 11, color: T.text, marginBottom: 4,
                  padding: "4px 8px", background: T.bg, borderLeft: `3px solid ${T.blue}`,
                  borderRadius: 2
                }}>
                  {rec}
                </div>
              ))}
            </div>
          )}

          <button
            onClick={() => setAnalysis(null)}
            style={{
              marginTop: 16, padding: "8px 16px", borderRadius: 6,
              background: T.blue, color: "#fff", border: "none",
              fontSize: 12, fontWeight: 600, cursor: "pointer"
            }}
          >
            📸 Analyze Another Image
          </button>
        </div>
      )}

      {analysis && analysis.analysis_status === 'error' && (
        <div style={{
          padding: "16px", borderRadius: 6, textAlign: "center",
          background: T.red + "10", border: `1px solid ${T.red}40`
        }}>
          <div style={{fontSize: 14, color: T.red, fontWeight: 600, marginBottom: 8}}>
            Analysis Failed
          </div>
          <div style={{fontSize: 12, color: T.muted, marginBottom: 12}}>
            {analysis.error_message}
          </div>
          <button
            onClick={() => setAnalysis(null)}
            style={{
              padding: "6px 12px", borderRadius: 4, background: T.red,
              color: "#fff", border: "none", fontSize: 11, cursor: "pointer"
            }}
          >
            Try Again
          </button>
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

// ── 3D City Visualization ──────────────────────────────────────────────────────
function City3DVisualization({ zones, selectedZone, mapLayer, onZoneSelect }) {
  const canvasRef = useRef(null)
  const sceneRef = useRef(null)
  const rendererRef = useRef(null)
  const cameraRef = useRef(null)
  const animationFrameRef = useRef(null)
  const [isInitialized, setIsInitialized] = useState(false)
  const [loading, setLoading] = useState(true)

  // Initialize Three.js scene
  useEffect(() => {
    const initScene = async () => {
      if (!canvasRef.current) return

      try {
        // Dynamically import Three.js to avoid SSR issues
        const THREE = await import('three')

        // Scene setup
        const scene = new THREE.Scene()
        scene.fog = new THREE.Fog(0x87CEEB, 1000, 4000) // Atmospheric fog

        // Camera setup (cinematic angle)
        const camera = new THREE.PerspectiveCamera(
          60, // FOV for cinematic feel
          canvasRef.current.clientWidth / canvasRef.current.clientHeight,
          0.1,
          5000
        )
        camera.position.set(0, 600, 800) // Elevated cinematic view
        camera.lookAt(0, 0, 0)

        // Renderer setup with high quality
        const renderer = new THREE.WebGLRenderer({
          canvas: canvasRef.current,
          antialias: true,
          alpha: true,
          powerPreference: "high-performance"
        })
        renderer.setSize(canvasRef.current.clientWidth, canvasRef.current.clientHeight)
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
        renderer.shadowMap.enabled = true
        renderer.shadowMap.type = THREE.PCFSoftShadowMap
        renderer.outputColorSpace = THREE.SRGBColorSpace

        // Enhanced lighting setup
        const ambientLight = new THREE.AmbientLight(0x87CEEB, 0.4) // Soft blue ambient
        scene.add(ambientLight)

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
        directionalLight.position.set(100, 400, 200)
        directionalLight.castShadow = true
        directionalLight.shadow.mapSize.width = 2048
        directionalLight.shadow.mapSize.height = 2048
        scene.add(directionalLight)

        // City ground plane with grid
        const groundGeometry = new THREE.PlaneGeometry(2000, 2000, 20, 20)
        const groundMaterial = new THREE.MeshLambertMaterial({
          color: 0x2a3440,
          transparent: true,
          opacity: 0.8
        })
        const ground = new THREE.Mesh(groundGeometry, groundMaterial)
        ground.rotation.x = -Math.PI / 2
        ground.receiveShadow = true
        scene.add(ground)

        // City grid lines
        const gridHelper = new THREE.GridHelper(2000, 40, 0x4a5568, 0x374151)
        gridHelper.material.transparent = true
        gridHelper.material.opacity = 0.3
        scene.add(gridHelper)

        // Create city blocks for each zone
        const cityBlocks = createCityBlocks(zones, THREE)
        cityBlocks.forEach(block => scene.add(block))

        // Particle system for risk visualization
        const particles = createRiskParticles(zones, THREE)
        scene.add(particles)

        // Store references
        sceneRef.current = scene
        rendererRef.current = renderer
        cameraRef.current = camera

        setIsInitialized(true)
        setLoading(false)

        // Animation loop
        const animate = () => {
          animationFrameRef.current = requestAnimationFrame(animate)

          // Smooth camera rotation
          const time = Date.now() * 0.0001
          camera.position.x = Math.cos(time) * 600
          camera.position.z = Math.sin(time) * 600
          camera.lookAt(0, 0, 0)

          // Update particle effects
          if (particles && particles.children) {
            particles.children.forEach((particleSystem, i) => {
              if (particleSystem.material && particleSystem.material.uniforms) {
                particleSystem.material.uniforms.time.value = time * 10
              }
            })
          }

          renderer.render(scene, camera)
        }

        animate()

      } catch (error) {
        console.error('3D Scene initialization error:', error)
        setLoading(false)
      }
    }

    initScene()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [zones])

  // Update zone colors based on risk levels
  useEffect(() => {
    if (!sceneRef.current || !isInitialized) return

    sceneRef.current.children.forEach(child => {
      if (child.userData && child.userData.zoneId !== undefined) {
        const zone = zones[child.userData.zoneId]
        if (zone) {
          const risks = getRisks(zone)
          const riskValue = risks[mapLayer] || 0

          // Update building color based on risk
          let color
          if (riskValue > 0.7) color = 0xff4444      // Red
          else if (riskValue > 0.5) color = 0xff8800 // Orange
          else if (riskValue > 0.3) color = 0xffdd00 // Yellow
          else color = 0x44ff44                      // Green

          if (child.material) {
            child.material.color.setHex(color)
            child.material.emissive.setHex(color)
            child.material.emissiveIntensity = riskValue * 0.3
          }
        }
      }
    })
  }, [zones, mapLayer, isInitialized])

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!canvasRef.current || !cameraRef.current || !rendererRef.current) return

      const width = canvasRef.current.clientWidth
      const height = canvasRef.current.clientHeight

      cameraRef.current.aspect = width / height
      cameraRef.current.updateProjectionMatrix()
      rendererRef.current.setSize(width, height)
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [isInitialized])

  const createCityBlocks = (zones, THREE) => {
    const blocks = []

    zones.forEach((zone, index) => {
      // Calculate position based on lat/lng (simplified projection)
      const x = (zone.lng - 73.82) * 40000
      const z = (zone.lat - 18.48) * 40000

      // Building height based on population
      const population = parseInt(zone.pop.replace(/,/g, '')) || 100000
      const height = Math.max(20, (population / 10000) * 30)

      // Create building geometry
      const geometry = new THREE.CylinderGeometry(25, 35, height, 8)
      const material = new THREE.MeshPhongMaterial({
        transparent: true,
        opacity: 0.8,
        shininess: 100
      })

      const building = new THREE.Mesh(geometry, material)
      building.position.set(x, height / 2, z)
      building.castShadow = true
      building.receiveShadow = true
      building.userData = { zoneId: index, zoneName: zone.name }

      // Add zone label
      const labelCanvas = document.createElement('canvas')
      const labelContext = labelCanvas.getContext('2d')
      labelCanvas.width = 256
      labelCanvas.height = 64
      labelContext.font = '24px Arial'
      labelContext.fillStyle = '#ffffff'
      labelContext.textAlign = 'center'
      labelContext.fillText(zone.name, 128, 40)

      const labelTexture = new THREE.CanvasTexture(labelCanvas)
      const labelMaterial = new THREE.MeshBasicMaterial({
        map: labelTexture,
        transparent: true
      })
      const labelGeometry = new THREE.PlaneGeometry(60, 15)
      const label = new THREE.Mesh(labelGeometry, labelMaterial)
      label.position.set(x, height + 20, z)
      label.lookAt(0, height + 20, 100)

      blocks.push(building, label)
    })

    return blocks
  }

  const createRiskParticles = (zones, THREE) => {
    const particleGroup = new THREE.Group()

    zones.forEach((zone, index) => {
      const x = (zone.lng - 73.82) * 40000
      const z = (zone.lat - 18.48) * 40000

      // Particle system for risk visualization
      const particleCount = 100
      const positions = new Float32Array(particleCount * 3)
      const colors = new Float32Array(particleCount * 3)

      for (let i = 0; i < particleCount; i++) {
        positions[i * 3] = x + (Math.random() - 0.5) * 100
        positions[i * 3 + 1] = Math.random() * 100
        positions[i * 3 + 2] = z + (Math.random() - 0.5) * 100

        colors[i * 3] = Math.random()
        colors[i * 3 + 1] = Math.random() * 0.5
        colors[i * 3 + 2] = 1
      }

      const particleGeometry = new THREE.BufferGeometry()
      particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

      const particleMaterial = new THREE.PointsMaterial({
        size: 2,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
      })

      const particles = new THREE.Points(particleGeometry, particleMaterial)
      particleGroup.add(particles)
    })

    return particleGroup
  }

  if (loading) {
    return (
      <div style={{
        width: "100%", height: "400px", display: "flex", alignItems: "center",
        justifyContent: "center", background: T.bg, borderRadius: 8,
        border: `1px solid ${T.border}`
      }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 18, marginBottom: 12 }}>🌆</div>
          <div style={{ fontSize: 14, color: T.text, fontWeight: 600, marginBottom: 8 }}>
            Initializing 3D City Model
          </div>
          <div style={{ fontSize: 12, color: T.muted }}>
            Loading Three.js WebGL Renderer...
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={{
      width: "100%", height: "400px", position: "relative",
      borderRadius: 8, border: `1px solid ${T.border}`,
      overflow: "hidden", background: "linear-gradient(to bottom, #87CEEB, #98D8E8)"
    }}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block" }}
      />

      {/* 3D Controls Overlay */}
      <div style={{
        position: "absolute", top: 16, right: 16, zIndex: 100,
        background: "rgba(0,0,0,0.7)", backdropFilter: "blur(10px)",
        borderRadius: 8, padding: "8px 12px"
      }}>
        <div style={{ color: "#fff", fontSize: 11, fontWeight: 700, marginBottom: 4 }}>
          🌆 3D CITY MODEL
        </div>
        <div style={{ color: "#aaa", fontSize: 10 }}>
          Auto-rotating • Risk heatmaps • WebGL
        </div>
      </div>

      {/* Legend */}
      <div style={{
        position: "absolute", bottom: 16, left: 16, zIndex: 100,
        background: "rgba(255,255,255,0.9)", backdropFilter: "blur(10px)",
        borderRadius: 6, padding: "8px 12px", fontSize: 10
      }}>
        <div style={{ marginBottom: 4, fontWeight: 700, color: T.text }}>Risk Legend:</div>
        {[
          ['🟢 Low', '< 30%'],
          ['🟡 Medium', '30-50%'],
          ['🟠 High', '50-70%'],
          ['🔴 Critical', '> 70%']
        ].map(([label, range]) => (
          <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <span>{label}</span>
            <span style={{ color: T.muted }}>{range}</span>
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
  const [mapFocus,    setMapFocus]    = useState(null)

  const zone  = zones[sel]
  const risks = getRisks(zone, interventionRisks)

  const handleInterventionUpdate = useCallback((newRisks) => {
    setInterventionRisks(newRisks)
  }, [])

  const handleGraphNodeClick = useCallback((metric) => {
    setMapLayer(metric)
    setTab("Overview")

    // Find zone with highest risk for this metric and get precise sub-location
    let maxRisk = 0
    let targetZoneIndex = sel
    let targetLocation = null

    zones.forEach((z, i) => {
      const zoneRisk = getRisks(z, interventionRisks)[metric]
      if (zoneRisk > maxRisk) {
        maxRisk = zoneRisk
        targetZoneIndex = i

        // Get specific sub-location for this zone and metric
        const zoneSubLocations = ZONE_SUB_LOCATIONS[z.name]
        if (zoneSubLocations) {
          targetLocation = zoneSubLocations[metric]
        }
      }
    })

    // Switch to the zone with highest risk for this metric
    if (targetZoneIndex !== sel) {
      setSel(targetZoneIndex)
    }

    // Focus map on specific sub-location with smooth animation
    if (targetLocation) {
      setMapFocus({
        lat: targetLocation.lat,
        lng: targetLocation.lng,
        zoom: targetLocation.zoom,
        label: targetLocation.label,
        timestamp: Date.now() // Forces re-focus even if same coordinates
      })
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
                <TileMap zones={zones} selected={sel} onSelect={setSel} mapLayer={mapLayer} externalFocus={mapFocus} />
                
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
                  <PredictiveAnalyticsPanel zoneName={zone.name} selectedMetric={mapLayer} />
                  <EventTimeline zoneName={zone.name} />
                </div>
                <div>
                  <InterventionControls zoneName={zone.name} onRiskUpdate={handleInterventionUpdate} />
                  <ComputerVisionPanel zoneName={zone.name} />
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
        @keyframes fadeIn { 0%{opacity:0; transform: translateY(-10px)} 100%{opacity:1; transform: translateY(0)} }
        @keyframes pinBounce { 0%{transform: translateY(-5px)} 50%{transform: translateY(0px)} 100%{transform: translateY(-5px)} }
        @keyframes accuracyPulse { 0%{r:10; opacity:0.8} 50%{r:18; opacity:0.3} 100%{r:10; opacity:0.8} }
        @keyframes crosshairSpin { 0%{transform: rotate(0deg)} 100%{transform: rotate(360deg)} }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        button:focus, input:focus { outline: none; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: ${T.borderL}; border-radius: 4px; border: 2px solid ${T.bg}; }
      `}</style>
    </div>
  )
}