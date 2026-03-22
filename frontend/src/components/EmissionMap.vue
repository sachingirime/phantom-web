<template>
  <section id="detection-map" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Permian Basin · NASA AVIRIS-NG Dataset</span>
        <h2>Emission Detection Map</h2>
        <p>
          178 methane sources detected and quantified across the Permian Basin, Texas.
          Plume extents and emission rates derived from NASA AVIRIS-NG hyperspectral imagery.
          Wind-driven animation is based on coincident surface wind measurements.
        </p>
      </div>

      <div class="map-wrapper">
        <div ref="mapEl" class="map-container" />
        <div v-if="loading" class="map-loading">
          <div class="spinner" />
          <span>Loading detection data…</span>
        </div>
      </div>

      <div v-if="summary" class="metrics-bar">
        <div class="metric">
          <span class="metric-value">{{ summary.total_sites }}</span>
          <span class="metric-label">Detection Sites</span>
        </div>
        <div class="metric">
          <span class="metric-value" style="color:#ef4444">{{ summary.critical_count }}</span>
          <span class="metric-label">Super-Emitters</span>
        </div>
        <div class="metric">
          <span class="metric-value">{{ summary.total_kg_hr.toLocaleString() }}</span>
          <span class="metric-unit">kg/hr</span>
          <span class="metric-label">Total Emissions</span>
        </div>
        <div class="metric">
          <span class="metric-value">{{ summary.median_kg_hr }}</span>
          <span class="metric-unit">kg/hr</span>
          <span class="metric-label">Median Rate</span>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const mapEl   = ref(null)
const summary = ref(null)
const loading = ref(true)

// ── Gas plume colormap — GHGSat/AVIRIS-NG scientific style ──────────────────
// deep navy → electric blue → cyan → yellow → orange → red → white
function thermalRGB(t) {
  t = Math.max(0, Math.min(1, t))
  const stops = [
    [10,   0, 110],   // deep navy      (low enhancement)
    [0,   55, 230],   // electric blue
    [0,  200, 255],   // cyan
    [60, 255, 180],   // cyan-green
    [255, 230,   0],  // yellow
    [255,  70,   0],  // orange-red
    [255, 255, 255],  // white           (peak enhancement)
  ]
  const n   = stops.length - 1
  const idx = Math.min(Math.floor(t * n), n - 1)
  const frac = t * n - idx
  const a = stops[idx], b = stops[idx + 1]
  return [
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  ]
}

// ── Realistic Gaussian plume simulation ──────────────────────────────────────
// Inspired by 2-D Stable Fluids (advect + diffuse) but rendered on a 2-D canvas.
// Physics:
//   • Wind direction from measured wind_direction_deg (meteorological "from" convention)
//   • Color mapped to Q_kg_hr on a log scale (blue=low, red=high)
//   • Layer A — static Gaussian concentration backbone (σ grows with √distance)
//   • Layer B — animated Lagrangian puffs (RK2-like advection + Langevin lateral noise)
//   • Layer C — pulsing hot source core
//   • All layers clipped to the real polygon boundary

function animatePlumes(canvas, map, pane, emissions, qMin, qMax) {
  const ctx = canvas.getContext('2d')
  const startTime = performance.now()
  let raf
  const mapPaneEl = map.getPane('mapPane')

  // Log-scale normalisation: spreads the 16–3358 kg/hr range more evenly
  const logQMin = Math.log(qMin + 1)
  const logQMax = Math.log(qMax + 1)
  function normQ(q) {
    return Math.max(0, Math.min(1,
      (Math.log(q + 1) - logQMin) / (logQMax - logQMin + 1e-9)))
  }

  function frame() {
    const t = (performance.now() - startTime) / 1000
    const W = canvas.width, H = canvas.height

    // Counteract the map-pane's pan transform so the canvas stays viewport-aligned.
    // latLngToContainerPoint() then gives correct canvas coordinates.
    const pos = L.DomUtil.getPosition(mapPaneEl)
    pane.style.left = (-pos.x) + 'px'
    pane.style.top  = (-pos.y) + 'px'

    ctx.clearRect(0, 0, W, H)

    emissions.forEach(row => {
      if (!row.plume?.polygons?.length) return

      const src = map.latLngToContainerPoint(L.latLng(row.latitude, row.longitude))
      if (src.x < -300 || src.x > W + 300 || src.y < -300 || src.y > H + 300) return

      // Color by emission rate Q_kg_hr (log scale)
      const qN = normQ(row.Q_kg_hr)
      const [cr, cg, cb] = thermalRGB(qN)

      // Wind: gas moves OPPOSITE to meteorological "from" direction
      const windFromDeg = row.wind_direction_deg ?? 225
      const windToDeg   = (windFromDeg + 180) % 360
      const windToRad   = windToDeg * Math.PI / 180
      // Slow whole-plume meander: ±12° oscillation (real wind direction variance)
      const meander = 0.21 * Math.sin(t * 0.25 + row.latitude * 4.2)
      const effRad  = windToRad + meander
      // Screen: north = –y, east = +x
      const nx =  Math.sin(effRad)   // along-wind  x
      const ny = -Math.cos(effRad)   // along-wind  y  (screen y flips N/S)
      const lx =  ny                 // lateral     x  (perpendicular)
      const ly = -nx                 // lateral     y

      // Project all sub-polygons to screen
      const allPtSets = row.plume.polygons
        .map(poly => poly.map(([lat, lon]) => {
          const p = map.latLngToContainerPoint(L.latLng(lat, lon))
          return { x: p.x, y: p.y }
        }))
        .filter(pts => pts.length >= 3)
      if (!allPtSets.length) return

      // Bounding box → scale parameters
      const allPts = allPtSets.flat()
      const bx0 = Math.min(...allPts.map(p => p.x)), bx1 = Math.max(...allPts.map(p => p.x))
      const by0 = Math.min(...allPts.map(p => p.y)), by1 = Math.max(...allPts.map(p => p.y))
      const polyDiag = Math.sqrt((bx1 - bx0) ** 2 + (by1 - by0) ** 2)
      if (polyDiag < 3) return   // too small at current zoom

      const polyLen  = Math.max(polyDiag, 15)
      const sig0     = Math.max(polyDiag * 0.07, 4)   // initial σ at source
      const puffLife = 4.0                              // seconds per puff cycle
      const strength = 0.65 + qN * 0.35               // opacity multiplier (0.65–1.0)

      // Clip everything to the actual polygon boundary
      ctx.save()
      ctx.beginPath()
      allPtSets.forEach(pts => {
        ctx.moveTo(pts[0].x, pts[0].y)
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
        ctx.closePath()
      })
      ctx.clip()

      // ── Layer A: Dense concentration backbone ───────────────────────────────
      const N_SPINE = 42
      for (let k = 0; k < N_SPINE; k++) {
        const frac  = k / (N_SPINE - 1)
        const dist  = frac * polyLen
        const sigma = sig0 * (1 + Math.sqrt(frac) * 2.5)
        const conc  = Math.exp(-frac * 2.8)
        // Tight lateral meander — narrow coherent spine like real plume backbone
        const lat   = sigma * 0.08 * Math.sin(t * 0.28 + frac * Math.PI * 1.5)

        const bx = src.x + nx * dist + lx * lat
        const by = src.y + ny * dist + ly * lat
        const a  = conc * strength * 0.28
        if (a < 0.006) continue

        const rg = ctx.createRadialGradient(bx, by, 0, bx, by, sigma)
        rg.addColorStop(0,    `rgba(${cr},${cg},${cb},${Math.min(a, 0.82)})`)
        rg.addColorStop(0.45, `rgba(${cr},${cg},${cb},${a * 0.45})`)
        rg.addColorStop(1,    `rgba(${cr},${cg},${cb},0)`)
        ctx.fillStyle = rg
        ctx.beginPath(); ctx.arc(bx, by, sigma, 0, Math.PI * 2); ctx.fill()
      }

      // ── Layer B: Meandering Lagrangian puffs ─────────────────────────────
      // Multi-scale turbulent lateral displacement (three nested eddy scales):
      //   Large  (~25 s period): whole-plume meander visible as snaking motion
      //   Medium (~7 s period) : inter-puff mixing, plume width variation
      //   Small  (~2 s period) : fast turbulent shredding near source
      // σ grows with √frac (Fickian diffusion). Each puff has a unique phase
      // seeded by its index k so paths don't overlap.
      const N_PUFFS = 80
      for (let k = 0; k < N_PUFFS; k++) {
        const phase = k / N_PUFFS
        const frac  = (t / puffLife + phase) % 1
        const dist  = frac * polyLen
        const sigma = sig0 * (0.35 + Math.sqrt(frac) * 2.5)

        // Three-scale lateral turbulence — tightened amplitudes for denser, coherent plume
        const lat = sigma * (
          0.72 * Math.sin(t * 0.22 + k * 1.618 + phase * 3.14) +   // large slow eddy
          0.36 * Math.sin(t * 0.90 + k * 2.718 + frac  * 6.28) +   // medium eddy
          0.14 * Math.sin(t * 2.60 + k * 4.130 + frac  * 9.42)     // small fast eddy
        )

        const bx = src.x + nx * dist + lx * lat
        const by = src.y + ny * dist + ly * lat

        const conc    = Math.exp(-frac * 1.6)
        const fadeIn  = Math.min(frac / 0.04, 1)
        const fadeOut = frac > 0.82 ? Math.max(0, 1 - (frac - 0.82) / 0.18) : 1
        const a = conc * fadeIn * fadeOut * strength * 0.62
        if (a < 0.006) continue

        const r  = sigma * 1.1
        const rg = ctx.createRadialGradient(bx, by, 0, bx, by, r)
        rg.addColorStop(0,    `rgba(${cr},${cg},${cb},${Math.min(a, 0.95)})`)
        rg.addColorStop(0.32, `rgba(${cr},${cg},${cb},${a * 0.62})`)
        rg.addColorStop(0.68, `rgba(${cr},${cg},${cb},${a * 0.20})`)
        rg.addColorStop(1,    `rgba(${cr},${cg},${cb},0)`)
        ctx.fillStyle = rg
        ctx.beginPath(); ctx.arc(bx, by, r, 0, Math.PI * 2); ctx.fill()
      }

      // ── Layer C: Hot pulsing source core ──────────────────────────────────
      const pulse = 0.82 + 0.18 * Math.sin(t * 2.8 + row.latitude * 6)
      const srcR  = Math.max(sig0 * 1.3, 6)
      const sg = ctx.createRadialGradient(src.x, src.y, 0, src.x, src.y, srcR)
      sg.addColorStop(0,    `rgba(255,252,220,${0.92 * pulse})`)
      sg.addColorStop(0.30, `rgba(${cr},${cg},${cb},${0.72 * pulse})`)
      sg.addColorStop(0.65, `rgba(${cr},${cg},${cb},${0.20 * pulse})`)
      sg.addColorStop(1,    `rgba(${cr},${cg},${cb},0)`)
      ctx.fillStyle = sg
      ctx.beginPath(); ctx.arc(src.x, src.y, srcR, 0, Math.PI * 2); ctx.fill()

      ctx.restore()
    })

    raf = requestAnimationFrame(frame)
  }

  frame()
  return () => cancelAnimationFrame(raf)
}

// ── State ─────────────────────────────────────────────────────────────────────
let mapInstance = null
let stopAnim    = null

onMounted(async () => {
  const res  = await fetch('/data/emissions.json')
  const data = await res.json()
  summary.value = data.summary
  loading.value = false

  const emissions = data.emissions

  // Emission rate range for colormap normalization (log scale in animatePlumes)
  const qs   = emissions.map(r => r.Q_kg_hr)
  const qMin = Math.min(...qs)
  const qMax = Math.max(...qs)

  // ── Leaflet map ─────────────────────────────────────────────────────────────
  mapInstance = L.map(mapEl.value, {
    center: [32.33, -101.81],
    zoom: 12,
    scrollWheelZoom: true,
  })

  const satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: '© Esri, Maxar, Earthstar Geographics', maxZoom: 20 }
  )
  const streets = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CARTO', subdomains: 'abcd', maxZoom: 20,
  })
  satellite.addTo(mapInstance)
  L.control.layers({ 'Satellite': satellite, 'Street Map': streets }, {}, { position: 'topright' }).addTo(mapInstance)

  // ── Custom Leaflet pane for the plume canvas ─────────────────────────────────
  // All Leaflet panes live inside .leaflet-map-pane, which has a CSS transform
  // (translate3d) that creates its own stacking context. z-indices on pane
  // children are therefore evaluated WITHIN that stacking context:
  //   tile-pane: 200  |  overlay-pane: 400  |  marker-pane: 600  |  popup-pane: 700
  // A canvas at z-500 (inside this stacking context) sits above tiles/overlays
  // but below markers and popups — exactly what we need.
  //
  // Pan compensation: the map-pane transform shifts during drag. We counteract
  // it each frame via L.DomUtil.getPosition so the canvas stays viewport-aligned
  // while we can continue using latLngToContainerPoint for coordinates.
  mapInstance.createPane('plumesPane')
  const plumesPane = mapInstance.getPane('plumesPane')
  plumesPane.style.zIndex = '500'
  plumesPane.style.pointerEvents = 'none'

  const canvas = document.createElement('canvas')
  canvas.style.cssText = 'position:absolute;pointer-events:none;'
  plumesPane.appendChild(canvas)

  function resizeCanvas() {
    canvas.width  = mapEl.value.offsetWidth
    canvas.height = mapEl.value.offsetHeight
  }
  resizeCanvas()
  new ResizeObserver(resizeCanvas).observe(mapEl.value)

  // Start animation (re-projects polygons every frame — handles pan/zoom correctly)
  stopAnim = animatePlumes(canvas, mapInstance, plumesPane, emissions, qMin, qMax)

  // ── Markers + popups ────────────────────────────────────────────────────────
  emissions.forEach(row => {
    const isCritical  = row.emission_category === 'Critical'
    const markerColor = isCritical ? '#ef4444' : '#10b981'
    const size        = isCritical ? 22 : 14
    const glowColor   = isCritical ? 'rgba(239,68,68,0.35)' : 'rgba(16,185,129,0.35)'

    const plumeSection = row.plume ? `
      <tr style="border-bottom:1px solid #f1f5f9;">
        <td style="padding:6px 0;color:#64748b;">Plume Area</td>
        <td style="padding:6px 0;font-weight:500;">${row.plume.det_pixels.toLocaleString()} px²</td>
      </tr>
      <tr style="border-bottom:1px solid #f1f5f9;">
        <td style="padding:6px 0;color:#64748b;">Mean Confidence</td>
        <td style="padding:6px 0;font-weight:500;">${(row.plume.mean_conf * 100).toFixed(2)}%</td>
      </tr>
      <tr>
        <td style="padding:6px 0;color:#64748b;">Peak Confidence</td>
        <td style="padding:6px 0;font-weight:500;">${(row.plume.max_conf * 100).toFixed(2)}%</td>
      </tr>` : ''

    const popup = `
      <div style="font-family:inherit;width:300px;border-radius:12px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.15);">
        <div style="background:${markerColor};color:#fff;padding:14px 16px;">
          <div style="font-weight:700;font-size:0.9rem;opacity:0.85;">${row.emission_category} Emission</div>
          <div style="font-size:1.7rem;font-weight:800;margin-top:2px;">${row.Q_kg_hr.toFixed(1)}<span style="font-size:0.9rem;font-weight:500;"> kg/hr</span></div>
        </div>
        <div style="padding:14px 16px;background:#fff;">
          <table style="width:100%;font-size:0.82rem;border-collapse:collapse;color:#1e293b;">
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:6px 0;color:#64748b;">Date</td>
              <td style="padding:6px 0;font-weight:500;">${row.date_formatted}</td>
            </tr>
            <tr style="border-bottom:1px solid #f1f5f9;">
              <td style="padding:6px 0;color:#64748b;">Wind Speed</td>
              <td style="padding:6px 0;font-weight:500;">${row.U_10_ms.toFixed(2)} m/s</td>
            </tr>
            ${plumeSection}
          </table>
        </div>
      </div>`

    const icon = L.divIcon({
      html: `<div style="position:relative;width:${size}px;height:${size}px;">
        <div style="position:absolute;inset:-5px;background:${glowColor};border-radius:50%;"></div>
        <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
          <circle cx="${size/2}" cy="${size/2}" r="${size/2-1}" fill="${markerColor}" opacity="0.95" stroke="white" stroke-width="2"/>
        </svg>
      </div>`,
      className: '', iconSize: [size, size], iconAnchor: [size/2, size/2],
    })

    L.marker([row.latitude, row.longitude], { icon })
      .bindPopup(popup, { maxWidth: 320 })
      .bindTooltip(`<b>${row.Q_kg_hr.toFixed(1)} kg/hr</b> · ${row.emission_category}`)
      .addTo(mapInstance)
  })

  // ── Legend ──────────────────────────────────────────────────────────────────
  const legend = L.control({ position: 'bottomright' })
  legend.onAdd = () => {
    const div = L.DomUtil.create('div')
    div.innerHTML = `
      <div style="background:rgba(15,23,42,0.88);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:14px 16px;font-family:inherit;font-size:0.75rem;color:#fff;min-width:200px;">
        <div style="font-weight:700;margin-bottom:10px;letter-spacing:0.06em;text-transform:uppercase;color:#94a3b8;">Emission Rate</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
          <div style="flex:1;height:10px;border-radius:4px;background:linear-gradient(to right,rgb(10,0,110),rgb(0,55,230),rgb(0,200,255),rgb(60,255,180),rgb(255,230,0),rgb(255,70,0),rgb(255,255,255));"></div>
        </div>
        <div style="display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:14px;font-size:0.7rem;">
          <span>Low (${qMin.toFixed(0)} kg/hr)</span><span>High (${qMax.toFixed(0)} kg/hr)</span>
        </div>
        <div style="border-top:1px solid rgba(255,255,255,0.1);padding-top:10px;">
          <div style="font-weight:700;margin-bottom:8px;letter-spacing:0.06em;text-transform:uppercase;color:#94a3b8;">Emitter Markers</div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="width:14px;height:14px;background:#ef4444;border-radius:50%;display:inline-block;flex-shrink:0;box-shadow:0 0 6px rgba(239,68,68,0.5);"></span>
            <span>Super-Emitters (≥100 kg/hr)</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <span style="width:12px;height:12px;background:#10b981;border-radius:50%;display:inline-block;flex-shrink:0;margin-left:1px;box-shadow:0 0 6px rgba(16,185,129,0.5);"></span>
            <span>Standard (&lt;100 kg/hr)</span>
          </div>
          <div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;display:flex;flex-direction:column;gap:3px;color:#94a3b8;">
            <div>Sites: <span style="color:#fff;font-weight:600;">${data.summary.total_sites}</span></div>
            <div>Total: <span style="color:#fff;font-weight:600;">${data.summary.total_kg_hr.toLocaleString()} kg/hr</span></div>
            <div>Region: <span style="color:#fff;font-weight:600;">Permian Basin</span></div>
          </div>
        </div>
      </div>`
    return div
  }
  legend.addTo(mapInstance)
})

onUnmounted(() => {
  if (stopAnim) stopAnim()
  if (mapInstance) mapInstance.remove()
})
</script>

<style scoped>
.section {
  padding: 6rem 2rem;
  background: #0f172a;
}
.container { max-width: 1280px; margin: 0 auto; }

.section-header {
  text-align: center;
  max-width: 720px;
  margin: 0 auto 3rem;
}
.section-tag {
  display: inline-block;
  background: rgba(59,130,246,0.15);
  color: #60a5fa;
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 100px;
  padding: 0.3rem 1rem;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 1rem;
}
.section-header h2 {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #fff;
  margin-bottom: 1rem;
}
.section-header p { font-size: 1rem; color: #94a3b8; line-height: 1.75; }

.map-wrapper {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.1);
  box-shadow: 0 24px 64px rgba(0,0,0,0.4);
}
.map-container { height: 640px; width: 100%; }

@media (max-width: 1024px) { .map-container { height: 520px; } }
@media (max-width: 768px)  { .map-container { height: 420px; } }
@media (max-width: 480px)  { .map-container { height: 340px; } }

.map-loading {
  position: absolute; inset: 0;
  background: rgba(15,23,42,0.92);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 1rem; font-size: 0.95rem; color: #94a3b8;
}
.spinner {
  width: 32px; height: 32px;
  border: 3px solid rgba(255,255,255,0.1);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.metrics-bar {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px; margin-top: 1px;
  background: rgba(255,255,255,0.06);
  border-radius: 0 0 16px 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
  border-top: none;
}
.metric {
  display: flex; flex-direction: column;
  align-items: center; padding: 1.5rem 1rem;
  background: rgba(15,23,42,0.7);
  text-align: center; gap: 0.15rem;
  transition: background 0.2s;
}
.metric:hover { background: rgba(30,41,59,0.8); }
.metric-value { font-size: 1.9rem; font-weight: 800; color: #fff; line-height: 1; }
.metric-unit  { font-size: 0.72rem; color: #64748b; font-weight: 500; }
.metric-label { font-size: 0.72rem; font-weight: 500; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

@media (max-width: 768px) {
  .metrics-bar { grid-template-columns: repeat(2, 1fr); }
  .metric { padding: 1.1rem 0.75rem; }
  .metric-value { font-size: 1.5rem; }
  .section-header { margin-bottom: 2rem; }
}
@media (max-width: 480px) {
  .section { padding: 3rem 1rem !important; }
  .metric-value { font-size: 1.3rem; }
}
</style>

<style>
@media (max-width: 600px) {
  .leaflet-bottom.leaflet-right { display: none !important; }
}
</style>
