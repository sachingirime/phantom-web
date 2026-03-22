<template>
  <section id="detection-map" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Validated Detection Dataset · Permian Basin</span>
        <h2>Emission Detection Map</h2>
        <p>
          Methane plumes detected and quantified by PHANTOM across the Permian Basin, Texas.
          Plume boundaries animated by real wind speed and inferred direction.
          Data sourced from NASA AVIRIS-NG hyperspectral flight campaigns.
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

// ── Plasma/Inferno colormap — vivid dark-purple → magenta → orange → yellow ──
function thermalRGB(t) {
  t = Math.max(0, Math.min(1, t))
  // Key stops: deep-indigo → violet → hot-pink → orange → bright-yellow
  const stops = [
    [13,   8,  135],   // t=0.00  deep indigo
    [84,   2,  163],   // t=0.20  purple
    [174,  42, 121],   // t=0.40  hot pink / magenta
    [238, 107,  34],   // t=0.60  orange
    [253, 210,  36],   // t=0.80  yellow
    [240, 249,  33],   // t=1.00  bright yellow-white
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

// ── Animate plumes inside real polygon boundaries ─────────────────────────────
// Strategy:
//   • Wind direction  = vector from emission source → polygon centroid (geometry-derived)
//   • Wind speed      = U_10_ms from dataset
//   • Color intensity = mean_conf mapped through thermal colormap
//   • 3 gradient "puffs" cycle along the wind axis, clipped to polygon shape
//   • Radial turbulence blobs add lateral dispersion inside the boundary

function animatePlumes(canvas, map, emissions, confMin, confMax) {
  const ctx = canvas.getContext('2d')
  const startTime = performance.now()
  let raf

  function frame() {
    const t = (performance.now() - startTime) / 1000
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0, 0, W, H)

    emissions.forEach(row => {
      if (!row.plume || !row.plume.polygons.length) return

      const src = map.latLngToContainerPoint(L.latLng(row.latitude, row.longitude))

      // Skip if source point is far off-screen
      if (src.x < -200 || src.x > W + 200 || src.y < -200 || src.y > H + 200) return

      const ct = (row.plume.mean_conf - confMin) / (confMax - confMin + 1e-6)
      const [cr, cg, cb] = thermalRGB(ct)
      const strength = Math.min(row.Q_kg_hr / 2500, 1)
      const windPx   = (row.U_10_ms || 3) * 5

      // Project ALL polygon points to screen space (needed for outlines + clip)
      const allPtSets = row.plume.polygons
        .map(polygon => polygon.map(([lat, lon]) => {
          const p = map.latLngToContainerPoint(L.latLng(lat, lon))
          return { x: p.x, y: p.y }
        }))
        .filter(pts => pts.length >= 3)
      if (!allPtSets.length) return

      // Find the largest polygon to derive wind direction and clip region
      let bestPts = null, bestArea = 0
      allPtSets.forEach(pts => {
        const bx0 = Math.min(...pts.map(p => p.x)), bx1 = Math.max(...pts.map(p => p.x))
        const by0 = Math.min(...pts.map(p => p.y)), by1 = Math.max(...pts.map(p => p.y))
        const area = (bx1 - bx0) * (by1 - by0)
        if (area > bestArea) { bestArea = area; bestPts = pts }
      })
      if (!bestPts) return

      // Wind direction: source → polygon centroid (works at any zoom)
      const cx = bestPts.reduce((s, p) => s + p.x, 0) / bestPts.length
      const cy = bestPts.reduce((s, p) => s + p.y, 0) / bestPts.length
      const dx = cx - src.x, dy = cy - src.y
      const pixDist = Math.sqrt(dx * dx + dy * dy)

      // Normalised wind direction (fallback: upward if polygon is co-located)
      const nx = pixDist > 0.5 ? dx / pixDist : 0
      const ny = pixDist > 0.5 ? dy / pixDist : -1
      const px = -ny, py = nx   // perpendicular

      // ── When polygon is too small on-screen (zoomed out): small glow only ──
      const polyDiag = Math.sqrt(bestArea)
      if (polyDiag < 8) {
        const r = 4 + strength * 7
        ctx.save()
        ctx.globalCompositeOperation = 'source-over'
        const sg0 = ctx.createRadialGradient(src.x, src.y, 0, src.x, src.y, r)
        sg0.addColorStop(0,   `rgba(${cr},${cg},${cb},0.9)`)
        sg0.addColorStop(0.5, `rgba(${cr},${cg},${cb},0.3)`)
        sg0.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`)
        ctx.fillStyle = sg0
        ctx.beginPath(); ctx.arc(src.x, src.y, r, 0, Math.PI * 2); ctx.fill()
        ctx.restore()
        return
      }

      // ── SMOKE SIMULATION ──────────────────────────────────────────────────
      // Clip to UNION of ALL polygons so smoke never bleeds outside boundaries.
      ctx.save()
      ctx.globalCompositeOperation = 'source-over'
      ctx.beginPath()
      allPtSets.forEach(pts => {
        ctx.moveTo(pts[0].x, pts[0].y)
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y)
        ctx.closePath()
      })
      ctx.clip()  // union clip — all sub-polygons in one path

      // Puff travel length = polygon diagonal. Cycle speed from real wind speed.
      const polyLen  = Math.max(polyDiag * 0.95, 30)
      const cycleSec = Math.max(polyLen / (windPx + 0.1), 0.6)

      // ── Layer 0: Subtle base tint fills the entire polygon interior ────────
      // fillRect is clipped to the union polygon — no bleeding.
      ctx.fillStyle = `rgba(${cr},${cg},${cb},${0.06 + strength * 0.05})`
      ctx.fillRect(0, 0, W, H)

      // ── Layer 1: Slow background volume wisps ──────────────────────────────
      const N_WISPS = 10
      for (let k = 0; k < N_WISPS; k++) {
        const kP = ((t / (cycleSec * 2.5)) + k / N_WISPS) % 1
        const lat = polyLen * 0.22 * Math.sin(t * 0.38 + k * 2.09)
        const bx  = src.x + nx * polyLen * kP + px * lat
        const by  = src.y + ny * polyLen * kP + py * lat
        const r   = (16 + strength * 26) * (0.5 + kP * 1.2)
        const a   = Math.sin(kP * Math.PI) * (0.13 + strength * 0.10)
        if (a < 0.005) continue
        const rg = ctx.createRadialGradient(bx, by, 0, bx, by, r)
        rg.addColorStop(0,    `rgba(${cr},${cg},${cb},${a})`)
        rg.addColorStop(0.45, `rgba(${cr},${cg},${cb},${a * 0.55})`)
        rg.addColorStop(1,    `rgba(${cr},${cg},${cb},0)`)
        ctx.fillStyle = rg
        ctx.beginPath(); ctx.arc(bx, by, r, 0, Math.PI * 2); ctx.fill()
      }

      // ── Layer 2: Main smoke puffs — dense two-frequency billowing stream ───
      const N_PUFFS = 40
      for (let k = 0; k < N_PUFFS; k++) {
        const kP = ((t / cycleSec) + k / N_PUFFS) % 1
        // Two-frequency lateral: golden-ratio offsets give non-repeating motion
        const latAmp = polyLen * 0.18
        const lat = latAmp * (
          0.58 * Math.sin(t * 0.72 + k * 1.618 + kP * Math.PI) +
          0.42 * Math.sin(t * 1.48 + k * 2.937 + kP * Math.PI * 1.8)
        )
        const bx = src.x + nx * polyLen * kP + px * lat
        const by = src.y + ny * polyLen * kP + py * lat

        const baseR = 5 + strength * 11
        const r     = baseR * (0.30 + kP * 2.0)   // expands as it travels

        // Quick fade-in (0–8%), full plateau, gradual fade-out (60–100%)
        const fadeIn  = Math.min(kP / 0.08, 1)
        const fadeOut = kP > 0.60 ? Math.max(0, 1 - (kP - 0.60) / 0.40) : 1
        const a       = fadeIn * fadeOut * (0.30 + strength * 0.18)
        if (a < 0.008) continue

        const rg = ctx.createRadialGradient(bx, by, 0, bx, by, r)
        rg.addColorStop(0,    `rgba(${cr},${cg},${cb},${a})`)
        rg.addColorStop(0.35, `rgba(${cr},${cg},${cb},${a * 0.75})`)
        rg.addColorStop(0.70, `rgba(${cr},${cg},${cb},${a * 0.30})`)
        rg.addColorStop(1,    `rgba(${cr},${cg},${cb},0)`)
        ctx.fillStyle = rg
        ctx.beginPath(); ctx.arc(bx, by, r, 0, Math.PI * 2); ctx.fill()
      }

      // ── Layer 3: Dense core puffs near source — tight, high alpha ─────────
      const N_CORE = 14
      for (let k = 0; k < N_CORE; k++) {
        const kP = ((t / (cycleSec * 0.5)) + k / N_CORE) % 1
        const travelFrac = kP * 0.35           // only covers first 35% of polygon
        const lat = polyLen * 0.10 * Math.sin(t * 1.1 + k * 1.618)
        const bx  = src.x + nx * polyLen * travelFrac + px * lat
        const by  = src.y + ny * polyLen * travelFrac + py * lat
        const r   = (3 + strength * 7) * (0.5 + kP * 0.8)
        const a   = Math.sin(kP * Math.PI) * (0.40 + strength * 0.22)
        if (a < 0.01) continue
        const rg = ctx.createRadialGradient(bx, by, 0, bx, by, r)
        rg.addColorStop(0,   `rgba(${cr},${cg},${cb},${a})`)
        rg.addColorStop(0.5, `rgba(${cr},${cg},${cb},${a * 0.6})`)
        rg.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`)
        ctx.fillStyle = rg
        ctx.beginPath(); ctx.arc(bx, by, r, 0, Math.PI * 2); ctx.fill()
      }

      // ── Layer 4: Pulsing hot source glow ──────────────────────────────────
      const pulse = 0.70 + 0.30 * Math.sin(t * 2.6 + row.latitude * 10)
      const glowR  = 7 + strength * 12
      const sg = ctx.createRadialGradient(src.x, src.y, 0, src.x, src.y, glowR)
      sg.addColorStop(0,   `rgba(255,255,210,${0.70 * pulse})`)
      sg.addColorStop(0.3, `rgba(${cr},${cg},${cb},${0.55 * pulse})`)
      sg.addColorStop(0.7, `rgba(${cr},${cg},${cb},${0.20 * pulse})`)
      sg.addColorStop(1,   `rgba(${cr},${cg},${cb},0)`)
      ctx.fillStyle = sg
      ctx.beginPath(); ctx.arc(src.x, src.y, glowR, 0, Math.PI * 2); ctx.fill()

      ctx.restore()  // removes union clip — no outlines drawn, smoke IS the boundary
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

  // Confidence range for thermal normalization
  const confs   = emissions.filter(r => r.plume).map(r => r.plume.mean_conf)
  const confMin = Math.min(...confs)
  const confMax = Math.max(...confs)

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

  // ── Canvas directly inside Leaflet container at z-index 450 ──────────────────
  // z-index 450 sits above the leaflet-map-pane stacking context (z=400).
  // Being inside mapEl (which Leaflet doesn't CSS-transform) keeps it stable
  // during pan/zoom. latLngToContainerPoint() is measured from mapEl, so
  // coordinates map exactly onto this canvas.
  const canvas = document.createElement('canvas')
  // z-index must be > 400 (.leaflet-pane z-index) so the canvas renders
  // above the entire leaflet-map-pane stacking context, not behind it.
  canvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:450;'
  mapEl.value.appendChild(canvas)

  function resizeCanvas() {
    canvas.width  = mapEl.value.offsetWidth
    canvas.height = mapEl.value.offsetHeight
  }
  resizeCanvas()
  new ResizeObserver(resizeCanvas).observe(mapEl.value)

  // Start animation (re-projects polygons every frame — handles pan/zoom correctly)
  stopAnim = animatePlumes(canvas, mapInstance, emissions, confMin, confMax)

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
      <div style="font-family:Inter,sans-serif;width:300px;border-radius:12px;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.15);">
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
      <div style="background:rgba(15,23,42,0.88);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:14px 16px;font-family:Inter,sans-serif;font-size:0.75rem;color:#fff;min-width:200px;">
        <div style="font-weight:700;margin-bottom:10px;letter-spacing:0.06em;text-transform:uppercase;color:#94a3b8;">Plume Intensity</div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
          <div style="flex:1;height:10px;border-radius:4px;background:linear-gradient(to right,rgb(0,0,255),rgb(0,255,255),rgb(0,255,0),rgb(255,255,0),rgb(255,0,0));"></div>
        </div>
        <div style="display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:14px;font-size:0.7rem;">
          <span>Low (${(confMin*100).toFixed(1)}%)</span><span>High (${(confMax*100).toFixed(1)}%)</span>
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
