<template>
  <section id="detection-map" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Validated Detection Dataset · Permian Basin</span>
        <h2>Emission Detection Map</h2>
        <p>
          Methane plumes detected and quantified by PHANTOM across the Permian Basin, Texas.
          Boundaries colored by AI detection confidence. Data sourced from NASA AVIRIS-NG hyperspectral flight campaigns.
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
import { ref, onMounted } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const mapEl = ref(null)
const summary = ref(null)
const loading = ref(true)

// Thermal colormap: blue → cyan → green → yellow → red (like GHGSat)
function thermalColor(t) {
  // t is 0–1
  t = Math.max(0, Math.min(1, t))
  let r, g, b
  if (t < 0.25) {
    r = 0; g = Math.round(t * 4 * 255); b = 255
  } else if (t < 0.5) {
    r = 0; g = 255; b = Math.round((1 - (t - 0.25) * 4) * 255)
  } else if (t < 0.75) {
    r = Math.round((t - 0.5) * 4 * 255); g = 255; b = 0
  } else {
    r = 255; g = Math.round((1 - (t - 0.75) * 4) * 255); b = 0
  }
  return `rgb(${r},${g},${b})`
}

function normalize(val, min, max) {
  if (max === min) return 0.5
  return (val - min) / (max - min)
}

onMounted(async () => {
  const res = await fetch('/data/emissions.json')
  const data = await res.json()
  summary.value = data.summary
  loading.value = false

  const emissions = data.emissions

  // Compute mean_conf range across all plumes for normalization
  const confs = emissions.filter(r => r.plume).map(r => r.plume.mean_conf)
  const confMin = Math.min(...confs)
  const confMax = Math.max(...confs)

  const map = L.map(mapEl.value, {
    center: [32.33, -101.81],
    zoom: 12,
    scrollWheelZoom: true,
  })

  const satellite = L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: '© Esri, Maxar, Earthstar Geographics', maxZoom: 20 }
  )
  const streets = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap © CARTO',
    subdomains: 'abcd',
    maxZoom: 20,
  })

  satellite.addTo(map)
  L.control.layers({ 'Satellite': satellite, 'Street Map': streets }, {}, { position: 'topright' }).addTo(map)

  // Draw plumes then markers
  emissions.forEach(row => {
    const isCritical = row.emission_category === 'Critical'
    const markerColor = isCritical ? '#ef4444' : '#10b981'
    const size = isCritical ? 22 : 14
    const glowColor = isCritical ? 'rgba(239,68,68,0.35)' : 'rgba(16,185,129,0.35)'

    // Build popup HTML (shared by marker and plume polygons)
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

    // Draw plume polygons colored by thermal confidence
    if (row.plume && row.plume.polygons.length) {
      const t = normalize(row.plume.mean_conf, confMin, confMax)
      const fillColor = thermalColor(t)

      row.plume.polygons.forEach(polygon => {
        L.polygon(polygon, {
          color: fillColor,
          weight: 1.5,
          opacity: 1,
          fill: true,
          fillColor: fillColor,
          fillOpacity: 0.55,
        })
          .bindPopup(popup, { maxWidth: 320 })
          .addTo(map)
      })
    }

    // Marker on top
    const icon = L.divIcon({
      html: `<div style="position:relative;width:${size}px;height:${size}px;">
        <div style="position:absolute;inset:-5px;background:${glowColor};border-radius:50%;"></div>
        <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
          <circle cx="${size/2}" cy="${size/2}" r="${size/2-1}" fill="${markerColor}" opacity="0.95" stroke="white" stroke-width="2"/>
        </svg>
      </div>`,
      className: '',
      iconSize: [size, size],
      iconAnchor: [size/2, size/2],
    })

    L.marker([row.latitude, row.longitude], { icon })
      .bindPopup(popup, { maxWidth: 320 })
      .bindTooltip(`<b>${row.Q_kg_hr.toFixed(1)} kg/hr</b> · ${row.emission_category}`)
      .addTo(map)
  })

  // Colorbar legend
  const legend = L.control({ position: 'bottomright' })
  legend.onAdd = () => {
    const div = L.DomUtil.create('div')
    div.innerHTML = `
      <div style="background:rgba(15,23,42,0.88);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:14px 16px;font-family:Inter,sans-serif;font-size:0.75rem;color:#fff;min-width:200px;">
        <div style="font-weight:700;margin-bottom:10px;letter-spacing:0.06em;text-transform:uppercase;color:#94a3b8;">Plume Intensity</div>

        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
          <div style="flex:1;height:10px;border-radius:4px;background:linear-gradient(to right,rgb(0,0,255),rgb(0,255,255),rgb(0,255,0),rgb(255,255,0),rgb(255,0,0));"></div>
        </div>
        <div style="display:flex;justify-content:space-between;color:#94a3b8;margin-bottom:14px;font-size:0.7rem;">
          <span>Low (${(confMin*100).toFixed(1)}%)</span>
          <span>High (${(confMax*100).toFixed(1)}%)</span>
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
      </div>
    `
    return div
  }
  legend.addTo(map)
})
</script>

<style scoped>
.section {
  padding: 6rem 2rem;
  background: #0f172a;
}

.container {
  max-width: 1280px;
  margin: 0 auto;
}

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

.section-header p {
  font-size: 1rem;
  color: #94a3b8;
  line-height: 1.75;
}

.map-wrapper {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.1);
  box-shadow: 0 24px 64px rgba(0,0,0,0.4);
}

.map-container {
  height: 640px;
  width: 100%;
}

@media (max-width: 1024px) { .map-container { height: 520px; } }
@media (max-width: 768px)  { .map-container { height: 420px; } }
@media (max-width: 480px)  { .map-container { height: 340px; } }

.map-loading {
  position: absolute;
  inset: 0;
  background: rgba(15,23,42,0.92);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  font-size: 0.95rem;
  color: #94a3b8;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(255,255,255,0.1);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.metrics-bar {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  margin-top: 1px;
  background: rgba(255,255,255,0.06);
  border-radius: 0 0 16px 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.08);
  border-top: none;
}

.metric {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem 1rem;
  background: rgba(15,23,42,0.7);
  text-align: center;
  gap: 0.15rem;
  transition: background 0.2s;
}
.metric:hover { background: rgba(30,41,59,0.8); }

.metric-value {
  font-size: 1.9rem;
  font-weight: 800;
  color: #fff;
  line-height: 1;
}
.metric-unit {
  font-size: 0.72rem;
  color: #64748b;
  font-weight: 500;
}
.metric-label {
  font-size: 0.72rem;
  font-weight: 500;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

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
