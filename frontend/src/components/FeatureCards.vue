<template>
  <section id="what-we-do" class="section">
    <div class="container">
      <div class="section-header">
        <h2>What We Do</h2>
        <p>
          PHANTOM processes airborne hyperspectral imagery to detect methane plumes, estimate
          emission rates, and attribute sources — covering the full workflow from raw sensor
          data to structured detection reports.
        </p>
      </div>

      <div class="cards-grid">
        <div v-for="(card, i) in cards" :key="card.title" class="card reveal" :class="`reveal-delay-${(i % 3) + 1}`">
          <div class="card-icon-wrap" :style="{ background: card.color + '12', borderColor: card.color + '30' }">
            <span class="card-abbr" :style="{ color: card.color }">{{ card.abbr }}</span>
          </div>
          <h3 class="card-title">{{ card.title }}</h3>
          <p class="card-text">{{ card.text }}</p>
          <div class="card-footer">
            <span v-for="tag in card.tags" :key="tag" class="tag">{{ tag }}</span>
          </div>
        </div>
      </div>

      <!-- Professional Pipeline -->
      <div class="pipeline">
        <div class="pipeline-header">
          <span class="section-tag">End-to-End Workflow</span>
          <h3>From Sensor to Report</h3>
          <p>The end-to-end workflow from raw hyperspectral radiance data to geo-referenced emission reports</p>
        </div>

        <div class="pipeline-steps">
          <div v-for="(step, i) in pipeline" :key="step.label" class="pipeline-step">
            <div class="step-card">
              <div class="step-img-wrap">
                <img :src="step.img" :alt="step.label" class="step-img" />
                <div class="step-num-badge">{{ i + 1 }}</div>
              </div>
              <div class="step-body">
                <div class="step-num">{{ step.num }}</div>
                <div class="step-label">{{ step.label }}</div>
                <div class="step-desc">{{ step.desc }}</div>
              </div>
            </div>
            <div v-if="i < pipeline.length - 1" class="pipeline-arrow">
              <svg viewBox="0 0 40 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M0 8 H32 M26 2 L38 8 L26 14" stroke="#4ade80" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const cards = [
  {
    abbr: 'HS',
    color: '#2563eb',
    title: 'Multi-Platform Hyperspectral Imaging',
    text: 'Processes SWIR radiance data from satellite, airborne, and UAV hyperspectral sensors to identify the CH₄ absorption signature. Validated on NASA AVIRIS-NG campaigns over the Permian Basin.',
    tags: ['Satellite', 'Airborne', 'UAV', 'SWIR'],
  },
  {
    abbr: 'ML',
    color: '#7c3aed',
    title: 'Physics-Informed Detection',
    text: 'A transformer architecture with adversarial training separates methane plumes from surface reflectance artifacts and atmospheric confounders. Achieves 100% F1 on the critical emitter class.',
    tags: ['Physics-Informed', 'Transformer', 'Adversarial Training'],
  },
  {
    abbr: 'ER',
    color: '#0891b2',
    title: 'Emission Rate Estimation',
    text: 'Plume-integrated emission rates are computed using the Integrated Methane Enhancement (IME) method with coincident wind measurements. Output is reported in kg/hr per source.',
    tags: ['IME Method', 'kg/hr', 'Wind-Coupled'],
  },
  {
    abbr: 'SA',
    color: '#059669',
    title: 'Source Attribution',
    text: 'Detected plumes are geo-referenced and linked to facility-level infrastructure using UTM-projected spatial data, supporting site-specific attribution and reporting.',
    tags: ['Geo-Referenced', 'UTM Projection', 'Facility-Level'],
  },
  {
    abbr: 'DD',
    color: '#d97706',
    title: 'Structured Data Deliverables',
    text: 'Detection results are packaged as geo-referenced reports including plume boundaries, emission rates, confidence scores, and wind parameters — ready for integration into existing workflows.',
    tags: ['GeoJSON', 'Detection Reports', 'Plume Metadata'],
  },
  {
    abbr: 'SE',
    color: '#dc2626',
    title: 'Super-Emitter Identification',
    text: 'Sites are ranked by emission rate to help operators prioritize the highest-impact sources. In the Permian Basin dataset, 130 of 178 detected sites exceeded the critical emitter threshold.',
    tags: ['Ranked Alerts', 'Critical Threshold', 'Permian Basin'],
  },
]

const pipeline = [
  {
    num: '01',
    label: 'Hyperspectral Acquisition',
    desc: 'SWIR radiance data collected via satellite or aircraft over the target region',
    img: '/images/earth-from-space.jpg',
  },
  {
    num: '02',
    label: 'PHANTOM Model',
    desc: 'Physics-informed transformer detects CH₄ spectral signatures',
    img: '/images/hyperspectral.jpg',
  },
  {
    num: '03',
    label: 'Plume Delineation',
    desc: 'Plume boundaries mapped and emission rate estimated in kg/hr',
    img: '/images/emission-sources.jpg',
  },
  {
    num: '04',
    label: 'Source Attribution',
    desc: 'Site-level attribution with emission rates quantified per source',
    img: '/images/drone.jpg',
  },
  {
    num: '05',
    label: 'Detection Report',
    desc: 'Structured outputs delivered with rates, confidence, and location data',
    img: '/images/ground-sensors.jpg',
  },
]
</script>

<style scoped>
.section {
  padding: 6rem 2rem;
  background: #fff;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.section-tag {
  display: inline-block;
  background: #f0fdf4;
  color: #16a34a;
  border: 1px solid #bbf7d0;
  border-radius: 100px;
  padding: 0.3rem 1rem;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 1rem;
}

.section-header {
  text-align: center;
  max-width: 700px;
  margin: 0 auto 4rem;
}

.section-header h2 {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #0f172a;
  margin-bottom: 1rem;
}

.section-header p {
  font-size: 1.1rem;
  color: #64748b;
  line-height: 1.8;
}

.cards-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  margin-bottom: 5rem;
}

@media (max-width: 960px) {
  .cards-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 600px) {
  .cards-grid { grid-template-columns: 1fr; }
}

.card {
  background: #fff;
  border-radius: 16px;
  padding: 2rem;
  border: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
}

.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 20px 48px rgba(0,0,0,0.08);
  border-color: #bfdbfe;
}

.card-icon-wrap {
  width: 48px;
  height: 48px;
  border-radius: 10px;
  border: 1px solid;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.card-abbr {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.04em;
}

.card-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #0f172a;
}

.card-text {
  font-size: 0.95rem;
  color: #64748b;
  line-height: 1.8;
  flex: 1;
}

.card-footer {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  padding-top: 0.5rem;
  border-top: 1px solid #f1f5f9;
}

.tag {
  background: #f8fafc;
  color: #475569;
  border: 1px solid #e2e8f0;
  border-radius: 100px;
  padding: 0.2rem 0.7rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}

/* Pipeline */
.pipeline {
  background: linear-gradient(135deg, #0d1f12 0%, #172b1c 100%);
  border-radius: 24px;
  padding: 3.5rem 3rem;
}

.pipeline-header {
  text-align: center;
  margin-bottom: 3rem;
}

.pipeline-header .section-tag {
  background: rgba(22,163,74,0.15);
  color: #4ade80;
  border-color: rgba(22,163,74,0.35);
}

.pipeline-header h3 {
  font-size: clamp(1.5rem, 3vw, 2rem);
  font-weight: 800;
  color: #fff;
  margin-bottom: 0.6rem;
  letter-spacing: -0.02em;
}

.pipeline-header p {
  font-size: 0.95rem;
  color: #94a3b8;
  max-width: 540px;
  margin: 0 auto;
  line-height: 1.7;
}

.pipeline-steps {
  display: flex;
  align-items: stretch;
  justify-content: center;
  gap: 0;
  flex-wrap: nowrap;
  overflow-x: auto;
  padding-bottom: 0.5rem;
}

.pipeline-step {
  display: flex;
  align-items: stretch;
  flex-shrink: 0;
}

.step-card {
  width: 180px;
  border-radius: 16px;
  overflow: hidden;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  transition: transform 0.25s, background 0.25s;
  cursor: default;
  display: flex;
  flex-direction: column;
}

.step-card:hover {
  transform: translateY(-4px);
  background: rgba(255,255,255,0.09);
}

.step-img-wrap {
  position: relative;
  height: 130px;
  overflow: hidden;
}

.step-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  filter: brightness(0.65) saturate(0.8);
  transition: filter 0.3s;
}

.step-card:hover .step-img {
  filter: brightness(0.8) saturate(1);
}

.step-num-badge {
  position: absolute;
  top: 8px;
  left: 8px;
  width: 24px;
  height: 24px;
  background: #16a34a;
  color: #fff;
  border-radius: 50%;
  font-size: 0.7rem;
  font-weight: 800;
  display: flex;
  align-items: center;
  justify-content: center;
}

.step-body {
  padding: 0.9rem;
}

.step-num {
  font-size: 0.68rem;
  font-weight: 800;
  color: #4ade80;
  letter-spacing: 0.1em;
  margin-bottom: 0.3rem;
}

.step-label {
  font-size: 0.82rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 0.3rem;
  line-height: 1.2;
}

.step-desc {
  font-size: 0.72rem;
  color: #94a3b8;
  line-height: 1.5;
}

.pipeline-arrow {
  width: 40px;
  flex-shrink: 0;
  padding: 0 2px;
  margin-top: -20px;
  align-self: center;
}

.pipeline-arrow svg {
  width: 40px;
  height: 16px;
}

@media (max-width: 900px) {
  .pipeline-steps { flex-direction: column; align-items: center; gap: 0; }
  .pipeline-step { flex-direction: column; align-items: center; }
  .pipeline-arrow { width: 24px; height: 32px; display: flex; align-items: center; justify-content: center; transform: rotate(90deg); margin: 2px 0; }
  .pipeline-arrow svg { width: 32px; height: 16px; }
  .step-card { width: 280px; }
}

@media (max-width: 768px) {
  .pipeline { padding: 2.5rem 1.25rem; border-radius: 16px; }
  .pipeline-header { margin-bottom: 2rem; }
  .section-header { margin-bottom: 2.5rem; }
  .step-card { width: 260px; }
}

@media (max-width: 480px) {
  .pipeline { padding: 2rem 1rem; }
  .cards-grid { gap: 1rem; }
  .card { padding: 1.5rem; }
  .step-card { width: 100%; max-width: 320px; }
}
</style>
