<template>
  <section id="what-we-do" class="section">
    <div class="container">
      <div class="section-header">
        <h2>What We Do</h2>
        <p>
          PHANTOM combines physics-informed AI with satellite hyperspectral remote sensing to detect,
          quantify, and deliver methane emission intelligence — from orbital detection to ground-level
          verification and actionable data services.
        </p>
      </div>

      <div class="cards-grid">
        <div v-for="(card, i) in cards" :key="card.title" class="card reveal" :class="`reveal-delay-${(i % 3) + 1}`">
          <div class="card-icon-wrap">
            <span class="card-icon">{{ card.icon }}</span>
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
          <h3>From Satellite to Action</h3>
          <p>A fully automated pipeline from raw hyperspectral data to verified, actionable emission intelligence</p>
        </div>

        <div class="pipeline-steps">
          <div v-for="(step, i) in pipeline" :key="step.label" class="pipeline-step">
            <div class="step-card">
              <div class="step-img-wrap">
                <img :src="step.img" :alt="step.label" class="step-img" />
                <div class="step-num-badge">{{ i + 1 }}</div>
              </div>
              <div class="step-body">
                <div class="step-icon">{{ step.icon }}</div>
                <div class="step-label">{{ step.label }}</div>
                <div class="step-desc">{{ step.desc }}</div>
              </div>
            </div>
            <div v-if="i < pipeline.length - 1" class="pipeline-arrow">
              <svg viewBox="0 0 40 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M0 8 H32 M26 2 L38 8 L26 14" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
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
    icon: '🛰️',
    title: 'Aerial & Satellite Detection',
    text: 'We process data from airborne and satellite hyperspectral platforms to detect the invisible spectral signature of methane gas — even in complex landscapes and terrains. Validated on NASA AVIRIS-NG flight campaigns across the Permian Basin and California.',
    tags: ['AVIRIS-NG', 'Hyperspectral', 'SWIR'],
  },
  {
    icon: '🧠',
    title: 'AI-Powered Analysis',
    text: 'PHANTOM\'s physics-informed transformer distinguishes real methane plumes from background artifacts and confounders with exceptional precision — achieving 100% F1 on critical super-emitters while keeping false positives at 11.43%.',
    tags: ['Physics-Informed', 'Transformer', 'Adversarial'],
  },
  {
    icon: '📊',
    title: 'Emission Quantification',
    text: 'Once a plume is detected, our system automatically quantifies its emission rate using the Integrated Methane Enhancement (IME) method combined with wind-field data. Results are reported in kg/hr — a clear, actionable number for mitigation planning.',
    tags: ['kg/hr', 'IME Method', 'Wind-Based'],
  },
  {
    icon: '📍',
    title: 'Local Quantification',
    text: 'High-resolution ground-level quantification using drone surveys and mobile sensors validates satellite detections and pinpoints exact source locations within industrial facilities. Enables facility-scale attribution and targeted leak repair.',
    tags: ['Drone Survey', 'Ground-Truth', 'Source Attribution'],
  },
  {
    icon: '🗄️',
    title: 'Data as a Service',
    text: 'Access PHANTOM\'s emission intelligence through structured data deliverables — including geo-referenced plume reports, API feeds, and interactive dashboards. Integrate methane monitoring directly into your operational workflows and ESG reporting pipelines.',
    tags: ['API Access', 'ESG Reporting', 'Dashboards'],
  },
  {
    icon: '🚨',
    title: 'Super-Emitter Prioritization',
    text: 'Automatically flag and rank the highest-impact emission sources by rate, frequency, and location. Focus mitigation resources where they matter most — targeting the 20% of sites responsible for 80% of total emissions to maximize climate impact per dollar spent.',
    tags: ['Risk Ranking', 'Mitigation ROI', 'Impact Prioritization'],
  },
]

const pipeline = [
  {
    icon: '🛰️',
    label: 'Satellite Imagery',
    desc: 'Hyperspectral data from airborne & orbital platforms over target regions',
    img: '/images/earth-from-space.jpg',
  },
  {
    icon: '🤖',
    label: 'PHANTOM AI',
    desc: 'Physics-informed transformer model detects methane spectral signatures',
    img: '/images/hyperspectral.jpg',
  },
  {
    icon: '📍',
    label: 'Detection & Quantification',
    desc: 'Plume boundaries mapped, emission rate computed in kg/hr',
    img: '/images/emission-sources.jpg',
  },
  {
    icon: '🚁',
    label: 'Local Verification',
    desc: 'Drone surveys provide ground-truth validation of flagged sites',
    img: '/images/drone.jpg',
  },
  {
    icon: '⚡',
    label: 'Actionable Decision',
    desc: 'Operators receive prioritized alerts and mitigation guidance',
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
  background: #eff6ff;
  color: #2563eb;
  border: 1px solid #bfdbfe;
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
  width: 52px;
  height: 52px;
  background: #f0f9ff;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.card-icon { font-size: 1.6rem; }

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
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
  border-radius: 24px;
  padding: 3.5rem 3rem;
}

.pipeline-header {
  text-align: center;
  margin-bottom: 3rem;
}

.pipeline-header .section-tag {
  background: rgba(59,130,246,0.15);
  color: #60a5fa;
  border-color: rgba(59,130,246,0.3);
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
  align-items: center;
  justify-content: center;
  gap: 0;
  flex-wrap: nowrap;
  overflow-x: auto;
  padding-bottom: 0.5rem;
}

.pipeline-step {
  display: flex;
  align-items: center;
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
  background: #3b82f6;
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

.step-icon {
  font-size: 1.2rem;
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
}

.pipeline-arrow svg {
  width: 40px;
  height: 16px;
}

@media (max-width: 900px) {
  .pipeline-steps { flex-wrap: wrap; justify-content: center; gap: 1rem; }
  .pipeline-arrow { display: none; }
  .step-card { width: 145px; }
}

@media (max-width: 768px) {
  .pipeline { padding: 2.5rem 1.25rem; border-radius: 16px; }
  .pipeline-header { margin-bottom: 2rem; }
  .pipeline-steps { gap: 0.75rem; }
  .step-card { width: 140px; }
  .section-header { margin-bottom: 2.5rem; }
}

@media (max-width: 480px) {
  .pipeline { padding: 2rem 1rem; }
  .cards-grid { gap: 1rem; }
  .card { padding: 1.5rem; }
  .step-card { width: 130px; }
}
</style>
