<template>
  <section id="performance" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Validation</span>
        <h2>Platform Performance</h2>
        <p>Model performance evaluated on NASA AVIRIS-NG airborne hyperspectral flight campaigns</p>
      </div>

      <div class="benchmark-block">
        <div class="benchmark-inner">
          <div class="benchmark-left">
            <h3>Benchmark Results</h3>
            <p>Evaluated on AVIRIS-NG California and Permian Basin flight campaigns across diverse terrain and atmospheric conditions. F1 score is 4× higher than the matched-filter baseline on the standard emitter class.</p>
          </div>
          <div class="stats-grid">
            <div v-for="stat in stats" :key="stat.label" class="stat-item">
              <span class="stat-value">{{ stat.value }}</span>
              <span class="stat-label">{{ stat.label }}</span>
              <div class="stat-bar-track">
                <div class="stat-bar-fill" :style="{ width: stat.bar, background: stat.barColor }"></div>
              </div>
            </div>
          </div>
        </div>
        <div class="benchmark-note">
          NASA AVIRIS-NG airborne hyperspectral imagery · Permian Basin and California flight campaigns · Diverse terrain and environmental conditions
        </div>
      </div>

      <div class="tech-block">
        <div class="tech-content">
          <h4>Technical Approach</h4>
          <p>
            PHANTOM uses a deep learning architecture that integrates physical principles of
            atmospheric radiative transfer with data-driven detection. The model is designed to
            reliably identify methane plumes under diverse terrain and atmospheric conditions
            where conventional methods produce high false positive rates.
          </p>
          <div class="tech-pills">
            <span v-for="tech in technologies" :key="tech" class="pill">{{ tech }}</span>
          </div>
        </div>
        <div class="tech-img-wrap">
          <img src="/images/aviris-ng.jpg" alt="NASA AVIRIS-NG hyperspectral sensor" />
        </div>
      </div>

      <div class="fusion-block">
        <div class="fusion-header">
          <span class="section-tag">Research Extension</span>
          <h3>Multi-Sensor Fusion</h3>
          <p>
            PHANTOM supports fusion of hyperspectral data with complementary imaging modalities.
            Integrating multiple sensor streams produces meaningful gains in both detection
            precision and false positive reduction.
          </p>
        </div>
        <div class="fusion-compare">
          <div class="compare-col">
            <div class="compare-label">Single Modality</div>
            <div class="compare-stats">
              <div class="compare-stat">
                <span class="compare-val">62.42%</span>
                <span class="compare-key">AUPRC</span>
                <div class="bar-track"><div class="bar-fill" style="width: 62.42%"></div></div>
              </div>
              <div class="compare-stat">
                <span class="compare-val">11.43%</span>
                <span class="compare-key">False Positive Rate</span>
                <div class="bar-track"><div class="bar-fill bar-fpr" style="width: 11.43%"></div></div>
              </div>
            </div>
          </div>
          <div class="fusion-arrow">
            <svg viewBox="0 0 40 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M0 12 H32 M26 4 L38 12 L26 20" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>+ Satellite</span>
          </div>
          <div class="compare-col compare-col-highlight">
            <div class="compare-label">Multi-Sensor Fusion</div>
            <div class="compare-stats">
              <div class="compare-stat">
                <span class="compare-val highlight-val">75%</span>
                <span class="compare-key">AUPRC</span>
                <div class="bar-track"><div class="bar-fill bar-fusion" style="width: 75%"></div></div>
              </div>
              <div class="compare-stat">
                <span class="compare-val highlight-val">~4%</span>
                <span class="compare-key">False Positive Rate</span>
                <div class="bar-track"><div class="bar-fill bar-fpr-fusion" style="width: 4%"></div></div>
              </div>
            </div>
          </div>
        </div>
        <p class="fusion-note">
          Multi-modal sensor fusion. MESA Lab, UC Merced.
        </p>
      </div>
    </div>
  </section>
</template>

<script setup>
const stats = [
  { value: '100%', label: 'F1 Score (Critical)', bar: '100%', barColor: '#34d399' },
  { value: '96.65%', label: 'F1 Score (Standard)', bar: '96.65%', barColor: '#60a5fa' },
  { value: '11.43%', label: 'False Positive Rate', bar: '11.43%', barColor: '#f97316' },
  { value: '62.42%', label: 'AUPRC', bar: '62.42%', barColor: '#a78bfa' },
]

const technologies = [
  'Physics-Informed NNs',
  'Transformer Architecture',
  'SWIR Hyperspectral',
  'Matched Filter',
  'Adversarial Training',
  'NASA AVIRIS-NG',
  'PyTorch',
  'UTM Geo-referencing',
]
</script>

<style scoped>
.section {
  padding: 6rem 2rem;
  background: #f8fafc;
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
  max-width: 600px;
  margin: 0 auto 4rem;
}

.section-header h2 {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #0f172a;
  margin-bottom: 0.75rem;
}

.section-header p {
  font-size: 1rem;
  color: #64748b;
  line-height: 1.7;
}

.benchmark-block {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
  border-radius: 20px;
  padding: 3rem;
  color: #fff;
  margin-bottom: 2rem;
}

.benchmark-inner {
  display: flex;
  gap: 3rem;
  align-items: flex-start;
  margin-bottom: 2rem;
  flex-wrap: wrap;
}

.benchmark-left {
  flex: 1;
  min-width: 220px;
}

.benchmark-left h3 {
  font-size: 1.5rem;
  font-weight: 800;
  margin-bottom: 0.75rem;
  letter-spacing: -0.02em;
}

.benchmark-left p {
  font-size: 0.9rem;
  color: rgba(148,163,184,0.85);
  line-height: 1.75;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.25rem;
  flex: 2;
}

.stat-item {
  display: flex;
  flex-direction: column;
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 1.25rem;
}

.stat-value {
  font-size: 2rem;
  font-weight: 800;
  color: #60a5fa;
  line-height: 1;
  margin-bottom: 0.4rem;
}

.stat-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: rgba(148,163,184,0.7);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 0.5rem;
}

.stat-bar-track {
  height: 3px;
  background: rgba(255,255,255,0.1);
  border-radius: 100px;
  overflow: hidden;
}

.stat-bar-fill {
  height: 100%;
  border-radius: 100px;
  opacity: 0.85;
}

.benchmark-note {
  font-size: 0.875rem;
  color: rgba(148,163,184,0.7);
  line-height: 1.7;
  border-top: 1px solid rgba(255,255,255,0.08);
  padding-top: 1.5rem;
}

.tech-block {
  background: #fff;
  border-radius: 20px;
  border: 1px solid #e2e8f0;
  overflow: hidden;
  display: grid;
  grid-template-columns: 1fr 1fr;
}

.tech-content {
  padding: 2.5rem;
}

.tech-content h4 {
  font-size: 0.78rem;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 1rem;
}

.tech-content p {
  font-size: 0.9rem;
  color: #64748b;
  line-height: 1.85;
  margin-bottom: 1.75rem;
}

.tech-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.pill {
  background: #f1f5f9;
  color: #334155;
  border: 1px solid #e2e8f0;
  border-radius: 100px;
  padding: 0.3rem 0.9rem;
  font-size: 0.75rem;
  font-weight: 500;
}

.tech-img-wrap {
  position: relative;
  min-height: 320px;
}

.tech-img-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}


/* Fusion block */
.fusion-block {
  margin-top: 2rem;
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  padding: 2.5rem 3rem;
}

.fusion-header {
  text-align: center;
  max-width: 580px;
  margin: 0 auto 2.5rem;
}

.fusion-header h3 {
  font-size: clamp(1.4rem, 3vw, 1.9rem);
  font-weight: 800;
  color: #0f172a;
  letter-spacing: -0.02em;
  margin: 0.5rem 0 0.75rem;
}

.fusion-header p {
  font-size: 0.925rem;
  color: #64748b;
  line-height: 1.8;
}

.fusion-compare {
  display: flex;
  align-items: center;
  gap: 2rem;
  max-width: 760px;
  margin: 0 auto;
  flex-wrap: wrap;
}

.compare-col {
  flex: 1;
  min-width: 220px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 1.5rem;
}

.compare-col-highlight {
  background: #eff6ff;
  border-color: #bfdbfe;
}

.compare-label {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #64748b;
  margin-bottom: 1.25rem;
}

.compare-col-highlight .compare-label {
  color: #2563eb;
}

.compare-stats {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.compare-stat {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.compare-val {
  font-size: 1.6rem;
  font-weight: 800;
  color: #0f172a;
  line-height: 1;
}

.highlight-val {
  color: #1d4ed8;
}

.compare-key {
  font-size: 0.72rem;
  color: #94a3b8;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.bar-track {
  height: 4px;
  background: #e2e8f0;
  border-radius: 100px;
  overflow: hidden;
  margin-top: 0.3rem;
}

.bar-fill {
  height: 100%;
  background: #94a3b8;
  border-radius: 100px;
}

.bar-fusion {
  background: #2563eb;
}

.bar-fpr {
  background: #f97316;
}

.bar-fpr-fusion {
  background: #22c55e;
}

.fusion-arrow {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
}

.fusion-arrow svg {
  width: 40px;
  height: 24px;
}

.fusion-arrow span {
  font-size: 0.7rem;
  font-weight: 600;
  color: #3b82f6;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.fusion-note {
  font-size: 0.8rem;
  color: #94a3b8;
  text-align: center;
  margin-top: 2rem;
  line-height: 1.7;
}

@media (max-width: 768px) {
  .tech-block { grid-template-columns: 1fr; }
  .benchmark-inner { flex-direction: column; gap: 1.5rem; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .tech-img-wrap { min-height: 220px; }
  .benchmark-block { padding: 2rem 1.25rem; }
  .tech-content { padding: 1.75rem 1.25rem; }
  .section-header { margin-bottom: 2.5rem; }
  .fusion-block { padding: 2rem 1.5rem; }
  .fusion-compare { gap: 1rem; }
}

@media (max-width: 600px) {
  .fusion-compare { flex-direction: column; }
  .fusion-arrow { flex-direction: column; align-items: center; margin: 0; }
  .fusion-arrow svg { transform: rotate(90deg); }
}

@media (max-width: 480px) {
  .stat-value { font-size: 1.5rem; }
  .benchmark-block { padding: 1.5rem 1rem; }
  .tech-pills { gap: 0.35rem; }
  .fusion-block { padding: 1.5rem 1rem; }
}
</style>
