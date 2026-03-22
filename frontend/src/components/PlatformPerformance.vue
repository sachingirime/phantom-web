<template>
  <section id="performance" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Validation</span>
        <h2>Platform Performance</h2>
        <p>AI-driven methane detection validated on real-world NASA AVIRIS-NG satellite data</p>
      </div>

      <div class="benchmark-block">
        <div class="benchmark-inner">
          <div class="benchmark-left">
            <h3>Benchmark Results</h3>
            <p>Validated on the full AVIRIS-NG California and Permian Basin flight campaigns with diverse terrain and environmental conditions. Outperforms existing detection methods by a factor of 4×.</p>
          </div>
          <div class="stats-grid">
            <div v-for="stat in stats" :key="stat.label" class="stat-item">
              <span class="stat-value">{{ stat.value }}</span>
              <span class="stat-label">{{ stat.label }}</span>
            </div>
          </div>
        </div>
        <div class="benchmark-note">
          <span>📡</span> Validated on NASA AVIRIS-NG hyperspectral imagery across real-world emission events ·
          Tested on diverse geographic locations and environmental conditions
        </div>
      </div>

      <div class="tech-block">
        <div class="tech-content">
          <h4>Technical Innovation</h4>
          <p>
            PHANTOM represents a breakthrough in methane detection technology by combining
            physics-informed neural networks with transformer-based architecture. Our system
            processes Short Wave Infrared spectral bands from hyperspectral sensors, applying
            matched filter preprocessing and adversarial training to achieve state-of-the-art
            performance. The platform delivers exceptional accuracy — outperforming existing
            detection methods by a factor of four.
          </p>
          <div class="tech-pills">
            <span v-for="tech in technologies" :key="tech" class="pill">{{ tech }}</span>
          </div>
        </div>
        <div class="tech-img-wrap">
          <img src="/images/aviris-ng.jpg" alt="NASA AVIRIS-NG hyperspectral sensor" />
          <div class="img-overlay">
            <span>NASA AVIRIS-NG Hyperspectral Sensor</span>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const stats = [
  { value: '100%', label: 'F1 Score (Critical)' },
  { value: '96.65%', label: 'F1 Score (Standard)' },
  { value: '11.43%', label: 'False Positive Rate' },
  { value: '62.42%', label: 'AUPRC' },
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

.img-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: linear-gradient(to top, rgba(15,23,42,0.85), transparent);
  padding: 2rem 1.5rem 1.25rem;
  color: #fff;
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}

@media (max-width: 768px) {
  .tech-block { grid-template-columns: 1fr; }
  .benchmark-inner { flex-direction: column; gap: 1.5rem; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .tech-img-wrap { min-height: 220px; }
  .benchmark-block { padding: 2rem 1.25rem; }
  .tech-content { padding: 1.75rem 1.25rem; }
  .section-header { margin-bottom: 2.5rem; }
}

@media (max-width: 480px) {
  .stat-value { font-size: 1.5rem; }
  .benchmark-block { padding: 1.5rem 1rem; }
  .tech-pills { gap: 0.35rem; }
}
</style>
