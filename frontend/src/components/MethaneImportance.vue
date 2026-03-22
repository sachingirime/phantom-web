<template>
  <section id="methane" class="section section-dark">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Climate Impact</span>
        <h2>Why Methane Concerns Matter</h2>
        <p>
          Methane is responsible for 30% of global warming since pre-industrial times.
          Detecting and reducing methane emissions is our fastest opportunity to slow climate change in the next decade.
        </p>
      </div>

      <div class="impact-grid">
        <div v-for="card in impactCards" :key="card.title" class="impact-card">
          <div class="impact-number">{{ card.number }}</div>
          <div class="impact-title">{{ card.title }}</div>
          <p class="impact-text">{{ card.text }}</p>
        </div>
      </div>

      <div class="sources-section">
        <h3 class="sub-heading">Emission Sources</h3>
        <div class="sources-grid">
          <div class="sources-block">
            <span class="block-tag">Major Emission Sources</span>
            <p>
              Human activities release methane from three primary sectors: fossil fuels (oil,
              gas, and coal operations), agriculture (livestock enteric fermentation and rice cultivation),
              and waste management (landfills and wastewater treatment). Natural sources — wetlands and
              other ecosystems — also contribute, but anthropogenic emissions represent the fastest and
              most controllable reduction opportunity.
            </p>
            <div class="img-wrap tall infographic">
              <img src="/images/methane-sources.png" alt="Methane emission sources" />
            </div>
          </div>

          <div class="sources-block">
            <span class="block-tag">Satellite Detection Importance</span>
            <p>
              Traditional ground-based monitoring covers less than 5% of global emission sites.
              Satellite hyperspectral imaging enables complete, objective global coverage — tracking
              emissions from remote oil fields, wetlands, and agricultural regions that would otherwise
              go undetected. PHANTOM makes this data actionable through AI-driven detection and
              real-time quantification, enabling targeted mitigation at the largest sources.
            </p>
            <div class="img-wrap tall infographic">
              <img src="/images/climate-impact.png" alt="Climate impact of methane" />
            </div>
          </div>
        </div>

        <!-- Wide atmospheric concentration image -->
        <div class="wide-img-block">
          <span class="block-tag">Atmospheric Methane Concentration</span>
          <div class="img-wrap wide infographic">
            <img src="/images/atmospheric-concentration.png" alt="Atmospheric methane concentration trend" />
          </div>
          <p class="img-caption">
            Global atmospheric methane concentration has risen sharply since industrialization,
            now exceeding 1,900 ppb — nearly triple pre-industrial levels. Rapid detection and
            mitigation are critical to reversing this trend.
          </p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
const impactCards = [
  {
    number: '80×',
    title: 'Warming Potential',
    text: 'Methane traps 80 times more heat than CO₂ over a 20-year period, making it the most critical greenhouse gas for immediate climate action.',
  },
  {
    number: '30%',
    title: 'Global Warming Contribution',
    text: 'Responsible for approximately 30% of global temperature rise. Its potency makes it second only to carbon dioxide in overall climate impact.',
  },
  {
    number: '45%',
    title: 'Reduction Potential by 2030',
    text: 'Cutting methane emissions by 45% by 2030 could prevent 0.3°C of warming by 2040 — our fastest and most cost-effective climate solution.',
  },
]
</script>

<style scoped>
.section {
  padding: 6rem 2rem;
}

.section-dark {
  background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
  color: #fff;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
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
  margin-bottom: 1rem;
}

.section-header p {
  font-size: 1.1rem;
  color: rgba(203,213,225,0.88);
  line-height: 1.8;
}

.impact-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 1.5rem;
  margin-bottom: 5rem;
}

.impact-card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 2.5rem;
  text-align: center;
  transition: background 0.25s, transform 0.25s;
}

.impact-card:hover {
  background: rgba(255,255,255,0.09);
  transform: translateY(-4px);
}

.impact-number {
  font-size: clamp(2.5rem, 6vw, 4rem);
  font-weight: 800;
  color: #60a5fa;
  line-height: 1;
  margin-bottom: 0.75rem;
}

.impact-title {
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 0.75rem;
}

.impact-text {
  font-size: 1rem;
  color: rgba(203,213,225,0.85);
  line-height: 1.8;
}

/* Sources */
.sources-section {
  border-top: 1px solid rgba(255,255,255,0.1);
  padding-top: 4rem;
}

.sub-heading {
  font-size: clamp(1.5rem, 4vw, 2rem);
  font-weight: 800;
  text-align: center;
  margin-bottom: 3rem;
  letter-spacing: -0.02em;
}

.sources-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 3rem;
  margin-bottom: 3.5rem;
}

.sources-block {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.block-tag {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #60a5fa;
}

.sources-block p {
  font-size: 1rem;
  color: rgba(203,213,225,0.9);
  line-height: 1.85;
}

.img-wrap {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.1);
}

.img-wrap.tall {
  height: 340px;
}

.img-wrap.wide {
  height: 360px;
  width: 100%;
}

.img-wrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* Infographic images (charts, diagrams) should not be cropped */
.img-wrap.infographic {
  background: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
}
.img-wrap.infographic img {
  object-fit: contain;
  padding: 0.5rem;
}

.img-wrap.cover img {
  object-fit: cover;
}

.wide-img-block {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.img-caption {
  font-size: 0.95rem;
  color: rgba(203,213,225,0.65);
  text-align: center;
  line-height: 1.7;
  max-width: 700px;
  margin: 0 auto;
}

@media (max-width: 768px) {
  .sources-grid { grid-template-columns: 1fr; gap: 2rem; }
  .sources-section { padding-top: 2.5rem; }
  .sub-heading { margin-bottom: 2rem; }
  .impact-grid { gap: 1rem; }
  .impact-card { padding: 1.75rem; }
  .img-wrap.tall { height: 260px; }
  .img-wrap.wide { height: 260px; }
  .section-header { margin-bottom: 2.5rem; }
}

@media (max-width: 480px) {
  .impact-grid { grid-template-columns: 1fr; }
  .img-wrap.tall { height: 220px; }
  .img-wrap.wide { height: 220px; }
}
</style>
