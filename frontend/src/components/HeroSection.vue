<template>
  <section class="hero">
    <div class="hero-bg" />
    <div class="hero-content">
      <div class="hero-badge reveal">
        <span class="badge-icon">📡</span>
        Permian Basin &nbsp;·&nbsp; AI-Validated Detection
      </div>
      <h1 class="hero-title reveal reveal-delay-1">PHANTOM</h1>
      <p class="hero-acronym reveal reveal-delay-2">
        Physics-Informed Hyperspectral Adversarial Network for Transformer-Optimized Methane Detection
      </p>
      <p class="hero-subtitle reveal reveal-delay-2">
        AI-driven methane detection and quantification from aerial hyperspectral imagery —
        validated on NASA AVIRIS-NG data, delivering actionable emission intelligence at scale.
      </p>
      <div class="hero-stats reveal reveal-delay-3">
        <div class="stat">
          <span class="stat-value">4×</span>
          <span class="stat-label">More Accurate</span>
        </div>
        <div class="stat-divider" />
        <div class="stat">
          <span class="stat-value">100%</span>
          <span class="stat-label">F1 on Critical Emitters</span>
        </div>
        <div class="stat-divider" />
        <div class="stat">
          <span class="stat-value">97 kg/hr</span>
          <span class="stat-label">Min. Detection Limit</span>
        </div>
        <div class="stat-divider" />
        <div class="stat">
          <span class="stat-value">178</span>
          <span class="stat-label">Sites Detected</span>
        </div>
      </div>
      <div class="hero-actions reveal reveal-delay-4">
        <a href="#detection-map" @click.prevent="scrollTo('detection-map')" class="btn-primary">View Detection Map</a>
        <a href="#what-we-do" @click.prevent="scrollTo('what-we-do')" class="btn-ghost">What We Do</a>
      </div>
    </div>
  </section>
</template>

<script setup>
function scrollTo(id) {
  const el = document.getElementById(id)
  if (el) { el.scrollIntoView({ behavior: 'smooth' }); history.replaceState(null, '', window.location.pathname) }
}
</script>

<style scoped>
.hero {
  position: relative;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background:
    linear-gradient(170deg, rgba(2,8,23,0.96) 0%, rgba(10,18,42,0.9) 50%, rgba(5,12,30,0.95) 100%),
    url('/images/earth-from-space.jpg') center/cover no-repeat;
  color: #fff;
  text-align: center;
  padding: 7rem 2rem 5rem;
  overflow: hidden;
}

.hero-bg {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 70% 50% at 50% 35%, rgba(37,99,235,0.09) 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 85% 75%, rgba(99,102,241,0.07) 0%, transparent 50%);
  pointer-events: none;
}

.hero-content {
  position: relative;
  max-width: 860px;
  width: 100%;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(37,99,235,0.12);
  border: 1px solid rgba(37,99,235,0.3);
  border-radius: 100px;
  padding: 0.4rem 1.25rem;
  font-size: 0.76rem;
  font-weight: 600;
  color: #93c5fd;
  letter-spacing: 0.05em;
  margin-bottom: 2rem;
  text-transform: uppercase;
}

.badge-icon { font-size: 0.85rem; }

.hero-title {
  font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-size: clamp(4rem, 14vw, 10rem);
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 0.9;
  margin-bottom: 1.5rem;
  background: linear-gradient(135deg, #fff 30%, rgba(148,163,184,0.55) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-acronym {
  font-size: clamp(0.72rem, 1.6vw, 0.875rem);
  color: rgba(100,116,139,0.85);
  letter-spacing: 0.02em;
  line-height: 1.7;
  margin-bottom: 1.25rem;
  font-style: italic;
  max-width: 640px;
  margin-left: auto;
  margin-right: auto;
}

.hero-subtitle {
  font-size: clamp(0.95rem, 2.2vw, 1.1rem);
  color: rgba(203,213,225,0.75);
  line-height: 1.8;
  margin-bottom: 3rem;
  max-width: 640px;
  margin-left: auto;
  margin-right: auto;
}

.hero-stats {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2rem;
  margin-bottom: 2.5rem;
  flex-wrap: wrap;
  padding: 1.75rem 2rem;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px;
  backdrop-filter: blur(8px);
}

.stat { display: flex; flex-direction: column; align-items: center; }

.stat-value {
  font-size: clamp(1.4rem, 3.5vw, 2.25rem);
  font-weight: 800;
  line-height: 1;
  color: #fff;
}

.stat-label {
  font-size: 0.65rem;
  font-weight: 500;
  color: rgba(100,116,139,0.85);
  margin-top: 0.35rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  text-align: center;
}

.stat-divider {
  width: 1px;
  height: 32px;
  background: rgba(255,255,255,0.1);
  flex-shrink: 0;
}

.hero-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.btn-primary {
  padding: 0.875rem 2rem;
  background: #2563eb;
  color: #fff;
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  text-decoration: none;
  transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
  box-shadow: 0 0 28px rgba(37,99,235,0.45);
}
.btn-primary:hover { background: #1d4ed8; transform: translateY(-2px); box-shadow: 0 0 40px rgba(37,99,235,0.6); }

.btn-ghost {
  padding: 0.875rem 2rem;
  border: 1px solid rgba(255,255,255,0.15);
  color: rgba(255,255,255,0.8);
  border-radius: 8px;
  font-weight: 600;
  font-size: 0.95rem;
  text-decoration: none;
  transition: border-color 0.2s, background 0.2s, color 0.2s;
}
.btn-ghost:hover { border-color: rgba(255,255,255,0.35); background: rgba(255,255,255,0.05); color: #fff; }

/* Tablet */
@media (max-width: 1024px) {
  .hero-stats { gap: 1.5rem; padding: 1.5rem; }
}

/* Mobile */
@media (max-width: 768px) {
  .hero { padding: 6rem 1.25rem 3.5rem; }
  .hero-badge { font-size: 0.68rem; padding: 0.35rem 1rem; text-align: center; }
  .hero-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    padding: 1.25rem;
  }
  .stat-divider { display: none; }
  .stat { padding: 0.5rem; background: rgba(255,255,255,0.04); border-radius: 10px; }
  .hero-actions { flex-direction: column; align-items: center; }
  .btn-primary, .btn-ghost { width: 100%; max-width: 320px; text-align: center; }
}

@media (max-width: 480px) {
  .hero { padding: 5.5rem 1rem 3rem; }
  .hero-badge { font-size: 0.62rem; }
}
</style>
