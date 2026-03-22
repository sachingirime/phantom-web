<template>
  <section class="hero">
    <div class="hero-bg" />
    <div class="hero-content">
      <div class="hero-badge reveal">
        Validated on NASA AVIRIS-NG
      </div>
      <h1 class="hero-title reveal reveal-delay-1">PHANTOM</h1>
      <p class="hero-acronym reveal reveal-delay-2">
        Greenhouse gas intelligent sensing that you can trust
      </p>
      <p class="hero-subtitle reveal reveal-delay-2">
        End-to-end methane sensing — from raw hyperspectral radiance to plume detection, emission quantification, source attribution, and structured reporting,
        validated on NASA AVIRIS-NG flight campaigns over the Permian Basin.
      </p>
      <div class="hero-stats reveal reveal-delay-3">
        <div class="stat">
          <span class="stat-value">100%</span>
          <span class="stat-label">F1, Critical Emitters</span>
        </div>
        <div class="stat">
          <span class="stat-value">96.65%</span>
          <span class="stat-label">F1, Standard Class</span>
        </div>
        <div class="stat">
          <span class="stat-value">97 Kg/hr</span>
          <span class="stat-label">Min. Verified Detection</span>
        </div>
        <div class="stat">
          <span class="stat-value">16 Kg/hr</span>
          <span class="stat-label">Lowest Observed Detection</span>
        </div>
        <div class="stat">
          <span class="stat-value">178</span>
          <span class="stat-label">Sites, Permian Basin</span>
        </div>
      </div>
      <div class="hero-actions reveal reveal-delay-4">
        <a href="#detection-map" @click.prevent="scrollTo('detection-map')" class="btn-primary">View Detection Map</a>
        <a href="#what-we-do" @click.prevent="scrollTo('what-we-do')" class="btn-ghost">How It Works</a>
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

.hero-bg::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 60px 60px;
  mask-image: radial-gradient(ellipse 80% 60% at 50% 40%, black 0%, transparent 80%);
  -webkit-mask-image: radial-gradient(ellipse 80% 60% at 50% 40%, black 0%, transparent 80%);
}

.hero-content {
  position: relative;
  max-width: 960px;
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


.hero-title {
  font-family: var(--font-display);
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
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  margin-bottom: 2.5rem;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 16px;
  backdrop-filter: blur(8px);
  overflow: hidden;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1.75rem 1rem;
  border-right: 1px solid rgba(255,255,255,0.07);
}

.stat:last-child { border-right: none; }

.stat-value {
  font-size: clamp(1.2rem, 2.8vw, 2rem);
  font-weight: 800;
  line-height: 1;
  color: #fff;
}

.stat-label {
  font-size: 0.62rem;
  font-weight: 500;
  color: rgba(100,116,139,0.85);
  margin-top: 0.4rem;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  text-align: center;
  max-width: 100px;
  line-height: 1.4;
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
  .hero-stats { grid-template-columns: repeat(5, 1fr); }
  .stat { padding: 1.5rem 0.75rem; }
}

/* Mobile */
@media (max-width: 768px) {
  .hero { padding: 6rem 1.25rem 3.5rem; }
  .hero-badge { font-size: 0.68rem; padding: 0.35rem 1rem; text-align: center; }
  .hero-stats {
    grid-template-columns: repeat(2, 1fr);
  }
  .stat:nth-child(2),
  .stat:nth-child(4) { border-right: none; }
  .stat:nth-child(3),
  .stat:nth-child(4),
  .stat:nth-child(5) { border-top: 1px solid rgba(255,255,255,0.07); }
  .stat:nth-child(5) { grid-column: span 2; border-right: none; }
  .hero-actions { flex-direction: column; align-items: center; }
  .btn-primary, .btn-ghost { width: 100%; max-width: 320px; text-align: center; }
}

@media (max-width: 480px) {
  .hero { padding: 5.5rem 1rem 3rem; }
  .hero-badge { font-size: 0.62rem; }
}
</style>
