<template>
  <footer class="footer">
    <div class="container">
      <div class="footer-top">
        <div class="footer-brand">
          <h2 class="footer-logo">PHANTOM</h2>
          <p class="footer-desc">
            Methane detection and emission rate estimation from airborne hyperspectral imagery.
            Research platform developed at MESA Lab, UC Merced.
          </p>
        </div>
        <div class="footer-nav">
          <div class="footer-col">
            <h4>Platform</h4>
            <a href="#what-we-do">What We Do</a>
            <a href="#detection-map">Detection Map</a>
            <a href="#performance">Performance</a>
          </div>
          <div class="footer-col">
            <h4>Science</h4>
            <a href="#methane">Why Methane</a>
            <a href="#performance">Validation</a>
          </div>
          <div class="footer-col">
            <h4>Contact</h4>
            <a href="mailto:contact@phantomch4.com">Request Access</a>
            <a href="mailto:contact@phantomch4.com">Data Services</a>
          </div>
        </div>
      </div>
      <div class="footer-bottom">
        <p>© {{ year }} PHANTOM Platform. All rights reserved.</p>
        <a href="mailto:contact@phantomch4.com" class="footer-email">contact@phantomch4.com</a>
      </div>
    </div>
  </footer>
</template>

<script setup>
const year = new Date().getFullYear()
</script>

<style scoped>
.footer {
  background: #080f1e;
  color: #fff;
  padding: 5rem 2rem 2.5rem;
  border-top: 1px solid rgba(255,255,255,0.06);
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.footer-top {
  display: flex;
  gap: 4rem;
  margin-bottom: 4rem;
  flex-wrap: wrap;
}

.footer-brand {
  flex: 1.5;
  min-width: 260px;
}

.footer-logo {
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  margin-bottom: 1rem;
}

.footer-desc {
  font-size: 0.9rem;
  color: rgba(148,163,184,0.75);
  line-height: 1.8;
  margin-bottom: 0.75rem;
  max-width: 380px;
}

.footer-affiliation {
  font-size: 0.8rem;
  color: rgba(148,163,184,0.45);
}

.footer-nav {
  display: flex;
  gap: 3rem;
  flex-wrap: wrap;
}

.footer-col {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  min-width: 120px;
}

.footer-col h4 {
  font-size: 0.72rem;
  font-weight: 700;
  color: rgba(148,163,184,0.5);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.25rem;
}

.footer-col a {
  color: rgba(255,255,255,0.65);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 500;
  transition: color 0.2s;
}

.footer-col a:hover {
  color: #fff;
}

.footer-bottom {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  border-top: 1px solid rgba(255,255,255,0.07);
  padding-top: 2rem;
}

.footer-bottom p {
  font-size: 0.78rem;
  color: rgba(148,163,184,0.35);
}

.footer-email {
  font-size: 0.82rem;
  color: rgba(148,163,184,0.55);
  text-decoration: none;
  transition: color 0.2s;
}
.footer-email:hover { color: #60a5fa; }

@media (max-width: 768px) {
  .footer-top { flex-direction: column; gap: 2.5rem; }
  .footer-nav { gap: 2rem; }
  .footer-bottom { flex-direction: column; align-items: flex-start; gap: 0.35rem; }
  .footer { padding: 3.5rem 1.25rem 2rem; }
}

@media (max-width: 480px) {
  .footer-nav { gap: 1.5rem; }
  .footer-col { min-width: 100px; }
}
</style>
