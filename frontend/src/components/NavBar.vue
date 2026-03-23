<template>
  <nav class="navbar" :class="{ scrolled: isScrolled }">
    <div class="nav-inner">
      <span class="nav-logo" @click="scrollTo('top')">
        <img src="/images/phantom_logo_nobg.png" alt="PHANTOM" class="nav-logo-img" />
      </span>

      <div class="nav-links">
        <a @click.prevent="scrollTo('what-we-do')" href="#">What We Do</a>
        <a @click.prevent="scrollTo('detection-map')" href="#">Detection Map</a>
        <a @click.prevent="scrollTo('methane')" href="#">Why Methane</a>
        <a @click.prevent="scrollTo('performance')" href="#">Performance</a>
        <a @click.prevent="scrollTo('contact')" href="#">Contact</a>
      </div>

      <a @click.prevent="scrollTo('contact')" href="#" class="nav-cta desktop-only">Request Demo</a>

      <button class="hamburger" :class="{ open: menuOpen }" @click="menuOpen = !menuOpen" aria-label="Menu">
        <span /><span /><span />
      </button>
    </div>

    <!-- Mobile menu -->
    <div class="mobile-menu" :class="{ open: menuOpen }">
      <a @click.prevent="mobileNav('what-we-do')" href="#">What We Do</a>
      <a @click.prevent="mobileNav('detection-map')" href="#">Detection Map</a>
      <a @click.prevent="mobileNav('methane')" href="#">Why Methane</a>
      <a @click.prevent="mobileNav('performance')" href="#">Performance</a>
      <a @click.prevent="mobileNav('contact')" href="#">Contact</a>
      <a @click.prevent="mobileNav('contact')" href="#" class="mobile-cta">Request Demo</a>
    </div>
  </nav>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const menuOpen = ref(false)
const isScrolled = ref(false)

function scrollTo(id) {
  if (id === 'top') { window.scrollTo({ top: 0, behavior: 'smooth' }); return }
  const el = document.getElementById(id)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth' })
    history.replaceState(null, '', window.location.pathname)
  }
}

function mobileNav(id) {
  menuOpen.value = false
  setTimeout(() => scrollTo(id), 200)
}

function onScroll() {
  isScrolled.value = window.scrollY > 20
}

onMounted(() => window.addEventListener('scroll', onScroll, { passive: true }))
onUnmounted(() => window.removeEventListener('scroll', onScroll))
</script>

<style scoped>
.navbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 1000;
  background: rgba(6, 18, 10, 0.72);
  backdrop-filter: blur(16px);
  border-bottom: 1px solid rgba(255,255,255,0.05);
  transition: background 0.3s, border-color 0.3s;
}
.navbar.scrolled {
  background: rgba(5, 15, 9, 0.97);
  border-bottom-color: rgba(255,255,255,0.09);
}

.nav-inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.5rem;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1.5rem;
}

.nav-logo {
  flex-shrink: 0;
  cursor: pointer;
  user-select: none;
  display: flex;
  align-items: center;
}

.nav-logo-img {
  height: 36px;
  width: auto;
  display: block;
  filter: brightness(1.1);
}

.nav-links {
  display: flex;
  gap: 1.75rem;
  flex: 1;
  justify-content: center;
}
.nav-links a {
  color: rgba(255,255,255,0.6);
  text-decoration: none;
  font-size: 0.875rem;
  font-weight: 500;
  transition: color 0.2s;
  white-space: nowrap;
  cursor: pointer;
}
.nav-links a:hover { color: #fff; }

.nav-cta {
  flex-shrink: 0;
  padding: 0.5rem 1.2rem;
  background: #16a34a;
  color: #fff !important;
  border-radius: 6px;
  font-size: 0.825rem;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.2s, box-shadow 0.2s;
  box-shadow: 0 0 16px rgba(22,163,74,0.35);
  white-space: nowrap;
  cursor: pointer;
}
.nav-cta:hover { background: #15803d; box-shadow: 0 0 24px rgba(22,163,74,0.55); }

/* Hamburger */
.hamburger {
  display: none;
  flex-direction: column;
  justify-content: center;
  gap: 5px;
  width: 36px;
  height: 36px;
  background: none;
  border: none;
  cursor: pointer;
  padding: 4px;
  flex-shrink: 0;
}
.hamburger span {
  display: block;
  height: 2px;
  background: #fff;
  border-radius: 2px;
  transition: transform 0.3s, opacity 0.3s, width 0.3s;
  transform-origin: center;
}
.hamburger.open span:nth-child(1) { transform: translateY(7px) rotate(45deg); }
.hamburger.open span:nth-child(2) { opacity: 0; }
.hamburger.open span:nth-child(3) { transform: translateY(-7px) rotate(-45deg); }

/* Mobile menu */
.mobile-menu {
  display: none;
  flex-direction: column;
  background: rgba(5, 15, 9, 0.98);
  border-top: 1px solid rgba(255,255,255,0.07);
  overflow: hidden;
  max-height: 0;
  transition: max-height 0.35s ease;
}
.mobile-menu.open { max-height: 420px; }
.mobile-menu a {
  padding: 1rem 1.5rem;
  color: rgba(255,255,255,0.75);
  text-decoration: none;
  font-size: 1rem;
  font-weight: 500;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  transition: color 0.2s, background 0.2s;
}
.mobile-menu a:hover { color: #fff; background: rgba(255,255,255,0.04); }
.mobile-cta {
  margin: 1rem 1.5rem 1.25rem !important;
  background: #16a34a !important;
  color: #fff !important;
  border-radius: 8px !important;
  text-align: center;
  border-bottom: none !important;
  font-weight: 600 !important;
}

@media (max-width: 768px) {
  .nav-links { display: none; }
  .desktop-only { display: none; }
  .hamburger { display: flex; }
  .mobile-menu { display: flex; }
}
</style>
