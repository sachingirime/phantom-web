<template>
  <div>
    <NavBar />
    <HeroSection />
    <FeatureCards />
    <EmissionMap />
    <MethaneImportance />
    <PlatformPerformance />
    <ContactSection />
    <FooterSection />
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import NavBar from './components/NavBar.vue'
import HeroSection from './components/HeroSection.vue'
import FeatureCards from './components/FeatureCards.vue'
import EmissionMap from './components/EmissionMap.vue'
import MethaneImportance from './components/MethaneImportance.vue'
import PlatformPerformance from './components/PlatformPerformance.vue'
import ContactSection from './components/ContactSection.vue'
import FooterSection from './components/FooterSection.vue'

onMounted(() => {
  const observer = new IntersectionObserver(
    entries => entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target) } }),
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  )
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el))
})
</script>
