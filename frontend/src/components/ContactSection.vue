<template>
  <section id="contact" class="section">
    <div class="container">
      <div class="section-header">
        <span class="section-tag">Contact</span>
        <h2>Get in Touch</h2>
        <p>
          Reach out to discuss detection capabilities, data access, or potential research collaborations.
        </p>
      </div>

      <div class="contact-grid">
        <!-- Form -->
        <div class="form-card">
          <form
            v-if="!submitted"
            action="https://formspree.io/f/xlgpkzaz"
            method="POST"
            @submit.prevent="handleSubmit"
          >
            <div class="field-row">
              <div class="field">
                <label>Full Name *</label>
                <input v-model="form.name" type="text" name="name" placeholder="Jane Smith" required />
              </div>
              <div class="field">
                <label>Organization *</label>
                <input v-model="form.org" type="text" name="organization" placeholder="Acme Energy Corp." required />
              </div>
            </div>

            <div class="field">
              <label>Email Address *</label>
              <input v-model="form.email" type="email" name="email" placeholder="jane@company.com" required />
            </div>

            <div class="field">
              <label>Use Case</label>
              <select v-model="form.usecase" name="use_case">
                <option value="">Select your primary use case</option>
                <option>Oil & Gas Leak Detection</option>
                <option>ESG / Regulatory Reporting</option>
                <option>Academic / Research</option>
                <option>Government / Policy</option>
                <option>Data Integration / API</option>
                <option>Other</option>
              </select>
            </div>

            <div class="field">
              <label>Message</label>
              <textarea v-model="form.message" name="message" rows="4" placeholder="Tell us about your monitoring needs, region of interest, or any questions..." />
            </div>

            <button type="submit" class="btn-submit" :disabled="loading">
              <span v-if="loading">Sending…</span>
              <span v-else>Send Request →</span>
            </button>

            <p v-if="error" class="form-error">{{ error }}</p>
          </form>

          <!-- Success state -->
          <div v-else class="success-state">
            <div class="success-icon">✓</div>
            <h3>Message Received</h3>
            <p>Thank you. We will follow up at <strong>{{ form.email }}</strong> as soon as possible.</p>
            <button class="btn-reset" @click="submitted = false">Send another</button>
          </div>
        </div>

        <!-- Sidebar -->
        <div class="sidebar">
          <div class="sidebar-card">
            <h4>Direct Contact</h4>
            <a href="mailto:contact@phantomch4.com" class="email-btn">
              <span class="email-icon">✉</span>
              <div>
                <div class="email-label">Email Us</div>
                <div class="email-addr">contact@phantomch4.com</div>
              </div>
            </a>
          </div>

          <div class="sidebar-card">
            <h4>What to Expect</h4>
            <ul class="expect-list">
              <li>
                <span class="dot" />
                Overview of detection and quantification outputs
              </li>
              <li>
                <span class="dot" />
                Discussion of your region or use case
              </li>
              <li>
                <span class="dot" />
                Information on data deliverable formats
              </li>
              <li>
                <span class="dot" />
                Research collaboration and partnership inquiries welcome
              </li>
            </ul>
          </div>

        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref } from 'vue'

const form = ref({ name: '', org: '', email: '', usecase: '', message: '' })
const submitted = ref(false)
const loading = ref(false)
const error = ref('')

async function handleSubmit() {
  loading.value = true
  error.value = ''
  try {
    const res = await fetch('https://formspree.io/f/xlgpkzaz', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        name: form.value.name,
        organization: form.value.org,
        email: form.value.email,
        use_case: form.value.usecase,
        message: form.value.message,
      }),
    })
    if (res.ok) {
      submitted.value = true
    } else {
      error.value = 'Something went wrong. Please email us directly.'
    }
  } catch {
    error.value = 'Network error. Please email contact@phantomch4.com directly.'
  } finally {
    loading.value = false
  }
}
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
  max-width: 600px;
  margin: 0 auto 4rem;
}

.section-header h2 {
  font-size: clamp(2rem, 5vw, 3rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #0f172a;
  margin-bottom: 1rem;
}

.section-header p {
  font-size: 1.05rem;
  color: #64748b;
  line-height: 1.8;
}

.contact-grid {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: 2rem;
  align-items: start;
}

/* Form card */
.form-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 20px;
  padding: 2.5rem;
}

.field-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  margin-bottom: 1.25rem;
}

.field label {
  font-size: 0.82rem;
  font-weight: 600;
  color: #374151;
  letter-spacing: 0.01em;
}

.field input,
.field select,
.field textarea {
  font-family: inherit;
  font-size: 0.925rem;
  color: #0f172a;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.75rem 1rem;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  width: 100%;
}

.field input:focus,
.field select:focus,
.field textarea:focus {
  border-color: #16a34a;
  box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.12);
  background: #fff;
}

.field textarea {
  resize: vertical;
  min-height: 110px;
}

.btn-submit {
  width: 100%;
  padding: 0.9rem;
  background: #16a34a;
  color: #fff;
  border: none;
  border-radius: 10px;
  font-family: inherit;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s, transform 0.2s, box-shadow 0.2s;
  box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35);
  margin-top: 0.5rem;
}

.btn-submit:hover:not(:disabled) {
  background: #15803d;
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(22, 163, 74, 0.45);
}

.btn-submit:disabled {
  opacity: 0.65;
  cursor: not-allowed;
}

.form-error {
  margin-top: 0.75rem;
  font-size: 0.85rem;
  color: #dc2626;
  text-align: center;
}

/* Success */
.success-state {
  text-align: center;
  padding: 3rem 1rem;
}

.success-icon {
  width: 60px;
  height: 60px;
  background: #dcfce7;
  color: #16a34a;
  border-radius: 50%;
  font-size: 1.6rem;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1.5rem;
  font-weight: 700;
}

.success-state h3 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.75rem;
}

.success-state p {
  font-size: 1rem;
  color: #64748b;
  line-height: 1.7;
  margin-bottom: 1.5rem;
}

.btn-reset {
  background: none;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 0.6rem 1.5rem;
  font-family: inherit;
  font-size: 0.875rem;
  color: #64748b;
  cursor: pointer;
  transition: border-color 0.2s, color 0.2s;
}

.btn-reset:hover {
  border-color: #16a34a;
  color: #16a34a;
}

/* Sidebar */
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.sidebar-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 1.5rem;
}

.sidebar-card h4 {
  font-size: 0.78rem;
  font-weight: 700;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  margin-bottom: 1rem;
}

.email-btn {
  display: flex;
  align-items: center;
  gap: 1rem;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 10px;
  padding: 1rem 1.25rem;
  text-decoration: none;
  transition: background 0.2s, border-color 0.2s, transform 0.2s;
}

.email-btn:hover {
  background: #dcfce7;
  border-color: #86efac;
  transform: translateY(-2px);
}

.email-icon {
  font-size: 1.4rem;
  color: #16a34a;
  flex-shrink: 0;
}

.email-label {
  font-size: 0.78rem;
  font-weight: 700;
  color: #16a34a;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.email-addr {
  font-size: 0.9rem;
  color: #15803d;
  font-weight: 500;
  margin-top: 0.1rem;
}

.expect-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.expect-list li {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.9rem;
  color: #374151;
  line-height: 1.5;
}

.dot {
  width: 7px;
  height: 7px;
  background: #16a34a;
  border-radius: 50%;
  flex-shrink: 0;
}


@media (max-width: 900px) {
  .contact-grid { grid-template-columns: 1fr; }
  .field-row { grid-template-columns: 1fr; }
  .sidebar { flex-direction: row; flex-wrap: wrap; }
  .sidebar-card { flex: 1; min-width: 260px; }
}

@media (max-width: 640px) {
  .sidebar { flex-direction: column; }
  .sidebar-card { min-width: unset; }
  .form-card { padding: 1.5rem 1.25rem; }
  .section-header { margin-bottom: 2.5rem; }
}

@media (max-width: 480px) {
  .form-card { padding: 1.25rem 1rem; }
  .btn-submit { font-size: 0.95rem; }
}
</style>
