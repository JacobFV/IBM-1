/* wires the opener, the article's figures and the registry lists. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

  // ---- the registry lists in §1
  const R = window.IBM_REGISTRY;
  if (R) {
    const fill = (id, n, html) => { $('n-' + id).textContent = n; $('list-' + id).innerHTML = html; };
    fill('fields', `${R.fields.length} fields · ${R.fields.reduce((a, f) => a + f.components.length, 0)} components`, R.fields.map((f) => `
      <div class="reg-item"><div class="reg-head"><code>${f.id}</code><span>${esc(f.doc)}</span></div>
        <ul class="reg-comps">${f.components.map((c) => `<li><code>${c.id.replace(f.id + '.', '')}</code><small>${esc(c.units || '')}</small><span>${esc(c.doc)}</span></li>`).join('')}</ul></div>`).join(''));
    fill('anatomy', `${R.anatomy.length} systems · ${R.anatomy.reduce((a, s) => a + s.labels.length, 0)} labels`, R.anatomy.map((a) => `
      <div class="reg-item"><div class="reg-head"><code>${a.id}</code><span>${esc(a.doc)}</span><small>${a.frame} · ${a.labels.length}</small></div>
        <p class="reg-labels">${a.labels.map((l) => `<span>${esc(l)}</span>`).join('')}</p></div>`).join(''));
    fill('topologies', `${R.topologies.length}`, R.topologies.map((t) => `
      <div class="reg-item"><div class="reg-head"><code>${t.id}</code><span>${esc(t.doc)}</span><small>on ${t.on.join(', ')}${t.directed ? ' · directed' : ''}</small></div></div>`).join(''));
    const ts = (s) => s >= 1 ? `${s} s` : s >= 1e-3 ? `${+(s * 1e3).toPrecision(3)} ms` : s >= 1e-6 ? `${+(s * 1e6).toPrecision(3)} µs` : `${+(s * 1e9).toPrecision(3)} ns`;
    fill('processes', `${R.processes.length}`, R.processes.map((p) => `
      <div class="reg-item"><div class="reg-head"><code>${p.id}</code><span>${esc(p.doc)}</span><small>${p.topology || ''}${p.timescale_s ? ' · ' + ts(p.timescale_s) : ''}</small></div>
        <p class="reg-io"><span>${p.inputs.map((v) => `<code>${v}</code>`).join(' ')}</span><i>→</i><span>${p.outputs.map((v) => `<code>${v}</code>`).join(' ')}</span></p></div>`).join(''));
  }

  // ---- references: an underlined phrase opens a note with a link to read more
  const tip = $('tip');
  if (tip) {
    const text = tip.querySelector('.tip-text'), link = tip.querySelector('.tip-link');
    let current = null, hideT = null;
    const show = (el) => {
      clearTimeout(hideT);
      if (current && current !== el) current.classList.remove('is-open');
      current = el; el.classList.add('is-open');
      text.textContent = el.dataset.tip; link.href = el.href; link.textContent = el.dataset.doc;
      tip.hidden = false;
      const r = el.getBoundingClientRect(), w = tip.offsetWidth;
      let x = r.left + window.scrollX; if (x + w > window.scrollX + window.innerWidth - 12) x = window.scrollX + window.innerWidth - 12 - w;
      tip.style.left = Math.max(8, x) + 'px'; tip.style.top = (r.bottom + window.scrollY + 8) + 'px';
    };
    const hide = () => { hideT = setTimeout(() => { tip.hidden = true; if (current) current.classList.remove('is-open'); current = null; }, 180); };
    document.querySelectorAll('a.ref').forEach((el) => {
      el.addEventListener('mouseenter', () => show(el));
      el.addEventListener('mouseleave', hide);
      el.addEventListener('focus', () => show(el));
      el.addEventListener('blur', hide);
      el.addEventListener('click', (e) => { if (current !== el || tip.hidden) { e.preventDefault(); show(el); } });
    });
    tip.addEventListener('mouseenter', () => clearTimeout(hideT));
    tip.addEventListener('mouseleave', hide);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !tip.hidden) { tip.hidden = true; if (current) current.classList.remove('is-open'); } });
  }

  const B = window.IBMBrain;
  if (!B) return;
  B.createHero({
    hero: $('hero'), canvas: $('hero-canvas'), svg: $('hero-leaders'), ring: $('ring'),
    selTitle: $('sel-title'), selGroup: $('sel-group'), selDoc: $('sel-doc'), selMeta: $('sel-meta'),
    inCol: $('inputs'), outCol: $('outputs'), closeBtn: $('sel-close'), hint: $('hero-hint'),
  });

  const { A, finish, nodesWhere, group, rest, N } = B;
  function anatomyWeights(parts) {
    const w = new Float32Array(N);
    for (let i = 0; i < N; i++) w[i] = rest[i] * 0.55;
    parts.forEach((fn) => nodesWhere(fn).forEach((i) => { w[i] = 1; }));
    return w;
  }
  B.mountSnapshots({
    anatomy: {
      az: -0.95, el: 0.22, dist: 410, aspect: 0.62,
      weights: () => anatomyWeights([(i) => B.region[i] === 'precentral' && B.hemi[i] === 'lh', (i) => B.region[i] === 'thalamus', (i) => B.region[i] === 'cerebellum', (i) => group[i] === 'eeg', (i) => group[i] === 'spinal']),
      annotations: () => finish([
        A.region('precentral gyrus', 'precentral', 'lh', { sub: 'cortical_areas · aparc', side: 'l' }),
        A.region('thalamus', 'thalamus', null, { sub: 'aseg · 31 declared nuclei', side: 'r' }),
        A.region('cerebellum', 'cerebellum', null, { sub: '21 declared lobules', side: 'r' }),
        A.group('EEG contacts', 'eeg', { sub: 'sensor_array · 60 digitised', side: 'l' }),
        A.group('spinal stub', 'spinal', { sub: 'drawn for orientation', side: 'r' }),
      ]),
    },
    materialize: { model: 'eeg_forward', az: -0.5, el: 0.28, dist: 440, aspect: 0.62, annotations: () => A.model_io('eeg_forward') },
    meg: { model: 'meg_to_text', az: -1.35, el: 0.18, dist: 430, aspect: 0.62, annotations: () => A.model_io('meg_to_text') },
    l_forward: { model: 'eeg_forward', az: -0.6, el: 0.3, dist: 470, aspect: 0.8 },
    l_decode: { model: 'meg_to_text', az: -1.3, el: 0.2, dist: 470, aspect: 0.8 },
    l_stim: { model: 'tms_response', az: -0.9, el: 0.55, dist: 470, aspect: 0.8 },
    l_state: { model: 'sleep_dynamics', az: 0.5, el: 0.3, dist: 470, aspect: 0.8 },
    l_body: { model: 'invasive_bci', az: -0.8, el: 0.45, dist: 470, aspect: 0.8 },
    l_surrogate: { model: 'macro_surrogate', az: -0.6, el: 0.3, dist: 470, aspect: 0.8 },
  });
})();
