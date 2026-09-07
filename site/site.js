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

  // ---- the curriculum as a DAG: columns by dependency depth, status by colour
  const dagEl = $('dag');
  if (dagEl && R && R.curriculum) {
    const stages = R.curriculum, byId = Object.fromEntries(stages.map((s) => [s.id, s]));
    const depth = {}; const d = (id) => depth[id] != null ? depth[id] : (depth[id] = byId[id].deps.length ? 1 + Math.max(...byId[id].deps.map(d)) : 0);
    stages.forEach((s) => d(s.id));
    const cols = []; stages.forEach((s) => { (cols[depth[s.id]] = cols[depth[s.id]] || []).push(s); });
    const CW = 160, RH = 66, PADX = 56, PADY = 30, rows = Math.max(...cols.map((c) => c.length));
    const W = PADX * 2 + CW * (cols.length - 1), H = PADY * 2 + RH * (rows - 1) + 26;
    const pos = {};
    cols.forEach((c, ci) => { const off = (rows - c.length) / 2; c.forEach((s, ri) => { pos[s.id] = [PADX + ci * CW, PADY + (off + ri) * RH]; }); });
    const short = (id) => id.replace(/^s/, '').split('.')[0];
    const SHORT = { 's0.gain': 'gain prior', 's0.window': 'window', 's1.regime': 'regime', 's2.spectra': 'spectral fit', 's3.paired': 'paired MEG', 's4.selfsup': 'self-supervised AV', 's4b.crossmodal': 'cross-modal', 's4c.longrange': 'long-range', 's5.ablate': 'ablation', 's6.scale': 'scale', 's7.joint': 'joint schedule', 's8.curriculum': 'curriculum', 's9.rl': 'schema, then RL' };
    const edges = stages.flatMap((s) => s.deps.map((dep) => { const a = pos[dep], b = pos[s.id]; const lit = byId[dep].status === 'done';
      return `<path class="edge${lit ? ' lit' : ''}" d="M${a[0] + 14},${a[1]} C${a[0] + CW * 0.55},${a[1]} ${b[0] - CW * 0.55},${b[1]} ${b[0] - 14},${b[1]}"/>`; })).join('');
    const nodes = stages.map((s) => { const [x, y] = pos[s.id]; return `<g class="node ${s.status}" transform="translate(${x},${y})"><title>${s.id} — ${s.title}. gate: ${esc(s.gate)}</title><circle r="13"/><text class="n">${short(s.id)}</text><text class="t" y="27">${esc(SHORT[s.id] || s.title)}</text></g>`; }).join('');
    dagEl.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="curriculum stages">${edges}${nodes}</svg>
      <p class="dag-key"><span class="k-done">done</span><span class="k-running">running</span><span class="k-ready">runnable</span><span>blocked</span><span class="k-failed">gate failed</span></p>`;
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
