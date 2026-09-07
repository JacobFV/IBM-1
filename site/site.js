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

  // ---- the curriculum: one stage per row, a coloured dot on a rail, curved deps
  const dagEl = $('dag');
  if (dagEl && R && R.curriculum) {
    const stages = R.curriculum, byId = Object.fromEntries(stages.map((s) => [s.id, s]));
    const depth = {}; const d = (id) => depth[id] != null ? depth[id] : (depth[id] = byId[id].deps.length ? 1 + Math.max(...byId[id].deps.map(d)) : 0);
    stages.forEach((s) => d(s.id));
    const order = stages.slice().sort((a, b) => depth[a.id] - depth[b.id] || stages.indexOf(a) - stages.indexOf(b));
    const NOTE = {
      's0.gain': 'the divergence was a declaration problem: gain was per edge and a node has 4118 incoming edges. association transfer now divides by fan-in.',
      's0.window': 'selecting the nonlinearity gives the graph a 1.5 s memory, longer than half the window. τ = 0.30 s puts the slow oscillation in band and the plan clears.',
      's1.regime': 'fitted, not argued: the slow-oscillation peak across eight scored N3 nights is 1.000 ± 0.296 Hz, and τ = 0.12 s puts the model there with a 43.5 mV swing.',
      's2.spectra': 'the existing values are scalp quantities reported as cortical. redo; do not inherit. the global joint fit is falsified and is not to be re-attempted.',
      's3.paired': 'cochleagram to MEG through the cortex. the variance gate passed and the rank gate failed, which is the decorative-cortex signature; the honest skill is at chance.',
      's4.selfsup': '20,000 steps: reconstruction 1.77 → 0.185 while effective rank rose 1.42 → 2.97.',
      's4b.crossmodal': 'occipito-temporal weights at 4.4× the random-pair baseline, but severing every long-range edge costs +0.1%. magnitude is not contribution.',
      's4c.longrange': 'long-range edges took the local distance prior and started 7× weaker. they now take a flat patchy prior; the ablation has to be re-run to confirm severing them costs.',
      's5.ablate': 'the pivotal node. bypass +324%, frozen +180%, association zeroed +180%, local only +0.1%. the cortex is load-bearing and its long-range half is inert.',
      's6.scale': 'eleven minutes of one film is memorization territory. data binds the first milestone, not compute.',
      's7.joint': 'one substrate, two likelihood terms, a single optimizer step. at 250k sites the shared kernel is 32.0M and the heads 12.9M.',
      's8.curriculum': 'predictive loss alone is a capture curriculum. select where error is high and declining; hold one expansion measure out of the loss.',
      's9.rl': 'needs the regime stage for attractors to exist at all, and basal ganglia, hippocampal indexing and replay, whose anatomy is declared and whose processes are not written.',
    };
    const num = (id) => id.replace(/^s/, '').split('.')[0];
    dagEl.innerHTML = `<svg class="rail" aria-hidden="true"></svg><ol>${order.map((s) => `
      <li class="stage ${s.status}" data-id="${s.id}"><span class="dot">${s.status === 'done' ? '<svg viewBox="0 0 16 16"><path d="M3.5,8.5 l3,3 l6,-7"/></svg>' : num(s.id)}</span>
        <div class="stage-text"><h4>${esc(s.title)}<small>${s.status === 'ready' ? 'runnable' : s.status === 'failed' ? 'gate failed' : s.status}</small></h4><p>${esc(NOTE[s.id] || '')}</p><p class="gate">gate · ${esc(s.gate)}</p></div></li>`).join('')}</ol>`;
    const rail = dagEl.querySelector('.rail');
    const drawRail = () => {
      const box = dagEl.getBoundingClientRect();
      const at = {}; dagEl.querySelectorAll('.stage').forEach((li) => { const r = li.querySelector('.dot').getBoundingClientRect(); at[li.dataset.id] = [r.left - box.left + r.width / 2, r.top - box.top + r.height / 2]; });
      rail.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`); rail.setAttribute('width', box.width); rail.setAttribute('height', box.height);
      const idx = Object.fromEntries(order.map((s, k) => [s.id, k]));
      rail.innerHTML = stages.flatMap((s) => s.deps.map((dep) => {
        const a = at[dep], b = at[s.id]; if (!a || !b) return '';
        const gap = idx[s.id] - idx[dep], lit = byId[dep].status === 'done';
        if (gap === 1) return `<path class="${lit ? 'lit' : ''}" d="M${a[0]},${a[1] + 14} L${b[0]},${b[1] - 14}"/>`;
        const bow = 22 + 10 * gap;
        return `<path class="${lit ? 'lit' : ''}" d="M${a[0]},${a[1] + 14} C${a[0] - bow},${a[1] + 40} ${b[0] - bow},${b[1] - 40} ${b[0]},${b[1] - 14}"/>`;
      })).join('');
    };
    drawRail(); window.addEventListener('resize', drawRail);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(drawRail);
  }

  // ---- the corpus as a deck of cards: horizontal, swipeable, in perspective
  const deck = $('deck-scroll');
  if (deck && window.IBM_CARDS) {
    const cards = window.IBM_CARDS;
    const pad = (n) => String(n).padStart(3, '0');
    deck.innerHTML = cards.map((c) => `
      <article class="card" data-access="${c.access}">
        <div class="card-top"><a class="card-n" href="${c.url}" target="_blank" rel="noopener">#${pad(c.n)}</a><span class="card-access">${esc(c.access)}</span></div>
        <h4 class="card-title">${esc(c.title)}</h4>
        <p class="card-prov">${esc(c.provider)}</p>
        <p class="card-desc">${esc(c.desc)}</p>
        <p class="card-use">${c.use.map((u) => `<span>${esc(u)}</span>`).join('')}<span class="card-streams">${c.streams} stream${c.streams === 1 ? '' : 's'}</span></p>
      </article>`).join('');
    const held = cards.filter((c) => c.access === 'held').length;
    $('deck-count').textContent = `${held} held · ${cards.filter((c) => c.binding === 'bound').length} bound`;
    const els = Array.from(deck.children);
    let raf = null;
    const place = () => {
      raf = null;
      const mid = deck.scrollLeft + deck.clientWidth / 2;
      els.forEach((el) => {
        const c = el.offsetLeft + el.offsetWidth / 2, dx = (c - mid) / deck.clientWidth;
        if (Math.abs(dx) > 1.2) { el.style.transform = ''; el.style.opacity = '0'; return; }
        const t = Math.max(-1, Math.min(1, dx * 1.6));
        el.style.transform = `perspective(1100px) rotateY(${(-t * 42).toFixed(1)}deg) translateZ(${(-Math.abs(t) * 160).toFixed(0)}px)`;
        el.style.opacity = (1 - Math.abs(t) * 0.45).toFixed(2);
        el.style.zIndex = String(100 - Math.round(Math.abs(t) * 50));
      });
    };
    const schedule = () => { if (!raf) raf = requestAnimationFrame(place); };
    deck.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    // drag to swipe with a mouse; touch scrolls natively
    let drag = null;
    deck.addEventListener('pointerdown', (e) => { if (e.pointerType !== 'mouse') return; drag = { x: e.clientX, left: deck.scrollLeft, moved: false }; deck.classList.add('dragging'); });
    deck.addEventListener('pointermove', (e) => { if (!drag) return; const dx = e.clientX - drag.x; if (Math.abs(dx) > 4) drag.moved = true; deck.scrollLeft = drag.left - dx; });
    const end = () => { deck.classList.remove('dragging'); setTimeout(() => { drag = null; }, 0); };
    deck.addEventListener('pointerup', end); deck.addEventListener('pointercancel', end); deck.addEventListener('pointerleave', end);
    deck.addEventListener('click', (e) => { if (drag && drag.moved) e.preventDefault(); }, true);
    const step = () => (els[0] ? els[0].offsetWidth + 18 : 240) * 3;
    $('deck-prev').addEventListener('click', () => deck.scrollBy({ left: -step(), behavior: 'smooth' }));
    $('deck-next').addEventListener('click', () => deck.scrollBy({ left: step(), behavior: 'smooth' }));
    place();
    setTimeout(place, 300);
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
