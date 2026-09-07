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
      's0.gain': 'early on, the simulation kept blowing up. the cause: signal strength was set per connection, and some spots in the model have thousands of connections feeding into them at once, so their combined signal was way too strong. the fix scales each connection down by how many others feed into the same place.',
      's0.window': 'the model needs to remember enough of the recent past to catch a slow brain rhythm, without wasting effort holding onto more than that. giving it about a third of a second of memory turned out to be the sweet spot.',
      's1.regime': 'real deep-sleep recordings show the brain\'s slowest wave rising and falling about once per second — measured across eight nights of sleep, not assumed. tuning the model\'s internal timing to match produces waves of a realistic size.',
      's2.spectra': 'some of the frequency numbers used so far were actually measured at the scalp, then mislabeled as if they described activity inside the brain — so they need to be re-measured properly. an earlier attempt to fit everything in one shot turned out wrong and will not be repeated.',
      's3.paired': 'this stage tries to predict real brain-scan recordings straight from sound. it looked promising by one measure, but a stricter check showed it wasn\'t really learning anything yet — just dressing up its numbers. honestly scored, it currently does no better than a coin flip.',
      's4.selfsup': 'the model was trained to guess what happens next in audio and video it has already seen, with no answer key. over the course of training it got substantially better at that guessing game, and started drawing on a richer variety of internal patterns to do it.',
      's4b.crossmodal': 'the connections linking the model\'s vision areas to its hearing and memory areas did grow noticeably stronger than chance would predict. but cutting those same long-distance connections barely changed performance — so despite looking meaningful, they are not yet doing real work.',
      's4c.longrange': 'the model\'s long-distance connections were starting out far too weak, because they inherited an assumption meant for short, local ones. they now start from a fairer baseline, and the earlier cutting test needs to be re-run to see whether it matters now.',
      's5.ablate': 'the key question: does the shared brain-like core actually matter, or could it be skipped? removing it, freezing it, or switching off what it had learned to associate all made the model much worse — so it clearly matters. but its long-distance wiring specifically isn\'t pulling its weight yet.',
      's6.scale': 'training on just eleven minutes of one film risks the model simply memorizing that clip rather than learning general patterns. more computing power won\'t fix that — what\'s needed next is more varied footage.',
      's7.joint': 'the model learns from two different kinds of data — video and brain recordings — at the same time, in one combined update, instead of alternating between them. most of its parameters are shared across both; only a smaller piece is specific to each.',
      's8.curriculum': 'training only on whatever is easiest to predict lets a model coast on shortcuts. instead, examples are chosen where it is still making mistakes but steadily improving, and progress is checked against a separate measure that isn\'t part of what it\'s optimizing for.',
      's9.rl': 'this stage — teaching the model to act toward a goal through trial and reward — can\'t start until earlier stages give it stable internal "concepts" to reason with, and until the brain regions for habit, memory, and replaying past experience, which exist only as placeholders today, are actually built.',
    };
    const num = (id) => id.replace(/^s/, '').split('.')[0];
    dagEl.innerHTML = `<svg class="rail" aria-hidden="true"></svg><ol>${order.map((s) => `
      <li class="stage ${s.status}" data-id="${s.id}"><span class="dot">${s.status === 'done' ? '<svg viewBox="0 0 16 16"><path d="M3.5,8.5 l3,3 l6,-7"/></svg>' : num(s.id)}</span>
        <div class="stage-text"><h4>${esc(s.title)}<small>${s.status === 'ready' ? 'runnable' : s.status === 'failed' ? 'gate failed' : s.status}</small></h4><p>${esc(NOTE[s.id] || '')}</p></div></li>`).join('')}</ol>`;
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
        <a class="card-n" href="${c.url}" target="_blank" rel="noopener">#${pad(c.n)}</a>
        <h4 class="card-title">${esc(c.title)}</h4>
        <p class="card-desc">${esc(c.desc)}</p>
      </article>`).join('');
    const els = Array.from(deck.children);
    const FADE = 1.2;
    let raf = null;
    const place = () => {
      raf = null;
      const mid = deck.scrollLeft + deck.clientWidth / 2;
      els.forEach((el) => {
        const c = el.offsetLeft + el.offsetWidth / 2, dx = (c - mid) / deck.clientWidth;
        if (Math.abs(dx) > FADE) { el.style.transform = ''; el.style.opacity = '0'; return; }
        const t = Math.max(-1, Math.min(1, dx * 1.6));
        el.style.transform = `perspective(1100px) rotateY(${(-t * 42).toFixed(1)}deg) translateZ(${(-Math.abs(t) * 160).toFixed(0)}px)`;
        el.style.opacity = Math.max(0, 1 - Math.abs(dx) / FADE).toFixed(2);
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
    l_forward: { model: 'eeg_forward', az: -0.6, el: 0.3, dist: 470, aspect: 0.8 },
    l_decode: { model: 'meg_to_text', az: -1.3, el: 0.2, dist: 470, aspect: 0.8 },
    l_stim: { model: 'tms_response', az: -0.9, el: 0.55, dist: 470, aspect: 0.8 },
    l_state: { model: 'sleep_dynamics', az: 0.5, el: 0.3, dist: 470, aspect: 0.8 },
    l_body: { model: 'invasive_bci', az: -0.8, el: 0.45, dist: 470, aspect: 0.8 },
    l_surrogate: { model: 'macro_surrogate', az: -0.6, el: 0.3, dist: 470, aspect: 0.8 },
  });
})();
