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
        <ul class="reg-comps">${f.components.map((c) => `<li><span class="comp-id"><code>${c.id.replace(f.id + '.', '')}</code>${c.units ? `<small>${esc(c.units)}</small>` : ''}</span><span>${esc(c.doc)}</span></li>`).join('')}</ul></div>`).join(''));
    fill('anatomy', `${R.anatomy.length} systems · ${R.anatomy.reduce((a, s) => a + s.labels.length, 0)} labels`, R.anatomy.map((a) => `
      <div class="reg-item"><div class="reg-head"><code>${a.id}</code><span>${esc(a.doc)}</span><small>${a.frame} · ${a.labels.length}</small></div>
        <p class="reg-labels">${a.labels.map((l) => `<span>${esc(l)}</span>`).join('')}</p></div>`).join(''));
    fill('topologies', `${R.topologies.length}`, R.topologies.map((t) => `
      <div class="reg-item"><div class="reg-head"><code>${t.id}</code><span>${esc(t.doc)}</span><small>on ${t.on.join(', ')}${t.directed ? ' · directed' : ''}</small></div></div>`).join(''));
    const ts = (s) => s >= 1 ? `${s} s` : s >= 1e-3 ? `${+(s * 1e3).toPrecision(3)} ms` : s >= 1e-6 ? `${+(s * 1e6).toPrecision(3)} µs` : `${+(s * 1e9).toPrecision(3)} ns`;
    fill('processes', `${R.processes.length}`, R.processes.map((p) => `
      <div class="reg-item"><div class="reg-head"><code>${p.id}</code><span>${esc(p.doc)}</span><small>${p.topology || ''}${p.timescale_s ? ' · ' + ts(p.timescale_s) : ''}</small></div>
        <p class="reg-io"><i>in</i><span>${p.inputs.map((v) => `<code>${v}</code>`).join(' ')}</span><i>out</i><span>${p.outputs.map((v) => `<code>${v}</code>`).join(' ')}</span></p></div>`).join(''));
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

  // ---- the corpus as an endless deck: one card in focus, the rest fanned
  // behind it and fading out.  the list is a ring -- swiping never reaches an
  // end -- so the cards are a small pool of elements re-filled as it turns,
  // placed by transform rather than by scrolling six hundred of them.
  const deck = $('deck-scroll');
  if (deck && window.IBM_CARDS && window.IBM_CARDS.length) {
    const CARDS = window.IBM_CARDS, NC = CARDS.length;
    const pad = (n) => String(n).padStart(3, '0');
    const mod = (a, b) => ((a % b) + b) % b;
    const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
    const pool = [];
    let cardW = 214, pitch = 146, span = 4;      // span: cards from focus to invisible
    let at = 0, vel = 0, snap = null, raf = null, drag = null, wheelT = null, dragEnded = 0;

    const grow = (n) => {
      while (pool.length < n) {
        const el = document.createElement('article');
        el.className = 'card';
        el.innerHTML = '<a class="card-n" target="_blank" rel="noopener"></a><h4 class="card-title"></h4><p class="card-desc"></p>';
        deck.appendChild(el);
        pool.push({ el, n: el.querySelector('.card-n'), t: el.querySelector('.card-title'), d: el.querySelector('.card-desc'), idx: -1 });
      }
      while (pool.length > n) pool.pop().el.remove();
    };
    // the deck is as wide as the window, not the text column, so no card is cut
    // off against a column edge; it fades out well before the page edge anyway.
    const layout = () => {
      const vw = document.documentElement.clientWidth;
      deck.style.width = vw + 'px';
      deck.style.marginLeft = -deck.parentElement.getBoundingClientRect().left + 'px';
      if (!pool.length) grow(3);
      cardW = pool[0].el.offsetWidth || 214;
      pitch = cardW * 0.72;                      // the cards overlap: a denser fan
      span = Math.max(3.2, (vw / 2 - cardW * 0.25) / pitch);
      grow(Math.min(27, 2 * Math.ceil(span) + 3));
    };
    const place = () => {
      const base = Math.round(at), frac = at - base, mid = (pool.length - 1) / 2;
      pool.forEach((s, n) => {
        const j = n - mid, k = j - frac, i = mod(base + j, NC);
        if (s.idx !== i) {
          const c = CARDS[i];
          s.idx = i; s.n.textContent = '#' + pad(c.n); s.n.href = c.url;
          s.t.textContent = c.title; s.d.textContent = c.desc; s.el.dataset.access = c.access;
        }
        const o = clamp(1.12 * (1 - Math.abs(k) / span), 0, 1);
        s.el.style.opacity = o.toFixed(3);
        s.el.hidden = o <= 0;
        if (o <= 0) return;
        const t = clamp(k / 2.4, -1, 1);
        // the card in focus keeps a little clearance; behind it they pack tight
        const x = k * pitch + (k < 0 ? -1 : 1) * Math.min(Math.abs(k), 1) * cardW * 0.1;
        s.el.style.transform = `translateX(-50%) translateX(${x.toFixed(1)}px) rotateY(${(-t * 46).toFixed(1)}deg) translateZ(${(-Math.abs(t) * 170).toFixed(0)}px)`;
        s.el.style.zIndex = String(100 - Math.round(Math.abs(k) * 10));
      });
    };
    // momentum after a swipe, then a settle onto whichever card is nearest
    const tick = () => {
      raf = null;
      if (snap != null) {
        at += (snap - at) * 0.2;
        if (Math.abs(snap - at) < 0.002) { at = snap; snap = null; }
      } else if (Math.abs(vel) > 2e-4) {
        at += vel; vel *= 0.93;
        if (Math.abs(vel) <= 2e-4) { vel = 0; snap = Math.round(at); }
      }
      place();
      if (snap != null || Math.abs(vel) > 2e-4) raf = requestAnimationFrame(tick);
    };
    const run = () => { if (!raf) raf = requestAnimationFrame(tick); };

    deck.addEventListener('pointerdown', (e) => {
      if (e.button) return;
      deck.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, at, moved: false, t: performance.now(), v: 0 };
      vel = 0; snap = null; deck.classList.add('dragging');
    });
    deck.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, was = at, now = performance.now();
      if (Math.abs(dx) > 4) drag.moved = true;
      at = drag.at - dx / pitch;
      drag.v = (at - was) / Math.max(8, now - drag.t) * 16;
      drag.t = now;
      place();
    });
    const release = () => {
      if (!drag) return;
      vel = clamp(drag.v, -0.5, 0.5);
      if (Math.abs(vel) < 0.004) { vel = 0; snap = Math.round(at); }
      if (drag.moved) dragEnded = performance.now();
      drag = null; deck.classList.remove('dragging');
      run();
    };
    deck.addEventListener('pointerup', release);
    deck.addEventListener('pointercancel', release);
    deck.addEventListener('click', (e) => { if (performance.now() - dragEnded < 120) e.preventDefault(); }, true);
    // a horizontal wheel (or shift-wheel) turns the deck; a plain one scrolls the page
    deck.addEventListener('wheel', (e) => {
      const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : (e.shiftKey ? e.deltaY : 0);
      if (!dx) return;
      e.preventDefault();
      snap = null; vel = 0; at += dx / pitch / 3; place();
      clearTimeout(wheelT); wheelT = setTimeout(() => { snap = Math.round(at); run(); }, 140);
    }, { passive: false });
    deck.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      vel = 0; snap = Math.round(snap == null ? at : snap) + (e.key === 'ArrowRight' ? 1 : -1); run();
    });
    window.addEventListener('resize', () => { layout(); place(); });
    layout(); place();
    setTimeout(() => { layout(); place(); }, 300);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { layout(); place(); });
  }

  // ---- §4: a model is a name; the checkpoints released for it live inside it
  // (site/data/releases.js, written by scripts/export_releases.py from the hub)
  const REL = window.IBM_RELEASES;
  const params = (n) => n == null ? '—' : n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? Math.round(n / 1e3) + 'k' : String(n);
  const ckptsOf = (el) => (REL && REL.models && REL.models[el.dataset.model]) || null;
  const hf = '<span class="hf" aria-hidden="true">\u{1f917}</span>';
  if (REL) {
    document.querySelectorAll('.lineup li[data-model]').forEach((li) => {
      const cks = ckptsOf(li);
      if (!cks || !cks.length) return;
      const name = esc(li.querySelector('code').textContent), id = 'ck-' + li.dataset.model;
      li.className = 'released';
      li.innerHTML = `<button class="model model-toggle" type="button" aria-expanded="false" aria-controls="${id}">
          ${hf}<code>${name}</code><em>${cks.length} ckpt${cks.length === 1 ? '' : 's'}</em><b>(${params(cks[0].params)})</b></button>
        <ul class="ckpts" id="${id}" hidden>${cks.map((c) => `
          <li><a class="model" href="${c.url}" target="_blank" rel="noopener" title="${esc(c.name)}">${hf}<code><i>${esc(c.name.split('.step')[0])}</i><span>.step${String(c.step).padStart(6, '0')}</span></code><b>(${params(c.params)})</b></a></li>`).join('')}</ul>`;
      const btn = li.querySelector('.model-toggle'), list = li.querySelector('.ckpts');
      btn.addEventListener('click', () => {
        const open = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', String(!open));
        list.hidden = open;
      });
    });
    // the note beside "explicit maps" names the same models, linked to the newest
    const tmpl = $('tip-maps');
    if (tmpl) tmpl.content.querySelectorAll('li[data-model]').forEach((li) => {
      const cks = ckptsOf(li);
      if (!cks || !cks.length) return;
      const name = esc(li.querySelector('code').textContent), c = cks[0];
      li.className = '';
      li.innerHTML = `<a class="model" href="${c.url}" target="_blank" rel="noopener" title="${esc(c.name)}">${hf}<code>${name}</code><b>(${params(c.params)})</b></a>`;
    });
  }

  // ---- references: an underlined phrase opens a note with a link to read more
  const tip = $('tip');
  if (tip) {
    const text = tip.querySelector('.tip-text'), link = tip.querySelector('.tip-link');
    const extra = tip.querySelector('.tip-extra');
    let current = null, hideT = null;
    const show = (el) => {
      clearTimeout(hideT);
      if (current && current !== el) current.classList.remove('is-open');
      current = el; el.classList.add('is-open');
      text.textContent = el.dataset.tip; link.href = el.href; link.textContent = el.dataset.doc;
      // a note may carry a list of its own -- the models an explicit map can be
      const list = el.dataset.tipList && document.getElementById(el.dataset.tipList);
      extra.innerHTML = list ? list.innerHTML : '';
      extra.hidden = !list;
      tip.classList.toggle('tip-wide', !!list);
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
    selTitle: $('sel-title'), selDoc: $('sel-doc'),
    inCol: $('inputs'), outCol: $('outputs'), closeBtn: $('sel-close'),
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
