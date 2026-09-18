/* wires the opener, the article's figures and the registry lists. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');

  // ---- the name of a materialized model, shared by every figure that names one
  // (the spine rail, the materialization diagram).  a registry id is snake_case and is
  // NOT the model's name: `eeg_to_image` is the request, `IBM-1-EEG-to-Image` is the
  // thing you can download.  the two were being used interchangeably on the page.
  //
  // derived rather than tabulated, because a hand table goes stale the first time a new
  // materialization is added and nobody notices which of the two lists is authoritative.
  // ACRONYMS are the whole reason a naive title-case will not do -- "Eeg-To-Image".
  const ACRONYM = new Set(['av', 'eeg', 'meg', 'ecog', 'ieeg', 'lfp', 'csd', 'bci', 'tms',
    'tes', 'bold', 'fmri', 'emg', 'mep', 'ecg', 'qsm', 'dbs', 'ai']);
  const JOINER = new Set(['to', 'in', 'of', 'and', 'from', 'via']);
  window.IBM_MODEL_NAME = (id) => 'IBM-1-' + String(id).split(/[_-]/).map((w, i) =>
    ACRONYM.has(w) ? w.toUpperCase()
      : (i > 0 && JOINER.has(w)) ? w
      : w.charAt(0).toUpperCase() + w.slice(1)).join('-');

  // ---- the registry lists in §1
  const R = window.IBM_REGISTRY;
  if (R) {
    // the registry lists are OPTIONAL markup -- §1 is brief on the current page and
    // does not carry them.  this used to assume both nodes existed, and the resulting
    // throw took out the whole IIFE below it: the snapshots, the tooltips and the
    // lightbox all stopped silently, which reads exactly like a WebGL problem.
    const fill = (id, n, html) => {
      const c = $('n-' + id), l = $('list-' + id);
      if (c) c.textContent = n;
      if (l) l.innerHTML = html;
    };
    // each entry is one run of fine print: the id, what it is, then the rest in a sentence
    const row = (id, doc, rest) => `<p><code>${id}</code> ${esc(doc)}${rest ? ` <span>${rest}</span>` : ''}</p>`;
    fill('fields', `${R.fields.length} fields · ${R.fields.reduce((a, f) => a + f.components.length, 0)} components`,
      R.fields.map((f) => row(f.id, f.doc, f.components.map((c) => `<code>${c.id.replace(f.id + '.', '')}</code>${c.units ? ` [${esc(c.units)}]` : ''} ${esc(c.doc.replace(/\.$/, ''))}`).join('; '))).join(''));
    fill('anatomy', `${R.anatomy.length} systems · ${R.anatomy.reduce((a, s) => a + s.labels.length, 0)} labels`,
      R.anatomy.map((a) => row(a.id, a.doc, `${a.frame}, ${a.labels.length} labels: ${a.labels.map(esc).join(', ')}`)).join(''));
    fill('topologies', `${R.topologies.length} topologies`,
      R.topologies.map((t) => row(t.id, t.doc, `on ${t.on.join(', ')}${t.directed ? '; directed' : ''}`)).join(''));
    const ts = (s) => s >= 1 ? `${s} s` : s >= 1e-3 ? `${+(s * 1e3).toPrecision(3)} ms` : s >= 1e-6 ? `${+(s * 1e6).toPrecision(3)} µs` : `${+(s * 1e9).toPrecision(3)} ns`;
    fill('processes', `${R.processes.length} processes`,
      R.processes.map((p) => row(p.id, p.doc, [p.topology, p.timescale_s && ts(p.timescale_s), p.inputs.length && `in ${p.inputs.map((v) => `<code>${v}</code>`).join(' ')}`, p.outputs.length && `out ${p.outputs.map((v) => `<code>${v}</code>`).join(' ')}`].filter(Boolean).join(' · '))).join(''));
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
  window.IBM_PARAMS = params;
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

  // ---- the release figures on the page come from data/releases.js, which
  // scripts/export_releases.py writes from the hub.  the kernel path and the
  // checkpoint count were hand-typed here once and went stale immediately --
  // the implicit/ directory was not even exported, so the page had no link to
  // the one artifact its central claim is about.
  if (REL) {
    const put = (key, fn) => document.querySelectorAll(`[data-rel="${key}"]`).forEach(fn);
    const nck = Object.keys(REL.models || {}).reduce((a, m) => a + REL.models[m].length, 0);
    const k = (REL.kernels || [])[0];
    if (REL.repo) put('repo', (e) => { e.textContent = REL.repo; });
    if (k) {
      put('kernel-path', (e) => { e.textContent = k.path; });
      put('kernel-href', (e) => { e.href = k.url; });
      put('kernel-label', (e) => {
        e.textContent = k.params ? `The Implicit Kernel (${params(k.params)})` : 'The Implicit Kernel';
      });
    }
    if (nck) put('ckpt-label', (e) => { e.textContent = `${nck} Task Checkpoints`; });
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
    materialize: { model: 'eeg_forward', az: -0.5, el: 0.28, dist: 440, aspect: 0.62, mono: 0.86, annotations: () => A.model_io('eeg_forward') },
    l_forward: { model: 'eeg_forward', az: -0.6, el: 0.3, dist: 470, aspect: 0.8 },
    l_decode: { model: 'meg_to_text', az: -1.3, el: 0.2, dist: 470, aspect: 0.8 },
    l_stim: { model: 'tms_response', az: -0.9, el: 0.55, dist: 470, aspect: 0.8 },
    l_body: { model: 'invasive_bci', az: -0.8, el: 0.45, dist: 470, aspect: 0.8 },
    l_surrogate: { model: 'macro_surrogate', az: -0.6, el: 0.3, dist: 470, aspect: 0.8 },
    // the visual pathway as the substrate actually declares it: 20 retinal sites,
    // the occipital port they arrive at, and the 60 digitised contacts the result is
    // read out at.  drawn from the graph rather than illustrated, so what is lit is
    // what the model has -- including that the port is a REGION, not a point.
    l_visual: {
      az: -1.15, el: 0.18, dist: 445, aspect: 0.92,
      weights: () => anatomyWeights([
        (i) => group[i] === 'retina',
        (i) => B.region[i] === 'lateraloccipital',
        (i) => B.region[i] === 'pericalcarine',
        (i) => B.region[i] === 'lingual',
        (i) => B.region[i] === 'cuneus',
        (i) => group[i] === 'eeg',
      ]),
      annotations: () => finish([
        A.group('retina', 'retina', { sub: 'three populations at 2.5 / 4.2 / 8.3 ms', side: 'l' }),
        A.region('occipital port', 'lateraloccipital', null, { sub: 'displaced, retrieval falls to 0.44%', side: 'l' }),
        A.region('striate cortex', 'pericalcarine', null, { sub: 'pericalcarine · lingual · cuneus', side: 'l' }),
        A.group('64-channel EEG', 'eeg', { sub: '60 digitised contacts', side: 'r' }),
      ]),
    },
  });
})();

/* the reel: click a tile, it fills the screen with its caption and description.
   once open, a horizontal swipe, the arrow keys or the side buttons step
   through every tile on the page in order, across piles, wrapping at the ends */
(function () {
  function init() {
  var lb = document.getElementById('lightbox');
  if (!lb) return;
  var stage = lb.querySelector('.lb-stage'),
      h3 = lb.querySelector('h3'),
      cap = lb.querySelector('.lb-cap'),
      desc = lb.querySelector('.lb-desc'),
      count = lb.querySelector('.lb-count');
  // the fanned reel band is gone; station media carry plain .tile elements, so the
  // lightbox collects both rather than only the ones under a .reel ancestor
  var tiles = Array.prototype.slice.call(document.querySelectorAll('.reel .tile, .st-media .tile'));
  var cur = -1;

  // within a pile the first tile sits on top and each later one behind it
  document.querySelectorAll('.pile').forEach(function (p) {
    var ts = p.querySelectorAll('.tile');
    ts.forEach(function (t, i) { t.style.setProperty('--z', String(ts.length - i)); });
  });

  function open(i) {
    cur = (i + tiles.length) % tiles.length;
    var t = tiles[cur];
    var kind = t.dataset.kind, src = t.dataset.src;
    // a <video> keeps playing behind a hidden lightbox otherwise, so it is
    // rebuilt each time rather than reused
    stage.innerHTML = '';
    var el;
    if (kind === 'video') {
      el = document.createElement('video');
      el.src = src; el.autoplay = true; el.loop = true;
      el.muted = true; el.playsInline = true; el.controls = true;
    } else {
      el = document.createElement('img'); el.src = src; el.alt = t.dataset.title;
    }
    stage.appendChild(el);
    h3.textContent = t.dataset.title;
    cap.textContent = t.dataset.cap;
    desc.textContent = t.dataset.desc;
    count.textContent = (cur + 1) + ' / ' + tiles.length;
    if (lb.hidden) {
      lb.hidden = false;
      document.body.style.overflow = 'hidden';
      lb.querySelector('.lb-close').focus();
    }
  }
  function step(d) { if (!lb.hidden) open(cur + d); }
  function close() {
    lb.hidden = true; stage.innerHTML = '';
    document.body.style.overflow = '';
    if (tiles[cur]) tiles[cur].focus({ preventScroll: true });
  }
  tiles.forEach(function (t, i) {
    t.addEventListener('click', function () { open(i); });
    t.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(i); }
    });
  });
  lb.addEventListener('click', function (e) {
    if (e.target.closest('.lb-prev')) return step(-1);
    if (e.target.closest('.lb-next')) return step(1);
    if (e.target === lb || e.target.classList.contains('lb-close')) close();
  });
  document.addEventListener('keydown', function (e) {
    if (lb.hidden) return;
    if (e.key === 'Escape') close();
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
  });
  // a mostly horizontal swipe steps; a vertical one (scrolling a long
  // description) is left alone, and so is a drag along a video's control bar
  var sx = null, sy = 0;
  lb.addEventListener('touchstart', function (e) {
    sx = null;
    if (e.touches.length !== 1) return;
    var p = e.touches[0], tg = e.target;
    if (tg.tagName === 'VIDEO' && p.clientY > tg.getBoundingClientRect().bottom - 56) return;
    sx = p.clientX; sy = p.clientY;
  }, { passive: true });
  lb.addEventListener('touchend', function (e) {
    if (sx == null) return;
    var p = e.changedTouches[0], dx = p.clientX - sx, dy = p.clientY - sy;
    sx = null;
    if (Math.abs(dx) > 45 && Math.abs(dx) > 1.3 * Math.abs(dy)) step(dx < 0 ? 1 : -1);
  }, { passive: true });
  }
  // the script tag sat BEFORE the lightbox markup, so getElementById returned
  // null and every tile silently did nothing.  bind after the document parses
  // regardless of where this file is included.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else { init(); }
})();
