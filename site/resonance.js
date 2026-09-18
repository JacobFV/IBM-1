/* ibm-1: the resonance tiles.
   eleven named loops -- five shown, six more behind the grid's "more" -- each
   drawn as the structures the substrate actually declares for it and lit as a
   wave travelling round the loop.

   two things this file is NOT allowed to imply, and both cost real time
   elsewhere on this page:

   1. the travelling rate on screen is NOT a measurement and is NOT the band.
      it is the band divided by a common SLOWDOWN, because a 13 Hz wave
      rendered at ~22 fps aliases into a stutter and reads as a different
      rhythm entirely.  between MIN_HZ and MAX_HZ the ratios between tiles are
      the ratios between the bands; outside it they are clamped -- gamma reads
      as the fastest but not twice beta, and the infraslow default-mode and
      arousal loops are sped up until they visibly move at all.  nothing on
      the page prints these numbers, and the section's closing line says so.
   2. ten of the eleven loops have never been put where they could fail.  the
      page no longer labels that on each tile (the user asked for the status
      lines to go), so it lives in three places instead: `data-status` on
      every <article>, the comment above the section in index.html, and
      docs/LOG.md.  the only numbers in the section are the two on the
      thalamo-cortical tile, held-out measurements against a declared prior,
      and no tile may gain one without the promotion criteria in that comment.

   the node sets below are real: every region name is an aparc/aseg label that
   exists in site/data/graph.js, checked against the graph's own inventory
   rather than typed from memory.  where a circuit's anatomy is finer than the
   substrate's (CA3 and CA1 are one `hippocampus` blob here; there is no
   subthalamic nucleus, no LGN or MGN apart from `thalamus`, no pons apart
   from `brainstem`) the nearest declared structure stands in, and the stage
   list says which. */
(function () {
  'use strict';
  const B = window.IBMBrain;
  if (!B || !B.registerSnaps) return;
  const { N, region, group, nodesWhere } = B;

  // a stage is a set of node indices; `reg` builds one from aparc/aseg labels.
  // it asserts the labels exist, because a typo here is silent -- the stage is
  // simply empty and the wave skips a beat that nobody can see is missing.
  const HAVE = new Set();
  for (let i = 0; i < N; i++) HAVE.add(region[i]);
  function reg() {
    const names = Array.prototype.slice.call(arguments);
    const missing = names.filter((n) => !HAVE.has(n));
    if (missing.length) console.warn('resonance: no such region in graph.js:', missing.join(', '));
    const s = new Set(names);
    return nodesWhere((i) => s.has(region[i]));
  }

  // rendered rate = band / SLOWDOWN, clamped.  see the header: depiction, not result.
  // MAX_HZ is set by the frame budget: with eleven tiles on screen brain.js
  // drops each to ~12 fps, and a four-stage wave much above 2.4 Hz then skips
  // whole stages between frames.
  const SLOWDOWN = 9, MIN_HZ = 0.2, MAX_HZ = 2.4, MAX_FAST = 3;
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);

  /* a travelling wave round an ordered ring of stages.
     pure in (w, t): it writes EVERY element of w from t alone, so calling it
     twice at the same t writes the same array -- the still frame served to
     prefers-reduced-motion and the moving one cannot disagree.  (the repo has
     paid twice for an instrument that answered differently on a second call.) */
  function loop(stages, opts) {
    const o = opts || {};
    const K = stages.length;
    const hz = clamp((o.band || 10) / SLOWDOWN, MIN_HZ, MAX_HZ);
    const width = o.width || 0.62 / K;      // pulse half-width, in cycles
    const floor = o.floor == null ? 0.055 : o.floor;   // the rest of the brain, drained
    const base = o.base == null ? 0.30 : o.base;       // the loop between pulses
    const fast = o.fast ? clamp(o.fast / SLOWDOWN, 0, MAX_FAST) : 0;   // a nested faster rhythm
    return function (w, t) {
      for (let i = 0; i < N; i++) {
        const g = group[i];
        w[i] = floor * (g === 'meg' ? 0.45 : g === 'eeg' ? 0.7 : 1);
      }
      const ph = t * hz;
      for (let k = 0; k < K; k++) {
        let u = (ph - k / K) % 1; if (u < 0) u += 1;
        const d = u < 0.5 ? u : 1 - u;          // wrapped distance from this stage's turn
        let g = Math.exp(-(d / width) * (d / width));
        if (fast) g *= 0.70 + 0.30 * Math.sin(2 * Math.PI * t * fast + k);
        const v = base + (1 - base) * g;
        const S = stages[k];
        for (let j = 0; j < S.length; j++) if (v > w[S[j]]) w[S[j]] = v;
      }
    };
  }

  const blank = () => new Float32Array(N);

  B.registerSnaps({
    // thalamus -> frontal -> parietal -> occipital -> thalamus.  the one loop
    // here that has been measured: scripts/fit_sleep_resonance.py fitted the
    // declared alpha_resonator to N2 spectra, split BY SUBJECT, and the centre
    // frequency moved off its 10 Hz prior to 13.45 Hz on held-out people.
    rz_tct: {
      az: 0.55, el: 0.28, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('thalamus'),
        reg('superiorfrontal', 'caudalmiddlefrontal'),
        reg('superiorparietal', 'precuneus'),
        reg('lateraloccipital', 'cuneus', 'pericalcarine'),
      ], { band: 13.45, base: 0.32 }),
    },

    // motor and premotor cortex -> striatum -> pallidum -> thalamus -> back.
    // the indirect arm's subthalamic nucleus is not a declared structure in
    // this graph, so the loop shown is the direct arm; pallidum stands for the
    // whole output nucleus (GPi and SNr are not separated here either).
    rz_bg: {
      az: -0.95, el: 0.34, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('precentral', 'caudalmiddlefrontal'),
        reg('putamen', 'caudate'),
        reg('pallidum'),
        reg('thalamus'),
      ], { band: 20, base: 0.28 }),
    },

    // medial temporal input -> hippocampus -> retrosplenial return.  `hippocampus`
    // is one aseg blob, so DG/CA3/CA1/subiculum cannot be staged separately, and
    // entorhinal is 8 nodes -- on its own it reads as a flicker rather than a
    // stage, so it shares the input stage with parahippocampal.  the fast nested
    // rhythm is the gamma each theta cycle carries.
    rz_hpc: {
      az: 1.1, el: -0.12, dist: 356, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('entorhinal', 'parahippocampal'),
        reg('hippocampus'),
        reg('isthmuscingulate', 'posteriorcingulate'),
      ], { band: 6.5, base: 0.26, fast: 45 }),
    },

    // occipito-temporal -> intraparietal -> frontal eye field.  IPS and FEF are
    // not aparc labels: superior/inferior parietal and caudal middle frontal
    // are the declared regions those two sit inside.
    rz_att: {
      az: -1.2, el: 0.24, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('lateraloccipital', 'fusiform'),
        reg('superiorparietal', 'inferiorparietal', 'supramarginal'),
        reg('caudalmiddlefrontal', 'precentral'),
      ], { band: 18, base: 0.26, width: 0.26 }),
    },

    // amygdala -> vmPFC and anterior cingulate -> accumbens -> hippocampus.
    // vmPFC is medial orbitofrontal here; the ventral tegmental area that
    // closes this loop in the literature is inside the `brainstem` blob and is
    // not separable, so it is left out rather than faked.
    rz_lim: {
      az: 0.5, el: -0.3, dist: 356, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('amygdala'),
        reg('medialorbitofrontal', 'rostralanteriorcingulate'),
        reg('accumbens'),
        reg('hippocampus', 'insula'),
      ], { band: 5, base: 0.26 }),
    },
  });

  // ---- the six behind "more".  all declared, none measured; same rules. ----
  B.registerSnaps({
    // motor cortex -> pons -> cerebellum -> thalamus (VL) -> motor cortex.  the
    // pontine nuclei are inside the `brainstem` blob, and VL is inside `thalamus`.
    rz_cbl: {
      az: -1.25, el: 0.12, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('precentral'),
        reg('brainstem'),
        reg('cerebellum'),
        reg('thalamus'),
      ], { band: 16, base: 0.26 }),
    },

    // posterior cingulate / precuneus hub -> medial prefrontal -> angular gyrus
    // (inside inferiorparietal) -> medial temporal.  a network, not a single
    // loop, so the "wave" is the order it is drawn in, not a claimed direction.
    // infraslow in life: MIN_HZ speeds it up until it moves at all.
    rz_dmn: {
      az: 0.95, el: 0.5, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('posteriorcingulate', 'isthmuscingulate', 'precuneus'),
        reg('medialorbitofrontal', 'rostralanteriorcingulate', 'superiorfrontal'),
        reg('inferiorparietal'),
        reg('parahippocampal', 'hippocampus'),
      ], { band: 1, base: 0.3, width: 0.3 }),
    },

    // cochlea -> brainstem (cochlear nucleus, olive, colliculus) -> MGN ->
    // Heschl's gyrus -> belt and parabelt.  MGN is inside `thalamus`; Heschl's
    // gyrus is aparc's transversetemporal.  gamma, so MAX_HZ clamps it.
    rz_aud: {
      az: -1.4, el: 0.06, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('cochlea'),
        reg('brainstem'),
        reg('thalamus'),
        reg('transversetemporal'),
        reg('superiortemporal', 'bankssts'),
      ], { band: 40, base: 0.26 }),
    },

    // brainstem arousal nuclei -> thalamus -> the cortex it gates, broadly.
    // the reticular formation, locus coeruleus and raphe are all inside the one
    // `brainstem` blob; the hypothalamus is not a declared structure here.
    rz_aro: {
      az: 0.95, el: 0.12, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('brainstem'),
        reg('thalamus'),
        reg('superiorfrontal', 'rostralmiddlefrontal', 'superiorparietal', 'precuneus', 'lateraloccipital'),
      ], { band: 1, base: 0.26, width: 0.28 }),
    },

    // retina -> LGN -> striate -> extrastriate.  the same pathway site.js's
    // l_visual figure draws; LGN is inside `thalamus`.
    rz_vis: {
      az: -1.15, el: 0.18, dist: 370, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('retina'),
        reg('thalamus'),
        reg('pericalcarine'),
        reg('cuneus', 'lingual'),
        reg('lateraloccipital'),
      ], { band: 10, base: 0.26 }),
    },

    // thalamus (VPL/VL) -> somatosensory -> motor -> medial motor strip.
    rz_mu: {
      az: -0.35, el: 0.95, dist: 360, aspect: 0.88, scalp: 0.24, weights: blank,
      pulse: loop([
        reg('thalamus'),
        reg('postcentral'),
        reg('precentral'),
        reg('paracentral'),
      ], { band: 11, base: 0.26 }),
    },
  });

  // ---- "more": the sixth grid cell --------------------------------------
  // the six extra tiles sit in a `hidden` wrapper that is display: contents
  // when shown, so they are grid items of the same grid.  the grid's own class
  // makes the tiles smaller.  every view's layout is stale after the re-flow
  // (all eleven change width, not just the new ones), so relayoutSnaps runs
  // on the next frame, after the browser has laid the grid out again.
  const btn = document.querySelector('.rz-more');
  const extra = btn && document.getElementById(btn.getAttribute('aria-controls'));
  const grid = btn && btn.closest('.rz-grid');
  if (btn && extra && grid) {
    const label = btn.querySelector('.rz-more-label');
    btn.addEventListener('click', () => {
      const open = btn.getAttribute('aria-expanded') !== 'true';
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      extra.hidden = !open;
      grid.classList.toggle('is-more', open);
      if (label) label.textContent = open ? 'Fewer' : 'More';
      requestAnimationFrame(() => { if (B.relayoutSnaps) B.relayoutSnaps(); });
      // closing shrinks the section by several screens; keep the reader at the
      // grid rather than stranding them in whatever section slid up under them
      if (!open) {
        const r = grid.getBoundingClientRect();
        if (r.top < 0) grid.scrollIntoView({ block: 'start', behavior: 'auto' });
      }
    });
  }
})();
