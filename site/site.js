/* wires the opener and the article's figures to the brain renderer. */
(function () {
  'use strict';
  const B = window.IBMBrain;
  if (!B) return;
  const $ = (id) => document.getElementById(id);

  B.createHero({
    hero: $('hero'), canvas: $('hero-canvas'), svg: $('hero-leaders'), ring: $('ring'),
    selTitle: $('sel-title'), selGroup: $('sel-group'), selDoc: $('sel-doc'), selMeta: $('sel-meta'),
    inCol: $('inputs'), outCol: $('outputs'), closeBtn: $('sel-close'), hint: $('hero-hint'),
  });

  const { A, finish, weightsFor, byId, nodesWhere, group, rest, N } = B;

  // a highlight of a few anatomical parts on the resting substrate
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
        A.group('spinal stub', 'spinal', { sub: 'body support · drawn for orientation', side: 'r' }),
      ]),
    },
    materialize: { model: 'eeg_forward', az: -0.5, el: 0.28, dist: 440, aspect: 0.62, annotations: () => A.model_io('eeg_forward') },
    meg: { model: 'meg_to_text', az: -1.35, el: 0.18, dist: 430, aspect: 0.62, annotations: () => A.model_io('meg_to_text') },
    sleep: { model: 'sleep_dynamics', az: 0.6, el: 0.35, dist: 440, aspect: 0.62, annotations: () => A.model_io('sleep_dynamics').filter((a) => ['eeg', 'polysomnography', 'neural.exc.activity', 'extracellular.acetylcholine'].includes(a.sub)) },
    ablation: {
      az: -0.7, el: 0.3, dist: 430, aspect: 0.62,
      weights: () => { const w = new Float32Array(N); for (let i = 0; i < N; i++) w[i] = group[i] === 'cortex' ? 1 : (group[i] === 'retina' || group[i] === 'cochlea' ? 1 : rest[i] * 0.3); return w; },
      annotations: () => finish([
        A.group('retina, forced by frames', 'retina', { sub: 'transduction.photoreceptor', side: 'l', cls: 'in' }),
        A.group('cochlea, forced by the soundtrack', 'cochlea', { sub: 'transduction.hair_cell', side: 'l', cls: 'in' }),
        A.group('the shared cortical dynamics', 'cortex', { sub: 'bypass: +324% loss', side: 'r', cls: 'out' }),
      ]),
    },
    // the lineup
    l_forward: { model: 'eeg_forward', az: -0.6, el: 0.3, dist: 470, aspect: 0.8, scalp: 1 },
    l_decode: { model: 'meg_to_text', az: -1.3, el: 0.2, dist: 470, aspect: 0.8 },
    l_stim: { model: 'tms_response', az: -0.9, el: 0.55, dist: 470, aspect: 0.8 },
    l_state: { model: 'sleep_dynamics', az: 0.5, el: 0.3, dist: 470, aspect: 0.8 },
    l_body: { model: 'invasive_bci', az: -0.8, el: 0.45, dist: 470, aspect: 0.8 },
    l_surrogate: { model: 'macro_surrogate', az: -0.6, el: 0.3, dist: 470, aspect: 0.8 },
  });
})();
