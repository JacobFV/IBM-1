/* the before/after band-power figure for the plasticity station.
   the curves are a 1/f^a aperiodic background with gaussian oscillatory peaks on top --
   the standard decomposition of a cortical spectrum -- evaluated here rather than traced
   by hand, so the axes, the peak frequencies and the band shading are all consistent with
   each other and with the alpha/beta bands they are labelled with.
   the shift drawn is a reorganisation of the kind a protocol targets: the aperiodic slope
   flattens slightly and the alpha peak moves up in frequency and amplitude. */
(function () {
  const host = document.getElementById("spectra");
  if (!host) return;
  const SVG = "http://www.w3.org/2000/svg";
  const el = (t, a, p) => { const e = document.createElementNS(SVG, t);
    for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const f = n => (+n).toFixed(2);

  const W = 520, H = 320, ML = 46, MR = 14, MT = 18, MB = 40;
  const F0 = 1, F1 = 45;                      /* Hz, log axis */
  const lx = hz => ML + (Math.log10(hz) - Math.log10(F0)) / (Math.log10(F1) - Math.log10(F0)) * (W - ML - MR);
  const P0 = -2.6, P1 = 1.25;                 /* log10 power */
  const ly = p => MT + (P1 - p) / (P1 - P0) * (H - MT - MB);

  /* log10 power = offset - slope*log10(f)  +  sum of gaussian peaks */
  const spectrum = (o, slope, peaks) => hz => {
    let p = o - slope * Math.log10(hz);
    let bump = 0;
    for (const [cf, amp, bw] of peaks) bump += amp * Math.exp(-((hz - cf) ** 2) / (2 * bw * bw));
    return Math.log10(10 ** p + bump);
  };
  const before = spectrum(0.92, 1.42, [[9.4, 0.46, 1.7], [21, 0.055, 3.4]]);
  const after  = spectrum(0.80, 1.22, [[10.9, 0.82, 1.9], [21, 0.075, 3.6]]);

  const path = fn => {
    let d = "";
    for (let i = 0; i <= 260; i++) {
      const hz = F0 * Math.pow(F1 / F0, i / 260);
      d += (i ? "L" : "M") + f(lx(hz)) + " " + f(ly(fn(hz)));
    }
    return d;
  };

  const svg = el("svg", { class: "spectra-svg", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "band power over the targeted region, before and after a protocol" }, host);

  /* the alpha band the shift is in */
  el("rect", { class: "sp-band", x: f(lx(8)), y: MT, width: f(lx(13) - lx(8)), height: H - MT - MB }, svg);
  el("text", { class: "sp-bandlabel", x: f((lx(8) + lx(13)) / 2), y: MT + 13 }, svg).textContent = "alpha";

  for (const hz of [1, 3, 10, 30]) {
    el("line", { class: "sp-grid", x1: f(lx(hz)), x2: f(lx(hz)), y1: MT, y2: H - MB }, svg);
    el("text", { class: "sp-tick", x: f(lx(hz)), y: H - MB + 15 }, svg).textContent = hz;
  }
  el("text", { class: "sp-axis", x: f((ML + W - MR) / 2), y: H - 8 }, svg).textContent = "frequency (Hz)";
  const yl = el("text", { class: "sp-axis", x: 0, y: 0,
    transform: `translate(13 ${(MT + H - MB) / 2}) rotate(-90)` }, svg);
  yl.textContent = "band power (log)";

  el("path", { class: "sp-before", d: path(before) }, svg);
  el("path", { class: "sp-after", d: path(after) }, svg);

  /* the peaks, marked where they actually sit on the curves */
  const mark = (fn, cf, cls) => el("circle", { class: cls, cx: f(lx(cf)), cy: f(ly(fn(cf))), r: 3.4 }, svg);
  mark(before, 9.4, "sp-dot-before");
  mark(after, 10.9, "sp-dot-after");

  const key = el("g", { class: "sp-key" }, svg);
  el("line", { class: "sp-before", x1: W - MR - 112, x2: W - MR - 92, y1: MT + 12, y2: MT + 12 }, key);
  el("text", { class: "sp-keytext", x: W - MR - 86, y: MT + 15.5 }, key).textContent = "before";
  el("line", { class: "sp-after", x1: W - MR - 112, x2: W - MR - 92, y1: MT + 30, y2: MT + 30 }, key);
  el("text", { class: "sp-keytext", x: W - MR - 86, y: MT + 33.5 }, key).textContent = "after protocol";

  host.classList.add("is-ready");
})();
