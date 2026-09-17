/* the spine.
   one trunk down the left that GROWS as you scroll: every station fuses a corpus in
   from the left and splinters a model out to the right, and the trunk is wider below
   each one than above it.  scrolling the page is meant to feel like one line of work
   getting steadily more capable, so width is cumulative and never decreases.

   it is a materialization spine, NOT a timeline.  the stations are models traced out
   of one declaration, which is why sleep, the body and stimulation can all sit on it;
   they have no position in training order and a chronological rail could not have
   carried them.

   past the last station that has actually been RUN the trunk continues as an open
   outline rather than a fill -- the same convention the lineage used for a thing that
   is declared rather than measured.  a declared station must never look like a result.

   geometry: the trunk's half-width is a sum of smoothsteps, one per station, so the
   widening is smooth everywhere and the edges leave and rejoin the vertical with
   matching slope.  everything is measured from the DOM after layout, so a section that
   grows or wraps cannot desynchronise the rail. */
(function () {
  const host = document.getElementById("spine");
  if (!host) return;
  const SVG = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(SVG, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const f = n => (+n).toFixed(1);

  const W = 150;          /* the rail's own user-unit width */
  const X = 62;           /* the trunk's centre line */
  const W0 = 7;           /* half-width above the first station */
  const STEP = 5.2;       /* half-width added by each station */
  const SPAN = 150;       /* vertical distance a widening is spread over */
  const SAMP = 5;         /* sampling step down the trunk, user units */

  const smooth = t => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

  let svg = null;
  function draw() {
    const secs = Array.from(document.querySelectorAll("[data-station]"));
    if (!secs.length) return;
    const H = host.offsetHeight;
    if (!H) return;                       /* hidden at narrow widths, by design */
    const top = host.getBoundingClientRect().top + window.scrollY;

    const st = secs.map(sec => ({
      y: sec.getBoundingClientRect().top + window.scrollY - top + 54,
      declared: sec.dataset.status === "declared",
      feed: sec.dataset.feed || "",
    }));

    /* half-width at y: W0 plus one smoothstep per station passed */
    const halfAt = y => {
      let h = W0;
      for (const s of st) h += STEP * smooth((y - (s.y - SPAN / 2)) / SPAN);
      return h;
    };
    /* where the run-and-measured part ends: just past the last station that ran */
    const lastRun = st.filter(s => !s.declared).pop();
    const cut = lastRun ? Math.min(H, lastRun.y + SPAN * 0.7) : 0;

    const edges = (y0, y1) => {
      const R = [], L = [];
      for (let y = y0; ; y += SAMP) {
        if (y > y1) y = y1;
        const h = halfAt(y);
        R.push([X + h, y]); L.push([X - h, y]);
        if (y >= y1) break;
      }
      return { R, L };
    };
    const poly = pts => pts.map(p => `L ${f(p[0])} ${f(p[1])}`).join(" ");

    if (svg) svg.remove();
    svg = el("svg", { class: "spine-svg", viewBox: `0 0 ${W} ${f(H)}`,
                      preserveAspectRatio: "none", "aria-hidden": "true" }, host);
    const defs = el("defs", {}, svg);
    const g = el("linearGradient", { id: "spine-fade", gradientUnits: "userSpaceOnUse",
                                     x1: 0, y1: 0, x2: 0, y2: f(H) }, defs);
    el("stop", { offset: 0, style: "stop-color:var(--accent);stop-opacity:0" }, g);
    el("stop", { offset: 0.05, style: "stop-color:var(--accent);stop-opacity:.5" }, g);
    el("stop", { offset: 1, style: "stop-color:var(--accent);stop-opacity:.72" }, g);

    /* the part that has run: filled */
    if (cut > 0) {
      const e = edges(0, cut);
      el("path", { class: "spine-trunk", fill: "url(#spine-fade)",
        d: `M ${f(X - W0)} 0 ${poly(e.R)} ${poly(e.L.slice().reverse())} Z` }, svg);
    }
    /* the part that is only declared: open outline, no fill */
    if (cut < H) {
      const e = edges(Math.max(0, cut - SAMP), H);
      el("path", { class: "spine-rim", d: `M ${f(e.R[0][0])} ${f(e.R[0][1])} ${poly(e.R)}` }, svg);
      el("path", { class: "spine-rim", d: `M ${f(e.L[0][0])} ${f(e.L[0][1])} ${poly(e.L)}` }, svg);
    }

    st.forEach(s => {
      const h = halfAt(s.y), cls = s.declared ? " is-declared" : "";
      /* a model splintering out to the right */
      el("path", { class: "spine-branch" + cls,
        d: `M ${f(X + h)} ${f(s.y)} C ${f(X + h + 34)} ${f(s.y)} ${f(X + h + 26)} ${f(s.y)} ${f(W)} ${f(s.y)}` }, svg);
      el("circle", { class: "spine-node" + cls, cx: f(X + h), cy: f(s.y), r: 4.6 }, svg);
      /* a corpus fusing in from the left, arriving just above the widening */
      if (s.feed)
        el("path", { class: "spine-feed" + cls,
          d: `M 0 ${f(s.y - 46)} C ${f(X - 40)} ${f(s.y - 46)} ${f(X - h - 16)} ${f(s.y - 10)} ${f(X - h)} ${f(s.y)}` }, svg);
    });

    host.classList.add("is-ready");
  }

  if (document.readyState === "complete") draw();
  else window.addEventListener("load", draw);
  let t;
  const again = () => { clearTimeout(t); t = setTimeout(draw, 140); };
  window.addEventListener("resize", again);
  if (window.ResizeObserver && host.parentElement) new ResizeObserver(again).observe(host.parentElement);
})();
