/* the lineage bundle.
   a vertical trunk of fixed lanes.  every change of lane is a VEER: a cubic whose
   tangent is vertical where it leaves and vertical where it lands, so no road ever
   runs sideways or turns a corner.
   a corpus veers in from its side and keeps the lane it lands in for life; later
   arrivals stack on the OUTSIDE of their side, so nothing already in the trunk moves.
   each corpus has its own colour and blends into the trunk colour across the veer
   where two lanes first come together, so merged lanes are one colour, no seam.
   a checkpoint veers out of the trunk's outer edge.
   labels sit beside the road they belong to; stage names sit at the left end of
   their rule.  a checkpoint's label is the word, its title, and one 🤗 link per
   released model with its size -- nothing else.  widths, links and sizes for
   checkpoints come from releases.js, never from hand-typed figures. */
(function () {
  const D = window.IBM_LINEAGE, host = document.getElementById("lineage");
  if (!D || !host) return;
  const REL = (window.IBM_RELEASES && window.IBM_RELEASES.models) || {};
  const SVG = "http://www.w3.org/2000/svg";

  const W = 1000;                           /* user units; the svg scales to its box */
  const UNIT = 2.0;                         /* px of width per released checkpoint */
  const GAP = 36;                           /* clear space between a veering road and the trunk */
  const STUB = 20;                          /* straight run before a veer in, after a veer out */
  const SEAM = 0.6;                         /* adjacent lanes overlap by this much */
  const TOP = 210, SPAN = 1450;
  const Y = s => TOP + D.stages[s].y * SPAN;
  const H = Y(D.stages.length - 1) + 180;
  const TRUNK = "var(--accent)";
  const hue = i => `var(--lane-${(i % 4) + 1})`;
  /* tall veers.  1.8 puts the steepest point at 48deg from vertical (not the ~40 this
     comment used to claim), which rails() then holds at constant width to within 9%.
     raising it flattens the veer further -- 2.2 would get that to 4.6% -- but the top
     corpus's veer already starts near y=0, so a taller one lifts its label off the canvas. */
  const rise = dx => Math.max(72, Math.abs(dx) * 1.8);
  const f = n => (+n).toFixed(1);
  const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(SVG, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  /* vertical at (xa,ya), vertical at (xb,yb).  used for the hairline leaders, where a
     stroke is already perpendicular to its path and nothing can pinch. */
  const veer = (xa, ya, xb, yb) => {
    const ym = (ya + yb) / 2;
    return `C ${f(xa)} ${f(ym)} ${f(xb)} ${f(ym)} ${f(xb)} ${f(yb)}`;
  };

  /* ---- a ribbon of CONSTANT PERPENDICULAR WIDTH along a veer ----
     a filled road cannot be drawn as two copies of the same cubic offset sideways by w.
     that keeps the HORIZONTAL width constant, and the width a reader sees is the one
     across the road, w*cos(theta) -- so every veer pinched to 66.9% of its lane at the
     steepest point (48deg from vertical, not the ~40 the rise comment claimed) and
     swelled back at both ends.  width here means "checkpoints released", so a road that
     narrows through a bend is telling the reader something untrue.
     instead: walk the CENTRELINE cubic and widen it by SEC(theta) as it slants, so the
     across-the-road width stays w.  two points a horizontal w/cos(theta) apart are exactly
     w apart measured across the road, which is the width the reader sees.
     the sideways offset stays HORIZONTAL rather than along the normal.  offsetting along
     the normal gives the same constant width but moves each edge VERTICALLY by
     (w/2)sin(theta), which grows linearly out of the join and creases the outer edge where
     the veer leaves the straight stub; sec(theta) grows as theta^2, so it leaves the
     straight run with matching width AND matching slope, and the join is invisible.
     the road bulges to w/cos(theta) horizontally through the bend, which GAP already clears. */
  const STEPS = 40;
  function rails(xa, ya, xb, yb, w) {
    const ym = (ya + yb) / 2, h = w / 2, right = [], left = [];
    for (let i = 0; i <= STEPS; i++) {
      const t = i / STEPS, u = 1 - t;
      const x = xa * u * u * u + 3 * xa * u * u * t + 3 * xb * u * t * t + xb * t * t * t;
      const y = ya * u * u * u + 3 * ym * u * u * t + 3 * ym * u * t * t + yb * t * t * t;
      const dx = 6 * (xb - xa) * u * t, dy = 1.5 * (yb - ya) * (u * u + t * t);
      const sec = Math.hypot(dx, dy) / Math.abs(dy);      /* dy is never 0: yb != ya */
      right.push([x + h * sec, y]);
      left.push([x - h * sec, y]);
    }
    return { right, left };
  }
  const trace = pts => pts.map(p => `L ${f(p[0])} ${f(p[1])}`).join(" ");
  const back = pts => trace(pts.slice().reverse());

  /* a checkpoint is as wide as what it released */
  const releases = x => (x.models || []).map(m => ({ model: m, list: REL[m] || [] }));
  const widthOf = x => (x.models && x.models.length)
    ? releases(x).reduce((a, r) => a + r.list.length, 0) : (x.w || 0);
  const size = p => p >= 1e9 ? (p / 1e9).toFixed(1) + "B" : p >= 1e6 ? (p / 1e6).toFixed(1) + "M" : Math.round(p / 1e3) + "K";

  /* --- lanes: fixed for the life of a strand.  each side stacks outward in entry order --- */
  const bySide = side => D.strands.filter(t => t.side === side).sort((a, b) => a.from - b.from);
  const extent = side => bySide(side).reduce((a, t) => a + t.w * UNIT, 0);
  const SPINE = W / 2 + (extent(-1) - extent(1)) / 2;   /* so the full trunk sits centred */
  const lane = {};
  for (const side of [-1, 1]) {
    let off = 0;
    for (const t of bySide(side)) {
      const w = t.w * UNIT;
      lane[t.id] = side < 0 ? { xl: SPINE - off - w, xr: SPINE - off } : { xl: SPINE + off, xr: SPINE + off + w };
      off += w;
    }
  }
  const live = s => D.strands.filter(t => t.from <= s && s <= t.to);
  const edge = (s, side) => {
    const xs = live(s).filter(t => t.side === side).map(t => side < 0 ? lane[t.id].xl : lane[t.id].xr);
    return xs.length ? (side < 0 ? Math.min(...xs) : Math.max(...xs)) : SPINE;
  };

  /* where each corpus's veer runs */
  const arrive = {};
  for (const t of D.strands) {
    const off = t.side * ((lane[t.id].xr - lane[t.id].xl) + GAP);
    const yM = Y(t.from) - 12, yC = yM - rise(off);
    arrive[t.id] = { off, yM, yC, yS: yC - STUB };
  }
  /* the span over which a corpus takes on the trunk colour: the veer of whichever
     strand first shares the trunk with it -- its own, if it is the one arriving.
     both lanes of a merge therefore change colour over the same rows. */
  function blend(t) {
    let best = null;
    for (const o of D.strands) {
      if (o === t) continue;
      const a = Math.max(t.from, o.from);
      if (a > Math.min(t.to, o.to)) continue;
      if (!best || a < best.a) best = { a, by: t.from === a ? t : o };
    }
    return best && arrive[best.by.id];
  }

  const svg = el("svg", { class: "lin-svg", viewBox: `0 0 ${W} ${f(H)}`, role: "img",
                          "aria-label": "corpora joining a trunk, and the checkpoints that leave it" }, host);
  const defs = el("defs", {}, svg);
  const gRules = el("g", { class: "lin-rules" }, svg);
  const gRoads = el("g", { class: "lin-roads" }, svg);     /* one group opacity: no seams, no darker overlaps */
  const gPlanned = el("g", { class: "lin-planned" }, svg);
  const gAnn = el("g", { class: "lin-anns" }, svg);

  function paint(t, i) {
    const b = blend(t), c = hue(i);
    if (!b) return c;
    const id = "lin-blend-" + t.id;
    const g = el("linearGradient", { id, gradientUnits: "userSpaceOnUse", x1: 0, x2: 0, y1: f(b.yC), y2: f(b.yM) }, defs);
    el("stop", { offset: 0, style: `stop-color:${c}` }, g);
    el("stop", { offset: 1, style: `stop-color:${TRUNK}` }, g);
    return `url(#${id})`;
  }

  /* ---- a corpus: a short straight run, a veer into its lane, then straight down ---- */
  function entry(t, i) {
    const { xl, xr } = lane[t.id], side = t.side, { off, yM, yC, yS } = arrive[t.id];
    const L = xl - (side > 0 ? SEAM : 0), R = xr + (side < 0 ? SEAM : 0);
    const yEnd = Y(t.to) + 40, w = R - L, cIn = L + off + w / 2, cLane = L + w / 2;
    const { right, left } = rails(cIn, yC, cLane, yM, w);   /* down the page, into the lane */
    const d =
      `M ${f(L + off)} ${f(yS)} L ${f(R + off)} ${f(yS)} L ${f(R + off)} ${f(yC)} ${trace(right)} ` +
      `L ${f(R)} ${f(yEnd)} L ${f(L)} ${f(yEnd)} L ${f(L)} ${f(yM)} ${back(left)} Z`;
    const p = el("path", { d, class: "lin-road" + (t.future ? " is-future" : "") }, t.future ? gPlanned : gRoads);
    p.style[t.future ? "stroke" : "fill"] = paint(t, i);
    return { edge: side < 0 ? L + off : R + off, y: yS + 10 };
  }

  /* ---- a checkpoint: a veer out of the trunk's outer edge, then a short straight run.
     the path is left open along the trunk, so a planned outline draws no line across it ---- */
  function exit(x) {
    const side = x.side, w = widthOf(x) * UNIT, e = edge(x.at, side), yA = Y(x.at) + 12;
    const off = side * (w + GAP), yB = yA + rise(off), yE = yB + STUB;
    if (!w) {                                           /* a control has no width: a leader */
      el("path", { class: "lin-leader" + (x.future ? " is-future" : ""),
                   d: `M ${f(e)} ${f(yA)} ${veer(e, yA, e + off, yB)} L ${f(e + off)} ${f(yE)}` }, gAnn);
      el("circle", { class: "lin-dot", r: 3.2, cx: f(e + off), cy: f(yE) }, gAnn);
      return { edge: e + off, y: yE };
    }
    const L = side < 0 ? e : e - w, R = L + w;
    const { right, left } = rails(L + w / 2, yA, L + off + w / 2, yB, w);   /* out of the trunk */
    const d =
      `M ${f(R)} ${f(yA)} ${trace(right)} L ${f(R + off)} ${f(yE)} ` +
      `L ${f(L + off)} ${f(yE)} L ${f(L + off)} ${f(yB)} ${back(left)}`;
    const p = el("path", { d, class: "lin-road" + (x.future ? " is-future" : "") }, x.future ? gPlanned : gRoads);
    p.style[x.future ? "stroke" : "fill"] = TRUNK;
    return { edge: side < 0 ? L + off : R + off, y: yE - 10 };
  }

  /* ---- labels: beside the road, no container ---- */
  function label(pt, side, future, body) {
    const g = el("g", { class: "lin-ann" + (future ? " is-future" : "") }, gAnn);
    const FH = 160, PAD = 12;
    const x = side < 0 ? 0 : pt.edge + PAD, width = side < 0 ? pt.edge - PAD : W - pt.edge - PAD;
    const fo = el("foreignObject", { x: f(x), y: f(pt.y - FH / 2), width: f(width), height: FH }, g);
    fo.innerHTML = body(side < 0 ? "to-left" : "to-right");
  }
  const linkAttrs = href => href
    ? ` href="${esc(href)}"${/^https?:/.test(href) ? ' target="_blank" rel="noopener"' : ""}` : "";
  const described = (kind, t) => cls => {
    const tag = t.href ? "a" : "div";
    return `<${tag} xmlns="http://www.w3.org/1999/xhtml" class="lin-ann-body ${cls}"${linkAttrs(t.href)}>` +
      `<span class="lin-kind">${kind}${t.future ? " · planned" : ""}</span>` +
      `<span class="lin-label">${esc(t.label)}</span>` +
      `<span class="lin-note">${esc(t.note)}</span>` +
      `<span class="lin-tip">${esc(t.detail)}</span></${tag}>`;
  };
  const checkpoint = x => cls =>
    `<div xmlns="http://www.w3.org/1999/xhtml" class="lin-ann-body ${cls}">` +
      `<span class="lin-kind">checkpoint</span>` +
      `<span class="lin-label">${esc(x.label)}</span>` +
      releases(x).filter(r => r.list.length).map(r =>
        `<a class="lin-hf"${linkAttrs(r.list[0].url)}><span class="hf" aria-hidden="true">🤗</span> ` +
        `<code>${esc(r.model)}</code> (${size(r.list[0].params)})</a>`).join("") +
    `</div>`;

  /* stage rules first, so the merges have something to happen against */
  D.stages.forEach((s, i) => {
    const y = Y(i), fut = i > D.now ? " is-future" : "";
    el("line", { class: "lin-rule" + fut, x1: 0, x2: W, y1: f(y), y2: f(y) }, gRules);
    el("text", { class: "lin-stage" + fut, x: 0, y: f(y - 8) }, gRules).textContent = s.label;
  });

  D.strands.forEach((t, i) => label(entry(t, i), t.side, t.future, described("dataset", t)));
  D.exits.forEach(x => label(exit(x), x.side, x.future,
                             x.kind === "control" ? described("control", x) : checkpoint(x)));

  host.classList.add("is-ready");
})();
