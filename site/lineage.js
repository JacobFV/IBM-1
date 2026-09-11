/* the lineage highway.
   every strand is ONE filled path: a centreline sampled densely, offset by half its
   width to each side, out along one edge and back along the other.  no strokes, no
   boxes -- a solid shape that veers, and that overlaps its neighbours where roads run
   together, which is what makes the trunk read as thick.
   annotations follow the brain viewer's language: a thin leader, a 2.2px dot where it
   lands, and a crisp label with no container. */
(function () {
  const D = window.IBM_LINEAGE, host = document.getElementById("lineage");
  if (!D || !host) return;
  const SVG = "http://www.w3.org/2000/svg";

  const W = 1000, H = 1700;                 /* user units; the svg scales to its box */
  const TRUNK = W / 2;
  const UNIT = 3.1;                         /* px of width per released checkpoint */
  const Y = s => 130 + D.stages[s].y * (H - 320);   /* headroom for the first and last labels */

  /* --- lanes: at each stage the live strands pack contiguously around the trunk --- */
  const live = s => D.strands.filter(t => t.from <= s && s <= t.to);
  function laneX(strand, s) {
    const act = live(s);
    const total = act.reduce((a, t) => a + t.w * UNIT, 0);
    let x = TRUNK - total / 2;
    for (const t of act) {
      const w = t.w * UNIT;
      if (t.id === strand.id) return x + w / 2;
      x += w;
    }
    return TRUNK;
  }

  /* a strand's centreline: its source, off to one side, then its lane at each stage */
  function centreline(t) {
    const pts = [];
    const s0 = t.from;
    pts.push([TRUNK + t.side * (W * 0.40), Y(s0) - 45]);      /* where the road comes in */
    for (let s = s0; s <= t.to; s++) pts.push([laneX(t, s), Y(s)]);
    pts.push([laneX(t, t.to), Y(t.to) + 40]);
    return pts;
  }

  /* Catmull-Rom through the control points, sampled -- smooth without hand-placed handles */
  function sample(pts, per = 26) {
    const P = [pts[0], ...pts, pts[pts.length - 1]], out = [];
    for (let i = 1; i < P.length - 2; i++) {
      const [p0, p1, p2, p3] = [P[i - 1], P[i], P[i + 1], P[i + 2]];
      for (let j = 0; j < per; j++) {
        const u = j / per, u2 = u * u, u3 = u2 * u;
        out.push([
          0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * u + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * u2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * u3),
          0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * u + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * u2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * u3)
        ]);
      }
    }
    out.push(pts[pts.length - 1]);
    return out;
  }

  /* one closed path: down the left edge, back up the right.  widthAt lets a road
     taper where it enters and where it ends. */
  function ribbon(line, halfWidth) {
    const L = [], R = [];
    for (let i = 0; i < line.length; i++) {
      const a = line[Math.max(0, i - 1)], b = line[Math.min(line.length - 1, i + 1)];
      let dx = b[0] - a[0], dy = b[1] - a[1];
      const m = Math.hypot(dx, dy) || 1; dx /= m; dy /= m;
      const h = halfWidth(i / (line.length - 1));
      L.push([line[i][0] - dy * h, line[i][1] + dx * h]);
      R.push([line[i][0] + dy * h, line[i][1] - dx * h]);
    }
    const seg = p => p.map((q, i) => (i ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1)).join(" ");
    return seg(L) + " " + seg(R.reverse()).replace(/^M/, "L") + " Z";
  }

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", "lin-svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "corpora joining a trunk, and the checkpoints that leave it");
  host.appendChild(svg);

  const gRoads = document.createElementNS(SVG, "g"); svg.appendChild(gRoads);
  const gAnn = document.createElementNS(SVG, "g"); svg.appendChild(gAnn);

  /* ---- the tributaries ---- */
  D.strands.forEach(t => {
    const line = sample(centreline(t));
    const half = t.w * UNIT / 2;
    const p = document.createElementNS(SVG, "path");
    /* taper in over the first fifth, and out over the last tenth */
    p.setAttribute("d", ribbon(line, u => half * Math.min(1, u / 0.18) * (u > 0.93 ? (1 - u) / 0.07 : 1)));
    p.setAttribute("class", "lin-road" + (t.future ? " is-future" : ""));
    gRoads.appendChild(p);
  });

  /* ---- the exits: a road leaving the trunk sideways ---- */
  D.exits.forEach(x => {
    if (!x.w) return;
    const y0 = Y(x.at), half = x.w * UNIT / 2;
    const line = sample([[TRUNK, y0 - 30], [TRUNK + x.side * 90, y0 + 20],
                         [TRUNK + x.side * (W * 0.34), y0 + 62]]);
    const p = document.createElementNS(SVG, "path");
    p.setAttribute("d", ribbon(line, u => half * (1 - 0.55 * u)));
    p.setAttribute("class", "lin-road lin-exit" + (x.future ? " is-future" : ""));
    gRoads.appendChild(p);
  });

  /* ---- annotations: leader, dot, label.  no container. ---- */
  function annotate(x, y, side, kindLabel, label, note, detail, href, future) {
    const g = document.createElementNS(SVG, "g");
    g.setAttribute("class", "lin-ann" + (future ? " is-future" : ""));
    const tx = TRUNK + side * (W * 0.455);
    const path = document.createElementNS(SVG, "path");
    path.setAttribute("class", "leader");
    path.setAttribute("d", `M ${x} ${y} C ${x + side * 60} ${y}, ${tx - side * 70} ${y - 14}, ${tx} ${y - 14}`);
    const dot = document.createElementNS(SVG, "circle");
    dot.setAttribute("class", "leader-dot"); dot.setAttribute("r", "3.2");
    dot.setAttribute("cx", x); dot.setAttribute("cy", y);
    const fo = document.createElementNS(SVG, "foreignObject");
    fo.setAttribute("x", side < 0 ? tx - 300 : tx); fo.setAttribute("y", y - 45);
    fo.setAttribute("width", "300"); fo.setAttribute("height", "100");
    const tag = href ? "a" : "div";
    fo.innerHTML =
      `<${tag} xmlns="http://www.w3.org/1999/xhtml" class="lin-ann-body ${side < 0 ? "to-left" : "to-right"}"` +
      (href ? ` href="${href}"${/^https?:/.test(href) ? ' target="_blank" rel="noopener"' : ""}` : "") + `>` +
        `<span class="lin-kind">${kindLabel}${future ? " · planned" : ""}</span>` +
        `<span class="lin-label">${label}</span>` +
        `<span class="lin-note">${note}</span>` +
        `<span class="lin-tip">${detail}</span>` +
      `</${tag}>`;
    g.appendChild(path); g.appendChild(dot); g.appendChild(fo);
    gAnn.appendChild(g);
  }

  D.strands.forEach(t => {
    const c = centreline(t);
    annotate(c[0][0], c[0][1], t.side, "dataset", t.label, t.note, t.detail, t.href, t.future);
  });
  D.exits.forEach(x => {
    const y = Y(x.at) + (x.w ? 62 : 0);
    const ex = TRUNK + x.side * (x.w ? W * 0.34 : 40);
    annotate(ex, y, x.side, x.kind === "control" ? "control" : "checkpoint",
             x.label, x.note, x.detail, x.href, x.future);
  });

  /* stage rules, faint, so the merges have something to happen against */
  D.stages.forEach((s, i) => {
    const y = Y(i);
    const l = document.createElementNS(SVG, "line");
    l.setAttribute("class", "lin-rule" + (i > D.now ? " is-future" : ""));
    l.setAttribute("x1", 40); l.setAttribute("x2", W - 40);
    l.setAttribute("y1", y); l.setAttribute("y2", y);
    gRoads.insertBefore(l, gRoads.firstChild);
    const t = document.createElementNS(SVG, "text");
    t.setAttribute("class", "lin-stage" + (i > D.now ? " is-future" : ""));
    t.setAttribute("x", 44); t.setAttribute("y", y - 8);
    t.textContent = s.label;
    gRoads.insertBefore(t, gRoads.firstChild);
  });

  host.classList.add("is-ready");
})();
