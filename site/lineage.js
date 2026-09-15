/* the lineage bundle.
   a straight vertical trunk of fixed lanes.  a corpus enters from its side with one
   rounded turn and then runs straight down in a lane that is assigned once and
   never moves -- later arrivals stack on the OUTSIDE of their side, so nothing
   already in the trunk has to shift.  a checkpoint peels off the trunk's outer
   edge as a plain sankey link.  no splines through every stage, no re-packing,
   nothing crosses anything.
   labels live in two reserved columns, one each side, so nothing is clipped.
   annotations keep the brain viewer's language: a dot where the road lands, and a
   bare label with no container. */
(function () {
  const D = window.IBM_LINEAGE, host = document.getElementById("lineage");
  if (!D || !host) return;
  const SVG = "http://www.w3.org/2000/svg";

  const W = 1000;                           /* user units; the svg scales to its box */
  const LC = 230;                           /* label column on each side */
  const UNIT = 3.1;                         /* px of width per released checkpoint */
  const R0 = 36;                            /* inner radius of an entry's turn */
  const Y = s => 100 + D.stages[s].y * 1150;
  const H = Y(D.stages.length - 1) + 190;

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
  const f = n => n.toFixed(1);

  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", "lin-svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "corpora joining a trunk, and the checkpoints that leave it");
  host.appendChild(svg);

  /* solid roads share one group opacity, so lanes that touch do not show a seam;
     planned roads are open outlines and sit in their own group at full opacity */
  const grp = cls => { const g = document.createElementNS(SVG, "g"); g.setAttribute("class", cls); svg.appendChild(g); return g; };
  const gRules = grp("lin-rules");
  const gExits = grp("lin-exits"), gExitsPlanned = grp("lin-planned");
  const gRoads = grp("lin-roads"), gRoadsPlanned = grp("lin-planned");
  const gAnn = grp("lin-anns");

  /* ---- a tributary: horizontal from its label, one turn, then straight down ----
     the band's bottom sits R0 above the stage rule, so the lane is vertical by the
     time it reaches the rule: that is the stage it joins at. */
  function entry(t) {
    const { xl, xr } = lane[t.id], w = xr - xl, side = t.side;
    const xs = side < 0 ? LC : W - LC;                 /* where the road comes in */
    const yb = Y(t.from) - 50, ya = yb - w;            /* band top and bottom */
    const yEnd = Y(t.to) + 40;
    const outer = side < 0 ? xr : xl, inner = side < 0 ? xl : xr;
    const Ro = R0 + w;                                 /* concentric with the inner turn */
    const yTurn = yb + R0;                             /* == ya + Ro */
    /* seams: overlap the shared edge with the neighbouring lane by a pixel */
    const seam = 1;
    const d =
      `M ${f(xs)} ${f(ya)} L ${f(outer + side * Ro)} ${f(ya)} ` +
      `C ${f(outer)} ${f(ya)} ${f(outer)} ${f(ya)} ${f(outer)} ${f(yTurn)} ` +
      `L ${f(outer)} ${f(yEnd)} L ${f(inner - side * seam)} ${f(yEnd)} L ${f(inner - side * seam)} ${f(yTurn)} ` +
      `C ${f(inner)} ${f(yb)} ${f(inner)} ${f(yb)} ${f(inner + side * R0)} ${f(yb)} ` +
      `L ${f(xs)} ${f(yb)} Z`;
    const p = document.createElementNS(SVG, "path");
    p.setAttribute("d", d);
    p.setAttribute("class", "lin-road" + (t.future ? " is-future" : ""));
    (t.future ? gRoadsPlanned : gRoads).appendChild(p);
    return { x: xs, y: (ya + yb) / 2 };
  }

  /* ---- an exit: a sankey link leaving the trunk's outer edge on its side ---- */
  function exit(x) {
    const side = x.side, w = x.w * UNIT;
    const x0 = edge(x.at, side), xs = side < 0 ? LC : W - LC;
    const y0 = Y(x.at) + 14;
    const ye = y0 + w / 2 + 10;                         /* the road drops a little as it leaves */
    if (!x.w) return { x: x0, y: y0, leader: true };
    const xm = (x0 + xs) / 2;
    const d =
      `M ${f(x0)} ${f(y0)} C ${f(xm)} ${f(y0)} ${f(xm)} ${f(ye - w / 2)} ${f(xs)} ${f(ye - w / 2)} ` +
      `L ${f(xs)} ${f(ye + w / 2)} C ${f(xm)} ${f(ye + w / 2)} ${f(xm)} ${f(y0 + w)} ${f(x0)} ${f(y0 + w)} Z`;
    const p = document.createElementNS(SVG, "path");
    p.setAttribute("d", d);
    p.setAttribute("class", "lin-road lin-exit" + (x.future ? " is-future" : ""));
    (x.future ? gExitsPlanned : gExits).appendChild(p);
    return { x: xs, y: ye };
  }

  /* ---- annotations: dot where the road lands, label in the column.  no container. ---- */
  function annotate(pt, side, kindLabel, label, note, detail, href, future) {
    const g = document.createElementNS(SVG, "g");
    g.setAttribute("class", "lin-ann" + (future ? " is-future" : ""));
    let { x, y } = pt;
    if (pt.leader) {                                    /* a control has no width: lead to it */
      const tx = side < 0 ? LC : W - LC;
      const path = document.createElementNS(SVG, "path");
      path.setAttribute("class", "leader");
      path.setAttribute("d", `M ${f(x)} ${f(y)} C ${f(x + side * 60)} ${f(y)}, ${f(tx - side * 60)} ${f(y + 30)}, ${f(tx)} ${f(y + 30)}`);
      g.appendChild(path);
      const dot = document.createElementNS(SVG, "circle");
      dot.setAttribute("class", "leader-dot"); dot.setAttribute("r", "3.2");
      dot.setAttribute("cx", f(x)); dot.setAttribute("cy", f(y));
      g.appendChild(dot);
      x = tx; y = y + 30;
    }
    const dot = document.createElementNS(SVG, "circle");
    dot.setAttribute("class", "leader-dot"); dot.setAttribute("r", "3.2");
    dot.setAttribute("cx", f(x)); dot.setAttribute("cy", f(y));
    const fo = document.createElementNS(SVG, "foreignObject");
    const FH = 120;
    fo.setAttribute("x", side < 0 ? 0 : W - LC); fo.setAttribute("y", f(y - FH / 2));
    fo.setAttribute("width", LC); fo.setAttribute("height", FH);
    const tag = href ? "a" : "div";
    fo.innerHTML =
      `<${tag} xmlns="http://www.w3.org/1999/xhtml" class="lin-ann-body ${side < 0 ? "to-left" : "to-right"}"` +
      (href ? ` href="${href}"${/^https?:/.test(href) ? ' target="_blank" rel="noopener"' : ""}` : "") + `>` +
        `<span class="lin-kind">${kindLabel}${future ? " · planned" : ""}</span>` +
        `<span class="lin-label">${label}</span>` +
        `<span class="lin-note">${note}</span>` +
        `<span class="lin-tip">${detail}</span>` +
      `</${tag}>`;
    g.appendChild(dot); g.appendChild(fo);
    gAnn.appendChild(g);
  }

  /* stage rules first, so the merges have something to happen against */
  D.stages.forEach((s, i) => {
    const y = Y(i);
    const l = document.createElementNS(SVG, "line");
    l.setAttribute("class", "lin-rule" + (i > D.now ? " is-future" : ""));
    l.setAttribute("x1", LC); l.setAttribute("x2", W - LC);
    l.setAttribute("y1", f(y)); l.setAttribute("y2", f(y));
    gRules.appendChild(l);
    const t = document.createElementNS(SVG, "text");
    t.setAttribute("class", "lin-stage" + (i > D.now ? " is-future" : ""));
    t.setAttribute("x", f(SPINE)); t.setAttribute("y", f(y - 9));
    t.textContent = s.label;
    gRules.appendChild(t);
  });

  D.strands.forEach(t => annotate(entry(t), t.side, "dataset", t.label, t.note, t.detail, t.href, t.future));
  D.exits.forEach(x => annotate(exit(x), x.side, x.kind === "control" ? "control" : "checkpoint",
                                x.label, x.note, x.detail, x.href, x.future));

  host.classList.add("is-ready");
})();
