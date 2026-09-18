/* the spine: the shared implicit model as a trunk, and the corpora fused into it.
   the trunk grows only on its LEFT, and it grows by exactly the width of the corpus
   ribbon arriving there -- the widening IS the fusion, not a decoration beside it.
   models are tapped off the RIGHT, which is why the trunk never narrows: a
   materialisation reads the kernel, it does not consume it.

   GEOMETRY.  nothing joins at a right angle.  every ribbon is a cubic whose tangent is
   vertical where it leaves its stub and vertical where it merges, so a corpus arrives
   ALONGSIDE the trunk rather than colliding with it, and the merge is tangential.

   the ribbons hold a constant PERPENDICULAR width.  drawing a filled road as two copies
   of one cubic offset sideways by w keeps the HORIZONTAL width constant, so the width a
   reader actually measures -- across the road -- falls to w*cos(theta), and every veer
   pinches in its middle and swells back at both ends.  rails() instead widens the
   horizontal offset by sec(theta): two points a horizontal w/cos(theta) apart are exactly
   w apart across the road.  the offset stays horizontal rather than along the normal,
   because a normal offset moves each edge vertically by (w/2)sin(theta), which grows
   linearly out of the join and creases the outer edge where the veer leaves its stub;
   sec(theta) grows as theta^2, so the join is invisible. */
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

  const W = 240;        /* the rail column, in user units == px */
  const X_R = 214;      /* the trunk's right edge; fixed, so growth reads leftward */
  const W0 = 11;        /* trunk width above the first corpus */
  const GROW = 9.5;     /* width each corpus adds */
  const SPAN = 165;     /* vertical distance a merge is spread over */
  const STUB = 30;      /* straight run before a ribbon starts to veer */
  const GAP = 24;       /* clear space between an arriving ribbon and the trunk */
  const TAP = 5.5;      /* width of a model tapped off the right */
  const N = 44;
  const hue = i => `var(--lane-${(i % 4) + 1})`;

  const smooth = t => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

  /* a ribbon of constant perpendicular width along a cubic with vertical tangents */
  function rails(xa, ya, xb, yb, w) {
    const ym = (ya + yb) / 2, h = w / 2, right = [], left = [];
    for (let i = 0; i <= N; i++) {
      const t = i / N, u = 1 - t;
      const x = xa * u * u * u + 3 * xa * u * u * t + 3 * xb * u * t * t + xb * t * t * t;
      const y = ya * u * u * u + 3 * ym * u * u * t + 3 * ym * u * t * t + yb * t * t * t;
      const dx = 6 * (xb - xa) * u * t, dy = 1.5 * (yb - ya) * (u * u + t * t);
      const sec = Math.hypot(dx, dy) / Math.abs(dy);
      right.push([x + h * sec, y]);
      left.push([x - h * sec, y]);
    }
    return { right, left };
  }
  const trace = pts => pts.map(p => `L ${f(p[0])} ${f(p[1])}`).join(" ");
  const back = pts => trace(pts.slice().reverse());

  let svg = null;
  function draw() {
    const secs = Array.from(document.querySelectorAll("[data-station]"));
    if (!secs.length) return;
    const H = host.offsetHeight;
    if (!H) return;                        /* hidden at narrow widths, by design */
    const top = host.getBoundingClientRect().top + window.scrollY;

    /* a station declares the corpora that fuse IN and the models that splinter OUT.
       a model id present in data/releases.js gets its hub link; anything else is WIP. */
    const REL = (window.IBM_RELEASES && window.IBM_RELEASES.models) || {};
    const st = secs.map(sec => ({
      y: sec.getBoundingClientRect().top + window.scrollY - top + 52,
      feeds: (sec.dataset.feed || "").split("·").map(t => t.trim()).filter(Boolean),
      models: (sec.dataset.models || "").split(",").map(t => t.trim()).filter(Boolean)
        .map(id => ({ id, url: (REL[id] && REL[id][0] && REL[id][0].url) || null })),
    }));

    /* trunk width at y: W0 plus one smoothstep per corpus already fused */
    const widthAt = y => {
      let w = W0;
      for (const s of st) w += s.feeds.length * GROW * smooth((y - (s.y - SPAN / 2)) / SPAN);
      return w;
    };
    const leftAt = y => X_R - widthAt(y);

    if (svg) svg.remove();
    svg = el("svg", { class: "spine-svg", viewBox: `0 0 ${W} ${f(H)}`,
                      preserveAspectRatio: "none", "aria-hidden": "true" }, host);
    const defs = el("defs", {}, svg);
    /* every road goes in ONE group carrying ONE opacity, and each path inside is fully
       opaque.  giving the trunk, the feeds and the taps their own alpha made every
       overlap composite to a different shade, so each merge and each tap showed a
       lighter patch with a hard seam across it -- the shapes were correct and the
       compositing was not.  the top-and-tail fade therefore cannot be per-path either:
       it is a MASK over the whole group, which fades the result rather than the parts. */
    const mg = el("linearGradient", { id: "spine-fademask", gradientUnits: "userSpaceOnUse",
                                      x1: 0, y1: 0, x2: 0, y2: f(H) }, defs);
    el("stop", { offset: 0, style: "stop-color:#000" }, mg);
    el("stop", { offset: 0.045, style: "stop-color:#fff" }, mg);
    el("stop", { offset: 0.93, style: "stop-color:#fff" }, mg);
    el("stop", { offset: 1, style: "stop-color:#000" }, mg);
    const mask = el("mask", { id: "spine-mask", maskUnits: "userSpaceOnUse",
                              x: 0, y: 0, width: W, height: f(H) }, defs);
    el("rect", { x: 0, y: 0, width: W, height: f(H), fill: "url(#spine-fademask)" }, mask);
    const roads = el("g", { class: "spine-roads", mask: "url(#spine-mask)" }, svg);
    const gLabels = el("g", { class: "spine-labels" }, svg);

    /* --- the trunk: a right edge that never moves and a left edge that steps out --- */
    const L = [];
    for (let y = 0; y <= H; y += 2) L.push([leftAt(y), y]);
    L.push([leftAt(H), H]);
    el("path", { class: "spine-trunk",
      d: `M ${f(X_R)} 0 L ${f(X_R)} ${f(H)} ${back(L)} Z` }, roads);

    st.forEach((s, i) => {
      /* ---- one tap per model that leaves here, each labelled and linked ---- */
      s.models.forEach((m, k) => {
        const oy = s.y - SPAN * 0.16 + k * 34, ey = oy + SPAN * 0.5;
        const tap = rails(X_R - TAP * 0.9, oy, W - TAP / 2 - 2, ey, TAP);
        el("path", { class: "spine-tap" + (m.url ? "" : " is-wip"),
          d: `M ${f(tap.right[0][0])} ${f(oy)} ${trace(tap.right)} L ${f(W - 2)} ${f(ey + 14)} ` +
             `L ${f(W - TAP - 2)} ${f(ey + 14)} ${back(tap.left)} Z` }, roads);
        const g = el("g", { class: "spine-out" + (m.url ? "" : " is-wip") }, gLabels);
        const fo = el("foreignObject", { x: f(W - TAP - 2), y: f(ey + 16), width: 200, height: 34 }, g);
        const a = m.url ? `<a xmlns="http://www.w3.org/1999/xhtml" href="${m.url}" target="_blank" rel="noopener">${m.id}</a>`
                        : `<span xmlns="http://www.w3.org/1999/xhtml">${m.id}<i> wip</i></span>`;
        fo.innerHTML = `<div xmlns="http://www.w3.org/1999/xhtml" class="spine-outlbl">${a}</div>`;
      });
      const oy = s.y - SPAN * 0.18, ey = oy + SPAN * 0.62;
      if (!s.feeds.length) return;
      /* one ribbon per corpus, side by side, so the trunk grows by the number of corpora
         that actually arrive here rather than by a fixed step per station */
      s.feeds.forEach((name, k) => {
        const yC = s.y - SPAN / 2, yM = s.y + SPAN / 2, yS = yC - STUB - k * 30;
        const lane = leftAt(yM) + GROW * (k + 0.5);
        const xI = leftAt(yM) - GAP - GROW * (s.feeds.length - k) - k * 16;
        const r = rails(xI, yC, lane, yM, GROW);
        const gid = "spine-blend-" + i + "-" + k;
        const bg = el("linearGradient", { id: gid, gradientUnits: "userSpaceOnUse",
                                          x1: 0, x2: 0, y1: f(yC), y2: f(yM) }, defs);
        el("stop", { offset: 0, style: `stop-color:${hue(i + k)}` }, bg);
        el("stop", { offset: 1, style: "stop-color:var(--accent)" }, bg);
        const fp = el("path", { class: "spine-feed",
          d: `M ${f(xI - GROW / 2)} ${f(yS)} L ${f(xI + GROW / 2)} ${f(yS)} ` +
             `L ${f(r.right[0][0])} ${f(yC)} ${trace(r.right)} ${back(r.left)} Z` }, roads);
        fp.style.fill = `url(#${gid})`;
        const t = el("text", { class: "spine-label", x: f(xI - GROW / 2 - 11), y: f(yS + 9) }, gLabels);
        t.textContent = name;
      });
      return;
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
