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

  /* THE RAIL'S WIDTH BUDGET, left to right, and why it is 304 and not 240.
     the rail has to hold five things side by side: the corpus name (94, right-aligned,
     free to spill into the band's own 40-48px page padding), the arriving ribbon and its
     clear gap (about 43), the trunk itself (11 + 9.5 per corpus, 96 at the foot), the tap
     (36), and then the model's name.  a model name used to be its snake_case registry id
     -- `eeg_to_image`, 12 characters -- and 240 was enough.  it is now the model's real
     name with its parameter count, `IBM-1-EEG-to-Image (13.2M)`, which needs 100 even
     wrapped to two lines, and at 240 that ran a hundred pixels into the station's prose.
     if any of those five grows again, this number and .spine-rail's width in site.css
     have to move together -- they must match exactly, because the viewBox is 1:1. */
  const W = 304;        /* the rail column, in user units == px */
  const X_R = 202;      /* the trunk's right edge; fixed, so growth reads leftward */
  const W0 = 11;        /* trunk width above the first corpus */
  const GROW = 9.5;     /* width each corpus adds */
  const SPAN = 165;     /* vertical distance a merge is spread over */
  const STUB = 46;      /* straight run before a ribbon starts to veer */
  const GAP = 24;       /* clear space between an arriving ribbon and the trunk */
  const TAP = 5.5;      /* width of ONE model tapped off the right */
  const TAP_END = 36;   /* how far right of the trunk a tap runs before its label */
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
       a model id present in data/releases.js gets its hub link; anything else is WIP.

       a feed is `name` or `name|url`, and the NAME IS THE DATASET, never a modality.
       "film + audio", "sleep EEG" and "TMS-EEG" were what this rail said for months, and
       none of the three is a thing anyone can go and fetch -- the corpus each stood for is
       named on a card under data/sources/ and the card is what the label now carries. */
    const REL = (window.IBM_RELEASES && window.IBM_RELEASES.models) || {};
    const st = secs.map(sec => ({
      y: sec.getBoundingClientRect().top + window.scrollY - top + 52,
      feeds: (sec.dataset.feed || "").split("·").map(t => t.trim()).filter(Boolean)
        .map(t => { const [name, url] = t.split("|"); return { name: name.trim(), url: (url || "").trim() }; }),
      models: (sec.dataset.models || "").split(",").map(t => t.trim()).filter(Boolean)
        .map(id => ({ id,
          url: (REL[id] && REL[id][0] && REL[id][0].url) || null,
          params: (REL[id] && REL[id][0] && REL[id][0].params) || null })),
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
      /* ---- one tap per model that leaves here, each labelled and linked ----
         a tap may not open until the trunk has finished widening.  it used to start at
         s.y - SPAN*0.16, which is a third of the way INTO the merge, so at every junction
         the model appeared to leave before the corpus feeding it had arrived -- the two
         events were drawn in the wrong order.  MERGE_END is where widthAt() reaches its
         new value, and no tap starts above it. */
      const MERGE_END = s.y + SPAN / 2;
      if (s.models.length) {
        /* ONE tap, however many models leave here.  giving each its own tap 34px below
           the last said they were materialized one after another, days apart; they come
           out of the same fusion of the same kernel, so they share the splinter point and
           the tap simply widens with how many of them there are. */
        const n = s.models.length;
        const wTap = TAP * (1 + 0.3 * (n - 1));
        const xEnd = X_R + TAP_END;
        const oy = MERGE_END + 6, ey = oy + SPAN * 0.42;
        const tap = rails(X_R - wTap * 0.9, oy, xEnd - wTap / 2, ey, wTap);
        const anyLive = s.models.some(m => m.url);
        el("path", { class: "spine-tap" + (anyLive ? "" : " is-wip"),
          d: `M ${f(tap.right[0][0])} ${f(oy)} ${trace(tap.right)} L ${f(xEnd)} ${f(ey + 14)} ` +
             `L ${f(xEnd - wTap)} ${f(ey + 14)} ${back(tap.left)} Z` }, roads);
        /* the label is the artefact, not the request: the hub's own icon, the model's
           NAME, and the parameter count of the newest checkpoint released for it --
           releases.js is sorted by step descending, so [0] is the latest and not a
           maximum over the run. */
        const NAME = window.IBM_MODEL_NAME || ((id) => id);
        const PAR = window.IBM_PARAMS || ((v) => v);
        const g = el("g", { class: "spine-out" + (anyLive ? "" : " is-wip") }, gLabels);
        const fo = el("foreignObject", { x: f(xEnd - wTap), y: f(ey + 16),
                                         width: 100, height: 30 * n + 8 }, g);
        const rows = s.models.map((m) => {
          const name = NAME(m.id);
          return m.url
            ? `<a href="${m.url}" target="_blank" rel="noopener">` +
              `<span class="hf" aria-hidden="true">\u{1f917}</span>${name} <b>(${PAR(m.params)})</b></a>`
            : `<span>${name}<i> wip</i></span>`;
        }).join("");
        fo.innerHTML = `<div xmlns="http://www.w3.org/1999/xhtml" class="spine-outlbl">${rows}</div>`;
      }
      const oy = s.y - SPAN * 0.18, ey = oy + SPAN * 0.62;
      if (!s.feeds.length) return;
      /* one ribbon per corpus, side by side, so the trunk grows by the number of corpora
         that actually arrive here rather than by a fixed step per station */
      s.feeds.forEach((feed, k) => {
        /* 36, not 30: a dataset name that wraps to two lines needs the extra clearance
           over the stub below it */
        const yC = s.y - SPAN / 2, yM = s.y + SPAN / 2, yS = yC - STUB - k * 36;
        const lane = leftAt(yM) + GROW * (k + 0.5);
        const xI = leftAt(yM) - GAP - GROW * (s.feeds.length - k) - k * 16;
        const r = rails(xI, yC, lane, yM, GROW);
        /* THE BLEND FINISHES BEFORE THE MERGE STARTS.  it used to run yC -> yM, which is
           exactly the stretch over which the ribbon veers into the trunk, so all the way
           down the join the ribbon was some intermediate of the corpus hue and the trunk's
           and the two edges of the seam were different colours.  worse, the ribbon is drawn
           over the trunk there, so the mismatch showed as a hard discontinuous patch rather
           than as a blend.  running the gradient over the STRAIGHT stub instead -- yS to yC
           -- means the ribbon is already the trunk's own colour at the instant it starts to
           turn, and the merge itself is one flat colour with no seam in it at all. */
        const gid = "spine-blend-" + i + "-" + k;
        /* a FIXED 30-unit ramp ending 6 above yC, not one stretched over the whole stub:
           the second ribbon's stub is 36 longer than the first's, so a stub-relative ramp
           gave the two corpora arriving at one station visibly different rates of change.
           an SVG gradient pads outside its range, so above the ramp the ribbon is the
           corpus's own flat colour and below it the trunk's. */
        const bg = el("linearGradient", { id: gid, gradientUnits: "userSpaceOnUse",
                                          x1: 0, x2: 0, y1: f(yC - 36), y2: f(yC - 6) }, defs);
        el("stop", { offset: 0, style: `stop-color:${hue(i + k)}` }, bg);
        el("stop", { offset: 1, style: "stop-color:var(--accent)" }, bg);
        const fp = el("path", { class: "spine-feed",
          d: `M ${f(xI - GROW / 2)} ${f(yS)} L ${f(xI + GROW / 2)} ${f(yS)} ` +
             `L ${f(r.right[0][0])} ${f(yC)} ${trace(r.right)} ${back(r.left)} Z` }, roads);
        fp.style.fill = `url(#${gid})`;
        /* a foreignObject rather than <text>: the deepest ribbon starts about 60 user
           units from the left edge, and a one-line <text> anchored there ran a dataset
           name off the page.  a right-aligned block WRAPS instead, and it can carry the
           link to the corpus, which a bare <text> could not. */
        /* 94 is as wide as the deepest label can be: the last station's second ribbon
           starts about 68 user units in, so the block's left edge lands at about -44,
           which is inside the band's own 40-48px page padding and nothing clips. */
        const LW = 94;
        const fo = el("foreignObject", { x: f(xI - GROW / 2 - 11 - LW), y: f(yS - 3),
                                         width: LW, height: 34 }, gLabels);
        const inner = feed.url
          ? `<a xmlns="http://www.w3.org/1999/xhtml" href="${feed.url}" target="_blank" rel="noopener">${feed.name}</a>`
          : `<span xmlns="http://www.w3.org/1999/xhtml">${feed.name}</span>`;
        fo.innerHTML = `<div xmlns="http://www.w3.org/1999/xhtml" class="spine-inlbl">${inner}</div>`;
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
