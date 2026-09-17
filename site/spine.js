/* the materialization spine.
   one continuous rail down the left of the page.  corpora fuse INTO it from the left;
   each proof-of-concept section is a model traced OUT of it to the right.  this is
   M = materialize(R, r, B, F, A, T, P) drawn as a road, and it deliberately carries NO
   chronology -- the lineage diagram owns training order, and only the training line is
   on it.  sleep, fusion, stimulation and the body are materializations of the same
   substrate and have no stage to sit at, so a timeline could not have carried them.

   the rail is one <svg> pinned behind the section column.  it is measured from the DOM
   after layout, so a section growing or wrapping cannot desynchronise it: every station
   is read from its section's own offsetTop rather than assumed.

   a station whose section is marked data-status="declared" is drawn OPEN -- dashed, no
   fill -- reusing the lineage's own convention that a planned thing is drawn open
   because it is declared rather than run. */
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

  const W = 120;            /* user units across the rail's own column */
  const RAIL = 26;          /* the trunk's width */
  const X = 46;             /* the trunk's centre within W */

  let svg = null;
  function draw() {
    const secs = Array.from(document.querySelectorAll(".poc[data-station]"));
    if (!secs.length) return;
    const top = host.getBoundingClientRect().top + window.scrollY;
    const H = host.offsetHeight;
    if (!H) return;

    if (svg) svg.remove();
    svg = el("svg", { class: "spine-svg", viewBox: `0 0 ${W} ${f(H)}`,
                      preserveAspectRatio: "none", "aria-hidden": "true" }, host);
    const defs = el("defs", {}, svg);
    /* the trunk fades in at the top and out at the bottom rather than butting into
       the section rules, so it reads as continuing past the page rather than stopping */
    const g = el("linearGradient", { id: "spine-fade", gradientUnits: "objectBoundingBox",
                                     x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
    el("stop", { offset: 0, style: "stop-color:var(--accent);stop-opacity:0" }, g);
    el("stop", { offset: 0.06, style: "stop-color:var(--accent);stop-opacity:.55" }, g);
    el("stop", { offset: 0.9, style: "stop-color:var(--accent);stop-opacity:.55" }, g);
    el("stop", { offset: 1, style: "stop-color:var(--accent);stop-opacity:0" }, g);

    el("rect", { class: "spine-trunk", x: f(X - RAIL / 2), y: 0, width: RAIL, height: f(H),
                 fill: "url(#spine-fade)" }, svg);

    secs.forEach(sec => {
      const y = sec.getBoundingClientRect().top + window.scrollY - top + 46;
      const declared = sec.dataset.status === "declared";
      const cls = declared ? " is-declared" : "";
      /* the branch out to the section: leaves the trunk vertically, lands horizontally */
      el("path", { class: "spine-branch" + cls,
                   d: `M ${f(X + RAIL / 2)} ${f(y)} C ${f(X + 40)} ${f(y)} ${f(X + 40)} ${f(y)} ${f(W)} ${f(y)}` }, svg);
      el("circle", { class: "spine-node" + cls, cx: f(X), cy: f(y), r: 7 }, svg);
      /* corpora fusing in from the left, where the section names one */
      if (sec.dataset.feed)
        el("path", { class: "spine-feed" + cls,
                     d: `M 0 ${f(y - 26)} C ${f(X - 30)} ${f(y - 26)} ${f(X - 30)} ${f(y)} ${f(X - RAIL / 2)} ${f(y)}` }, svg);
    });
    host.classList.add("is-ready");
  }

  const ready = () => { draw(); };
  if (document.readyState === "complete") ready();
  else window.addEventListener("load", ready);
  let t;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 180); });
  /* sections carry video and lazily-loaded art, so height settles after first paint */
  if (window.ResizeObserver && host.parentElement) {
    const ro = new ResizeObserver(() => { clearTimeout(t); t = setTimeout(draw, 120); });
    ro.observe(host.parentElement);
  }
})();
