/* the lineage track -- a vertical serpentine of what fed the substrate and what came
   out of it.  the path is drawn once as an SVG cubic chain; nodes are HTML so they
   can carry focus, a tooltip and a link.  everything up to IBM_LINEAGE.now is FILLED;
   everything after it is drawn unchecked, because it is planned rather than done. */
(function () {
  const data = window.IBM_LINEAGE;
  const host = document.getElementById("lineage");
  if (!data || !host) return;

  const N = data.nodes.length;
  const ROW = 128;                 /* vertical spacing between nodes */
  const PAD = 46;                  /* top and bottom breathing room */
  const H = PAD * 2 + ROW * (N - 1);
  const AMP = 30;                  /* how far the snake swings, in % of width */

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "lin-svg");
  svg.setAttribute("viewBox", `0 0 100 ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");

  const x = i => 50 + (i % 2 === 0 ? -AMP : AMP);
  const y = i => PAD + ROW * i;

  /* one cubic between each pair, so the path reads as a single snake */
  let d = `M ${x(0)} ${y(0)}`;
  for (let i = 1; i < N; i++) {
    const my = (y(i - 1) + y(i)) / 2;
    d += ` C ${x(i - 1)} ${my}, ${x(i)} ${my}, ${x(i)} ${y(i)}`;
  }

  const back = document.createElementNS(svg.namespaceURI, "path");
  back.setAttribute("d", d); back.setAttribute("class", "lin-path lin-path-future");
  svg.appendChild(back);

  /* the filled portion is the same path, clipped by stroke-dasharray to the
     fraction of segments already done -- measured on the real path length so it
     lands on the node rather than near it. */
  const done = document.createElementNS(svg.namespaceURI, "path");
  done.setAttribute("d", d); done.setAttribute("class", "lin-path lin-path-done");
  svg.appendChild(done);
  host.appendChild(svg);

  requestAnimationFrame(() => {
    const L = done.getTotalLength();
    const frac = N > 1 ? Math.min(data.now, N - 1) / (N - 1) : 1;
    done.style.strokeDasharray = `${L * frac} ${L}`;
  });

  const KIND = { corpus: "dataset", model: "checkpoint", note: "control" };
  data.nodes.forEach((n, i) => {
    const el = document.createElement(n.href ? "a" : "div");
    const future = i > data.now;
    el.className = `lin-node ${future ? "is-future" : "is-done"} lin-${n.kind}`;
    if (n.href) { el.href = n.href; if (/^https?:/.test(n.href)) { el.target = "_blank"; el.rel = "noopener"; } }
    el.style.top = `${(y(i) / H) * 100}%`;
    el.style.left = `${x(i)}%`;
    el.setAttribute("tabindex", "0");
    el.innerHTML =
      `<span class="lin-mark" aria-hidden="true"></span>` +
      `<span class="lin-body">` +
        `<span class="lin-kind">${KIND[n.kind] || n.kind}${future ? " · planned" : ""}</span>` +
        `<b class="lin-label">${n.label}</b>` +
        `<span class="lin-note">${n.note}</span>` +
      `</span>` +
      `<span class="lin-tip" role="tooltip">${n.detail}</span>`;
    host.appendChild(el);
  });

  host.classList.add("is-ready");
})();
