/* the materialization figure: one implicit brain, three models traced out of it.

   two jobs.  first, the variables: the implicit model carries every component the
   registry declares, so a sample of their real ids drifts around it -- the point being
   that the centre brain is not "a brain picture", it is the whole state.  a materialized
   brain beside it shows only what its own request reaches, which is why those three are
   annotated with their declared inputs and outputs and nothing else.

   second, the links: curves from the implicit brain to each model, measured from the DOM
   after layout so they survive the figures reflowing or the column changing width. */
(function () {
  const host = document.getElementById("mz");
  if (!host) return;

  /* ---- the variables floating around the implicit model ---- */
  const vars = document.getElementById("mz-vars");
  const R = window.IBM_REGISTRY;
  if (vars && R && R.fields) {
    const all = [];
    R.fields.forEach((f) => (f.components || []).forEach((c) => all.push(c.id)));
    /* spread the sample across fields rather than taking the first N, which would be
       every component of `blood` and none of anything else */
    const step = Math.max(1, Math.floor(all.length / 22));
    const pick = all.filter((_, i) => i % step === 0).slice(0, 22);
    vars.innerHTML = pick.map((id, i) => {
      const t = (i / pick.length) * Math.PI * 2;
      const rx = 46 + (i % 3) * 5, ry = 44 + ((i + 1) % 3) * 5;
      const x = 50 + Math.cos(t) * rx, y = 50 + Math.sin(t) * ry;
      return `<span class="mz-var" style="left:${x.toFixed(1)}%;top:${y.toFixed(1)}%;` +
             `animation-delay:${(-i * 0.7).toFixed(1)}s">${id}</span>`;
    }).join("");
  }

  /* ---- the links ---- */
  const svg = document.getElementById("mz-links");
  const from = host.querySelector(".mz-implicit .mz-brain");
  const tos = Array.from(host.querySelectorAll(".mz-outs .mz-brain"));
  if (!svg || !from || !tos.length) return;

  function draw() {
    const hb = host.getBoundingClientRect();
    if (!hb.width) return;
    const f = from.getBoundingClientRect();
    svg.setAttribute("viewBox", `0 0 ${hb.width.toFixed(0)} ${hb.height.toFixed(0)}`);
    svg.setAttribute("width", hb.width);
    svg.setAttribute("height", hb.height);
    const x0 = f.right - hb.left, y0 = f.top + f.height / 2 - hb.top;
    svg.innerHTML = tos.map((t) => {
      const b = t.getBoundingClientRect();
      const x1 = b.left - hb.left, y1 = b.top + b.height / 2 - hb.top;
      const mx = (x0 + x1) / 2;
      return `<path class="mz-link" d="M ${x0.toFixed(1)} ${y0.toFixed(1)} ` +
             `C ${mx.toFixed(1)} ${y0.toFixed(1)} ${mx.toFixed(1)} ${y1.toFixed(1)} ` +
             `${x1.toFixed(1)} ${y1.toFixed(1)}"/>` +
             `<circle class="mz-dot" cx="${x1.toFixed(1)}" cy="${y1.toFixed(1)}" r="3"/>`;
    }).join("");
    host.classList.add("is-ready");
  }

  if (document.readyState === "complete") draw();
  else window.addEventListener("load", draw);
  let t;
  const again = () => { clearTimeout(t); t = setTimeout(draw, 140); };
  window.addEventListener("resize", again);
  if (window.ResizeObserver) new ResizeObserver(again).observe(host);
})();
