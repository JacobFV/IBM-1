/* the THINGS retrieval panel: real trials from the designated test set.

   the pool is the WHOLE 200-image test set, so chance is 1/200 exactly.  showing the
   stimulus beside "the predicted image" would, on a hit, show the same photograph twice
   and give the reader no way to tell whether the model chose it out of 200 or was handed
   it -- the evidence is the ranking, so the ranking is what is drawn.
   the trials are the ones scripts/export_site_retrieval.py picked by a stated rule in
   dataset order, not by eye, and one of the three is a miss. */
(function () {
  const host = document.getElementById("retrieval");
  const D = window.IBM_RETRIEVAL;
  if (!host || !D) return;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  /* the evoked response, drawn as one polyline per exported channel */
  function traces(rows) {
    const W = 150, H = 66, pad = 3;
    let lo = Infinity, hi = -Infinity;
    rows.forEach((r) => r.forEach((v) => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    const span = (hi - lo) || 1;
    const paths = rows.map((r) => {
      const d = r.map((v, i) =>
        (i ? "L" : "M") + (pad + i / (r.length - 1) * (W - 2 * pad)).toFixed(1) + " " +
        (pad + (hi - v) / span * (H - 2 * pad)).toFixed(1)).join("");
      return `<path d="${d}"/>`;
    }).join("");
    return `<svg class="rt-eeg" viewBox="0 0 ${W} ${H}" role="img" aria-label="measured evoked response">${paths}</svg>`;
  }

  host.innerHTML = D.trials.map((t) => `
    <div class="rt-row">
      <figure class="rt-seen">
        <img src="${esc(t.seen.src)}" alt="${esc(t.seen.label)}" loading="lazy">
        <figcaption>${esc(t.seen.label)}</figcaption>
      </figure>
      <div class="rt-mid">${traces(t.eeg)}<span class="rt-arrow">&rarr;</span></div>
      <div class="rt-out">
        <ol class="rt-top5">${t.top5.map((c) => `
          <li class="${c.isTrue ? "is-true" : ""}">
            <img src="${esc(c.src)}" alt="${esc(c.label)}" loading="lazy">
            <span>${esc(c.label)}</span>
          </li>`).join("")}</ol>
        <p class="rt-rank ${t.rank === 1 ? "is-hit" : "is-miss"}">
          ${t.rank === 1 ? "rank 1 of 200" : `rank ${t.rank} of 200`}</p>
      </div>
    </div>`).join("");

  host.classList.add("is-ready");
})();
