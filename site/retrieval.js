/* the THINGS retrieval panel: real trials from the designated test set, drawn as the
   two-model round trip the station's two headline figures actually score.

   the pool is the WHOLE 200-image test set, so chance is 1/200 exactly.  showing the
   stimulus beside "the predicted image" would, on a hit, show the same photograph twice
   and give the reader no way to tell whether the model chose it out of 200 or was handed
   it -- the evidence is the ranking, so the ranking is what is drawn.
   the trials are the ones scripts/export_site_retrieval.py picked by a stated rule in
   dataset order, not by eye, and one of the three is a miss.

   a row is now TWO NAMED ARROWS, the same register as the materialization figure: an
   arrow with the model's name set over it, carrying the eye from one artefact to the
   next.  that is not decoration -- the station reports two separate retrieval directions
   (image -> EEG and EEG -> image) and the old single glyph drew only one of them, so the
   picture and the numbers were describing different things.

   BUT the waveform between the two arrows is a MEASURED evoked response, not something
   an image->EEG model emitted.  an arrow named for a model, pointing at a recording,
   invites exactly the wrong reading, so the trace carries its own "measured" caption and
   the head says it in words.  the arrow names the DIRECTION that was scored; it does not
   claim provenance over the bytes drawn under it.

   `image_to_eeg` is a direction, not a released checkpoint -- it appears in neither
   data/graph.js's materializations nor data/releases.js's models -- so the name is right
   but it gets NO link; a link to a hub URL that does not exist is worse than no link.
   `eeg_to_image` has three released checkpoints and the newest one is linked.

   every number below is read out of window.IBM_RETRIEVAL rather than typed, so the panel
   cannot drift from the run that produced it -- and every one of them is quoted against
   the pool it was measured over. */
(function () {
  const host = document.getElementById("retrieval");
  const D = window.IBM_RETRIEVAL;
  if (!host || !D) return;
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  /* site.js defines IBM_MODEL_NAME and index.html loads it first.  falling back rather
     than throwing: a reordering of the script tags should cost the panel its proper
     names, not delete the panel. */
  const NAME = window.IBM_MODEL_NAME || ((id) => id);
  const REL = window.IBM_RELEASES;
  /* releases.js is sorted by step descending, so [0] is the newest checkpoint */
  const newest = (id) => (REL && REL.models && REL.models[id] && REL.models[id][0]) || null;

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

  /* one leg of the round trip: the model's name over an arrow that stretches to whatever
     width the row can spare.  the shaft is aria-hidden because it carries no information
     a screen reader can use -- the direction is spelled out in .rt-sr instead, since the
     arrow IS meaningful here and losing it would lose the whole point of the row. */
  function leg(id, dir, said) {
    const ck = dir === "out" ? newest(id) : null;
    const name = esc(NAME(id));
    const body = ck
      ? `<a href="${esc(ck.url)}" target="_blank" rel="noopener"` +
        ` title="${esc(ck.name)} · ${window.IBM_PARAMS ? window.IBM_PARAMS(ck.params) : ck.params} parameters">${name}</a>`
      : name;
    return `<div class="rt-leg rt-leg-${dir}">
        <b>${body}</b>
        <span class="rt-shaft" aria-hidden="true"></span>
        <span class="rt-sr">${esc(said)}</span>
      </div>`;
  }

  const legIn = leg("image_to_eeg", "in", "the image that was seen drives the evoked response");
  const legOut = leg("eeg_to_image", "out",
    `the evoked response is ranked against all ${D.n} images in the test set`);

  /* the head carries what the two big figures beside this panel do NOT: the pool they
     were measured over, and therefore chance.  it does not restate the percentages --
     restating a number in a second place is how the two drift apart. */
  const head = `<p class="rt-head">Each row walks both directions the station scores.
    The pool is the whole <b>${D.n}</b>-image designated test set, so chance is
    <b>${(100 * D.chance).toFixed(1)}%</b>. Three real trials taken in dataset order &mdash;
    the first two hits and the first near-miss, not the best three. The trace in the middle
    is the <b>measured</b> response, not a synthesised one.</p>`;

  host.innerHTML = head + D.trials.map((t) => `
    <div class="rt-row">
      <figure class="rt-seen">
        <img src="${esc(t.seen.src)}" alt="${esc(t.seen.label)}" loading="lazy">
        <figcaption>${esc(t.seen.label)}</figcaption>
      </figure>
      <div class="rt-step">
        ${legIn}
        <figure class="rt-mid">
          ${traces(t.eeg)}
          <figcaption><span>measured &middot;</span> <span>${t.eeg.length} ch shown</span></figcaption>
        </figure>
      </div>
      <div class="rt-step rt-step-out">
        ${legOut}
        <div class="rt-out">
          <ol class="rt-top5">${t.top5.map((c) => `
            <li class="${c.isTrue ? "is-true" : ""}">
              <img src="${esc(c.src)}" alt="${esc(c.label)}" loading="lazy">
              <span>${esc(c.label)}${c.isTrue ? '<i class="rt-sr">, the image that was seen</i>' : ""}</span>
            </li>`).join("")}</ol>
          <p class="rt-rank ${t.rank === 1 ? "is-hit" : "is-miss"}">
            ${t.rank === 1 ? `rank 1 of ${D.n}` : `rank ${t.rank} of ${D.n}`}</p>
        </div>
      </div>
    </div>`).join("");

  host.classList.add("is-ready");
})();
