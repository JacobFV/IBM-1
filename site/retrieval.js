/* the THINGS retrieval panel: real trials from the designated test set, drawn as the
   two-model round trip the station's two headline figures actually score.

   the pool is the WHOLE 200-image test set, so chance is 1/200 exactly.  showing the
   stimulus beside "the predicted image" would, on a hit, show the same photograph twice
   and give the reader no way to tell whether the model chose it out of 200 or was handed
   it -- the evidence is the ranking, so the ranking is what is drawn.
   the trials are the FIRST 23 of the test set in dataset order, taken by
   scripts/export_site_retrieval.py before any rank was read.  it used to ship three
   picked as "the first two hits and the first near-miss" -- a stated rule, but a rule
   that read the outcome first, so three slots showed two hits whatever the real rate
   was.  in order, the misses come along unasked: of these 23, 12 land at rank 1, 20 in
   the top five, and three (antelope, beaver, bench) not in the top five at all -- their
   rows mark no candidate, and the rank line says where the true image went.

   the 23 rows slide DOWN continuously in a short masked window: the newest enters at the
   top and the oldest leaves at the bottom.  the track holds the list twice so the loop
   has no seam, and is laid out column-reverse so the DOM -- what a screen reader and the
   tab order walk -- stays in dataset order while the picture runs newest-first.  the
   second copy is aria-hidden and inert, so nothing is read or focused twice.

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
  const N = D.trials.length;
  const STATIC_ROWS = 4;
  const headFor = (still) => `<p class="rt-head">The first <b>${N}</b> trials of the
    designated test set, in dataset order &mdash; taken before any rank was read, so the
    misses are in here too. The pool is the whole <b>${D.n}</b>-image set, so chance is
    <b>${(100 * D.chance).toFixed(1)}%</b>. Each row walks both directions the station
    scores; the trace in the middle is the <b>measured</b> response, not a synthesised
    one. ${still
      ? `Motion is reduced, so the first ${Math.min(STATIC_ROWS, N)} are shown.`
      : "Hover or focus to hold it still."}</p>`;

  /* images carry data-src and are fetched only when the panel nears the viewport: 23
     rows x 6 pictures sit below the fold, and `loading=lazy` is no help here, because a
     row clipped by the window is "not visible" until the instant it slides in -- which is
     exactly when a pop-in shows.  the boxes are sized in CSS, so nothing moves on load. */
  const img = (src, alt) => `<img data-src="${esc(src)}" alt="${esc(alt)}" decoding="async">`;

  const row = (t) => `
    <div class="rt-row">
      <figure class="rt-seen">
        ${img(t.seen.src, t.seen.label)}
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
              ${img(c.src, c.label)}
              <span>${esc(c.label)}${c.isTrue ? '<i class="rt-sr">, the image that was seen</i>' : ""}</span>
            </li>`).join("")}</ol>
          <p class="rt-rank ${t.rank === 1 ? "is-hit" : "is-miss"}">
            rank ${t.rank} of ${D.n}${t.rank > 5 ? " &middot; not in the top five" : ""}</p>
        </div>
      </div>
    </div>`;

  const rows = D.trials.map(row).join("");
  const reduce = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

  let win = null, track = null, loaded = false;
  let H = 0, viewH = 0, speed = 0, s = 0;          // s: the track offset at the window's top
  let visible = false, hover = false, focus = false, raf = 0, last = 0;

  function loadImages() {
    if (loaded) return;
    loaded = true;
    host.querySelectorAll("img[data-src]").forEach((im) => { im.src = im.dataset.src; });
  }

  function render() {
    stop();
    const still = !!(reduce && reduce.matches);
    host.innerHTML = headFor(still) + (still
      ? `<div class="rt-window is-static"><div class="rt-track">${
          D.trials.slice(0, STATIC_ROWS).map(row).join("")}</div></div>`
      : `<div class="rt-window"><div class="rt-track">
           <div class="rt-copy">${rows}</div>
           <div class="rt-copy" aria-hidden="true" inert>${rows}</div>
         </div></div>`);
    win = host.querySelector(".rt-window");
    track = host.querySelector(".rt-track");
    if (loaded) { loaded = false; loadImages(); }
    if (!still) { measure(true); start(); }
    if (ro) observe();          // a re-render replaced the track it was watching
  }

  /* all geometry is READ from the laid-out rows, never assumed: a row is ~130px when it
     fits on one line and ~240px when it folds, and the fold depends on the panel's width.
     the copies are column-reverse, so the "A" copy (the real one) sits BELOW the aria-
     hidden one and H is the distance between the same row in each. */
  function measure(reset) {
    if (!win || win.classList.contains("is-static")) return;
    const copies = track.querySelectorAll(".rt-copy");
    const a = copies[0].firstElementChild, b = copies[1].firstElementChild;
    const frac = H ? s / H : 0;
    H = a.offsetTop - b.offsetTop;
    const avg = H / N;
    // three and a half rows when they are one line tall; two and a half when they fold,
    // or the window is taller than a phone screen
    viewH = Math.round(avg * (avg > 180 ? 2.5 : 3.5));
    win.style.height = viewH + "px";
    speed = avg / 3.2;                              // one row every ~3.2 s at any width
    // s lives in [H - viewH, 2H - viewH): the window then lies inside the track at every
    // phase, and s and s - H show the same pixels, which is what makes the loop seamless.
    // it starts at the bottom, where trial 0 is.
    s = reset ? 2 * H - viewH - 1 : frac * H;
    wrap();
    draw();
  }

  /* keep s inside [H - viewH, 2H - viewH) from BOTH sides.  wrapping only the side the
     loop travels toward was not enough: when the rows got shorter after the first
     measure, s was left beyond 2H - viewH, the window looked past the end of the track,
     and the panel showed an empty box that still reported itself as animating. */
  function wrap() {
    if (!(H > 0)) return;
    while (s < H - viewH) s += H;
    while (s >= 2 * H - viewH) s -= H;
  }
  function draw() { track.style.transform = `translate3d(0, ${-s}px, 0)`; }

  function tick(now) {
    const dt = Math.min(0.1, (now - last) / 1000);   // a background tab resumes, not leaps
    last = now;
    s -= speed * dt;                                  // window climbs the track = rows move DOWN
    wrap();
    draw();
    raf = requestAnimationFrame(tick);
  }
  function start() {
    if (raf || !visible || hover || focus || !win || win.classList.contains("is-static")) return;
    last = performance.now();
    raf = requestAnimationFrame(tick);
  }
  function stop() { if (raf) cancelAnimationFrame(raf); raf = 0; }

  /* the TRACK is observed, not the host: the host's height is pinned by the window, so a
     change in row height -- a webfont landing, a container query flipping the row from
     two lines to one -- never resized it, and H went stale without anything noticing. */
  function observe() {
    if (!("ResizeObserver" in window)) return;
    if (!ro) ro = new ResizeObserver(() => measure(false));
    ro.disconnect();
    ro.observe(host);
    if (track) ro.observe(track);
  }
  let ro = null;

  render();
  observe();

  /* nothing animates off-screen; images are fetched a screen ahead of arrival */
  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) loadImages();
    }), { rootMargin: "800px 0px" }).observe(host);
    new IntersectionObserver((es) => es.forEach((e) => {
      visible = e.isIntersecting;
      visible ? start() : stop();
    })).observe(host);
  } else { loadImages(); visible = true; start(); }

  host.addEventListener("mouseenter", () => { hover = true; stop(); });
  host.addEventListener("mouseleave", () => { hover = false; start(); });
  /* focus holds it still AND brings the focused row into the window: a link tabbed to
     while its row is clipped would otherwise be focused and invisible.  the window is
     overflow:clip, not hidden, so the browser cannot scroll it behind the transform. */
  host.addEventListener("focusin", (e) => {
    focus = true; stop();
    const r = e.target.closest && e.target.closest(".rt-row");
    if (r && win && !win.classList.contains("is-static")) {
      // clear of the fade at either edge; the track carries a fade's worth of padding
      // below trial 0, so even the last row can sit fully above the bottom fade
      const fade = parseFloat(getComputedStyle(win).getPropertyValue("--rt-fade")) || 0;
      s = Math.max(H - viewH, Math.min(track.offsetHeight - viewH, r.offsetTop - fade - 4));
      draw();
    }
  });
  host.addEventListener("focusout", (e) => {
    if (!host.contains(e.relatedTarget)) { focus = false; start(); }
  });


  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => measure(false));
  if (reduce) {
    const again = () => render();
    reduce.addEventListener ? reduce.addEventListener("change", again) : reduce.addListener(again);
  }

  host.classList.add("is-ready");
})();
