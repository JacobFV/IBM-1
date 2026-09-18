/* the materialization diagram: ONE scene, four brains.

   the implicit model sits at the LEFT in full atlas colour and carries every variable.
   three materialized models stack to its right, each drained to grey except the sites its
   own request reaches, with its declared inputs in the observed ramp and its outputs in
   the target one.

   READING ORDER IS THE LAYOUT.  each model is a lane read left to right: the declared
   inputs, then an arrow that cuts straight through the brain -- red on the way in, purple
   on the way out -- then the declared outputs.  the model's name sits over the arrow.
   nothing carries an "IN"/"OUT" tag or repeats its own id: side, colour and position say
   all three, and the tags were three redundant words per label on a figure that had no
   room for them.

   this replaced a RING of three models around the hub.  the ring only worked if every
   label sat radially outward from its brain, which put the longest name (IBM-1-EEG-to-Image)
   off the right edge of the canvas at every width below about 1100px -- the figure was
   correct and the annotation was outside the picture.  a lane stack has a fixed left and a
   fixed right, so every label's anchor side is known in advance and can be clamped into
   the canvas box; see put() at the bottom.

   built as one scene rather than four canvases because four separate renders cannot share
   a camera -- the stack only reads as one substrate and three views of it if all four
   brains are in the same space and turn together.

   a faint amber thread still runs from the hub into the lower-left of each model brain:
   that is the materialization itself, deliberately drawn thinner and dimmer than the data
   arrow so the two are never confused.  it arrives from BELOW the input column because
   arriving from the left would have crossed it. */
(function () {
  const host = document.getElementById("mz");
  const G = window.IBM_GRAPH;
  if (!host || !G || !window.THREE) return;
  const T = window.THREE;

  /* the displayed name comes from site.js's shared deriver, so this figure and the spine
     rail cannot drift into calling the same model two different things */
  const NAME = window.IBM_MODEL_NAME || ((id) => id);
  const MODELS = ["eeg_to_image", "speech_envelope", "tms_response"]
    .map((id) => ({ id, name: NAME(id) }));
  const MATS = Object.fromEntries((G.materializations || []).map((m) => [m.id, m]));
  const N = G.nodes.length;
  const CORTEX = G.cortex_range ? G.cortex_range[1] : N;

  const css = getComputedStyle(document.documentElement);
  const IN = new T.Color(css.getPropertyValue("--trace-in").trim() || "#f5424d");
  const OUT = new T.Color(css.getPropertyValue("--trace-out").trim() || "#a159fa");

  /* positions, centred and scaled so a brain is about one unit tall.

     THE GRAPH IS SURFACE RAS: +x right, +y ANTERIOR, +z SUPERIOR.  three.js is y-up, so
     the raw arrays render a head tipped 90 degrees onto its back, looking at the ceiling.
     brain.js line 15 already owns the fix -- (-x, z, y) -- and this used the raw triple
     instead, which is why these three brains and the hero's disagreed about which way up a
     head is.  mirroring x as well as swapping y and z keeps the determinant at +1, so it is
     a rotation and not a reflection: triangle winding, and therefore the cortex shell's
     normals, survive it. */
  const ctr = new T.Vector3().fromArray(G.center || [0, 0, 0]);
  const pos = new Float32Array(N * 3);
  let span = 0;
  for (let i = 0; i < N; i++) {
    const p = G.nodes[i].p;
    pos[i * 3] = -(p[0] - ctr.x); pos[i * 3 + 1] = p[2] - ctr.z; pos[i * 3 + 2] = p[1] - ctr.y;
    span = Math.max(span, Math.abs(pos[i * 3]), Math.abs(pos[i * 3 + 1]), Math.abs(pos[i * 3 + 2]));
  }
  const S = 1 / span;
  for (let i = 0; i < pos.length; i++) pos[i] *= S;

  const inRanges = (ranges, i) => {
    if (!ranges) return false;
    for (const r of ranges) if (i >= r[0] && i <= r[1]) return true;
    return false;
  };

  /* one brain: a point cloud, plus the cortex surface as a faint shell */
  function makeBrain(colorOf, opacity) {
    const g = new T.Group();
    const col = new Float32Array(N * 3);
    const c = new T.Color();
    for (let i = 0; i < N; i++) {
      colorOf(i, c);
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.BufferAttribute(pos, 3));
    geo.setAttribute("color", new T.BufferAttribute(col, 3));
    g.add(new T.Points(geo, new T.PointsMaterial({
      size: 0.016, vertexColors: true, transparent: true, opacity,
      sizeAttenuation: true, depthWrite: false,
    })));
    if (G.cortex_faces && G.cortex_faces.length) {
      const sg = new T.BufferGeometry();
      sg.setAttribute("position", new T.BufferAttribute(pos.slice(0, CORTEX * 3), 3));
      sg.setAttribute("color", new T.BufferAttribute(col.slice(0, CORTEX * 3), 3));
      sg.setIndex(G.cortex_faces.flat());
      sg.computeVertexNormals();
      g.add(new T.Mesh(sg, new T.MeshBasicMaterial({
        vertexColors: true, transparent: true, opacity: opacity * 0.3,
        side: T.DoubleSide, depthWrite: false,
      })));
    }
    return g;
  }

  const scene = new T.Scene();
  const world = new T.Group(); scene.add(world);

  /* ---- the frame the whole diagram is laid out in: two LAYOUTS ----
     every number is in world units, and the camera is fitted to the layout's HALF_X/HALF_Y
     (see resize), so nothing can drift outside the render.

     WIDE is the reading order the figure was designed for: the implicit model on the left,
     three lanes of materialized brains to its right.  TALL is for phones, and it is a
     different arrangement rather than the same one shrunk: at 390px the wide layout left
     each lane ~100px, the model names printed straight over their own port labels
     ("IBM-1-EEG-to-Imphotoreceptor state"), and the whole figure read as noise.  tall puts
     the implicit model on TOP and gives each lane the full width, so a lane keeps its
     left-to-right reading -- inputs, brain, outputs -- at a size a phone can read.  the
     layout is chosen from the CANVAS's own width (CLAUDE.md: a breakpoint is a statement
     about the viewport, and inside a station the two come apart) and the scene is rebuilt
     only when that choice flips, not on every resize. */
  const LAYOUTS = {
    wide: { HALF_X: 3.30, HALF_Y: 1.95, CAM_Y: 0.00, HUB_X: -2.15, HUB_Y: 0.00, HUB_S: 1.2,
            MODEL_X: 0.90, LANE_Y: [1.3, 0, -1.3], threads: "fan" },
    tall: { HALF_X: 2.45, HALF_Y: 3.10, CAM_Y: 0.33, HUB_X: 0.00, HUB_Y: 2.55, HUB_S: 0.95,
            MODEL_X: 0.00, LANE_Y: [0.6, -0.8, -2.2], threads: "arrow" },
  };
  const TALL_BELOW_PX = 640;
  const SUB = 0.44;                      /* a materialized brain's scale */
  const CHIP_GAP = 0.225;                /* vertical pitch of one input/output label */
  /* the arrow reaches 0.78 either side of a brain's centre, 0.34 clear of its edge.  it
     was 1.25 -- nearly two brain-widths -- and the ports read as a separate column of text
     on the far side of the picture rather than as that brain's own ports. */
  let L = LAYOUTS.wide, layoutName = null;
  let HUB_X, HUB_Y, HUB_S, MODEL_X, LANE_Y, TAIL, HEAD;
  let implicit = null, anchors = [], spinners = [];

  /* ---- annotations, as projected HTML so they stay crisp and themeable ----
     a mark owns its ANCHOR SIDE: an input hangs its right edge on its point, an output
     its left, a name and the hub their centre.  put() below needs that to clamp a label
     into the canvas instead of letting it run off the edge. */
  const layer = document.createElement("div");
  layer.className = "mz-labels";
  host.appendChild(layer);
  const marks = [];
  const addMark = (p, cls, html, anchor, dy, ride) => {
    const el = document.createElement("div");
    el.className = "mz-mark " + cls;
    el.innerHTML = html;
    layer.appendChild(el);
    marks.push({ p: p.clone(), el, anchor, dy: dy || 0, w: 0, ride: ride || null });
  };

  /* ---- build (or rebuild) the scene and its labels for one layout ---- */
  function build(name) {
    layoutName = name;
    L = LAYOUTS[name];
    ({ HUB_X, HUB_Y, HUB_S, MODEL_X, LANE_Y } = L);
    TAIL = MODEL_X - 0.78; HEAD = MODEL_X + 0.78;
    for (const o of world.children.slice()) {
      world.remove(o);
      o.traverse((n) => { if (n.geometry) n.geometry.dispose(); });
    }
    layer.innerHTML = "";
    marks.length = 0;
    anchors = [];
  /* the implicit model: every site in its own atlas colour */
  const atlas = G.nodes.map((n) => new T.Color(n.c || "#888"));
  implicit = makeBrain((i, c) => c.copy(atlas[i]), 0.95);
  implicit.scale.setScalar(HUB_S);
  implicit.position.set(HUB_X, HUB_Y, 0);
  world.add(implicit);

  MODELS.forEach((M, k) => {
    const m = MATS[M.id];
    const grey = new T.Color("#6f7681");
    const b = makeBrain((i, c) => {
      const isIn = (m.inputs || []).some((x) => inRanges(x.nodes, i));
      const isOut = (m.outputs || []).some((x) => inRanges(x.nodes, i));
      if (isIn) return c.copy(IN);
      if (isOut) return c.copy(OUT);
      if (inRanges(m.hot || m.involved, i)) return c.copy(atlas[i]).lerp(grey, 0.35);
      return c.copy(grey);
    }, 0.55);
    b.scale.setScalar(SUB);
    b.position.set(MODEL_X, LANE_Y[k], 0);
    world.add(b);
    anchors.push({ model: M, m, group: b, y: LANE_Y[k] });
  });

  /* ---- the data: one strand per port, bundled through the brain ----
     every input gets its own strand, and every strand converges on the brain's centre with
     a HORIZONTAL tangent; from there the bundle splits again, one strand to each output.
     a single straight arrow said "these inputs, this model, these outputs" as three
     separate facts; a bundle says the inputs are fused inside the model and the outputs
     are read back out of the same state, which is the claim.

     each strand is a cubic whose control points are pulled level with its two ends, so it
     leaves its port and enters the brain travelling horizontally -- the same
     tangent-continuity rule spine.js uses for the rail, for the same reason: a strand that
     arrives at an angle reads as colliding with the thing it joins.

     the colour ramp is laid by WORLD x across the whole lane, not per strand, so every
     strand is the same hue at the same distance along and the bundle's two halves meet in
     one colour at the centre, with no strand brighter or earlier than its neighbours. */
  const bundleMat = new T.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 });
  const headMat = new T.MeshBasicMaterial({ color: OUT, transparent: true, opacity: 0.85 });
  const rampByX = (g) => {
    const pa = g.attributes.position, col = new Float32Array(pa.count * 3), c = new T.Color();
    for (let i = 0; i < pa.count; i++) {
      c.copy(IN).lerp(OUT, Math.min(1, Math.max(0, (pa.getX(i) - TAIL) / (HEAD - TAIL))));
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    }
    g.setAttribute("color", new T.BufferAttribute(col, 3));
    return g;
  };
  const strand = (x0, y0, x1, y1) => {
    const k = (x1 - x0) * 0.55;
    const curve = new T.CubicBezierCurve3(
      new T.Vector3(x0, y0, 0), new T.Vector3(x0 + k, y0, 0),
      new T.Vector3(x1 - k, y1, 0), new T.Vector3(x1, y1, 0));
    world.add(new T.Mesh(rampByX(new T.TubeGeometry(curve, 40, 0.0085, 8, false)), bundleMat));
  };
  /* the SAME vertical pitch the port labels use, so each strand starts exactly at its label */
  const portY = (a, n, i) => a.y + ((n - 1) / 2 - i) * CHIP_GAP;
  anchors.forEach((a) => {
    const ins = a.m.inputs || [], outs = a.m.outputs || [];
    ins.forEach((_, i) => strand(TAIL, portY(a, ins.length, i), MODEL_X, a.y));
    outs.forEach((_, j) => {
      const y = portY(a, outs.length, j);
      strand(MODEL_X, a.y, HEAD, y);
      const head = new T.Mesh(new T.ConeGeometry(0.038, 0.11, 14), headMat);
      head.position.set(HEAD + 0.055, y, 0);
      head.rotation.z = -Math.PI / 2;
      world.add(head);
    });
  });

  /* ---- the materialization thread: hub -> each model ----
     each thread must END ON ITS BRAIN AND POINT AT IT.  the last version ended at
     (MODEL_X - 0.95*SUB, lane - 0.78*SUB) arriving horizontally, and every tip landed 30-45px
     below and left of its brain, pointing along the floor at nothing: 0.95*SUB was sized
     against the brain's HEIGHT, but normalisation divides by the superior-inferior extent,
     so a brain is only +-0.74 wide in x and +0.67/-1.0 tall.  now each thread ends on that
     brain's own lower-left surface, at the same bearing for all three, and its final
     tangent runs along the radius into the brain's centre, so the arrowhead aims at it.

     the family stays symmetric the way the user asked for: one start point on the hub, at
     the height the middle thread ends at, horizontal leaving tangents, and the same
     arrival bearing -- so the middle thread is nearly straight and the outer two are the
     same curve reflected, give or take the shared approach angle. */
  const threadMat = new T.MeshBasicMaterial({ color: 0xf0b34a, transparent: true, opacity: 0.4 });
  /* the brain's half-extents in THIS figure's units, from the normalised cloud */
  const BX = 0.74 * SUB, BY_DN = 1.0 * SUB, BY_UP = 0.67 * SUB;
  const BEAR = Math.PI * (205 / 180);                     /* lower-left, same for every lane */
  const ux = Math.cos(BEAR), uy = Math.sin(BEAR);
  /* where that bearing leaves an ellipse with the brain's half-extents, 0.9x so the tip
     sits just inside the edge of the cloud rather than on its outermost point */
  const rr = 0.9 / Math.sqrt((ux / BX) ** 2 + (uy / (uy < 0 ? BY_DN : BY_UP)) ** 2);
  const T_FROM = new T.Vector3(HUB_X + HUB_S * 0.66, HUB_Y + uy * rr, 0.05);
  /* TALL: the lanes are stacked UNDER the hub, so three threads cannot fan out sideways
     without cutting across every lane above their own.  routed down the left margin (the
     first try) they swung out past the canvas edge and ran behind the input labels, and read
     as tangle.  on a phone the relation is said once: ONE arrow from the implicit model down
     into the stack of models materialized from it. */
  if (L.threads === "arrow") {
    const from = new T.Vector3(HUB_X, HUB_Y - 1.0 * HUB_S - 0.06, 0.05);
    const to = new T.Vector3(HUB_X, LANE_Y[0] + 0.62, 0.05);
    world.add(new T.Mesh(new T.TubeGeometry(new T.LineCurve3(from, to), 8, 0.014, 8, false), threadMat));
    const head = new T.Mesh(new T.ConeGeometry(0.06, 0.16, 14), threadMat);
    head.position.copy(to).add(new T.Vector3(0, 0.08, 0));
    head.rotation.z = Math.PI;                                  /* pointing down */
    world.add(head);
  }
  if (L.threads === "fan") anchors.forEach((a) => {
    const c = new T.Vector3(MODEL_X, a.y, 0.05);
    const tip = c.clone().add(new T.Vector3(ux * rr, uy * rr, 0));
    const k = (tip.x - T_FROM.x) * 0.5;
    const curve = new T.CubicBezierCurve3(T_FROM.clone(),
      new T.Vector3(T_FROM.x + k, T_FROM.y, 0.05),
      tip.clone().add(new T.Vector3(ux * k * 0.8, uy * k * 0.8, 0)),   /* back out along the radius */
      tip);
    world.add(new T.Mesh(new T.TubeGeometry(curve, 48, 0.0085, 8, false), threadMat));
    const dir = curve.getTangent(1).normalize();
    const head = new T.Mesh(new T.ConeGeometry(0.036, 0.1, 12), threadMat);
    head.position.copy(tip).addScaledVector(dir, -0.05);   /* the cone is centred: tip ON the point */
    head.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), dir);
    world.add(head);
  });

  /* ---- the variables the implicit model carries, pinned to the brain itself ----
     each label is a point in the hub brain's LOCAL frame and is carried through its matrix
     every frame, so when the brain turns the labels turn with it -- the ones swinging round
     the back fade, the ones coming forward sharpen.  four hand-placed names in screen space
     said "here are some variables"; labels that orbit with the anatomy say they are fields
     defined OVER that anatomy, which is what a component of the implicit model is.

     the sites are chosen by FARTHEST-POINT sampling over the cortical nodes, so they spread
     across the whole surface instead of clumping where the node density is highest.  the
     sample is deterministic -- seeded from node 0, no random draw -- so the figure is the
     same on every load (CLAUDE.md: a function should return the same answer twice).

     the ids are taken across fields in round-robin, not in registry order: in order, the
     first twenty-six would all be components of one field.  a component is a field over the
     whole brain, so a label marks a place to READ it, not the only place it lives. */
  const R = window.IBM_REGISTRY;
  if (R && R.fields) {
    const perField = R.fields.map((f) => (f.components || []).map((c) => c.id));
    const ids = [];
    for (let k = 0; ids.length < 26 && perField.some((l) => l.length > k); k++)
      perField.forEach((l) => { if (l[k] && ids.length < 26) ids.push(l[k]); });
    const pick = [0], dmin = new Float32Array(CORTEX).fill(Infinity);
    const d2 = (i, j) => { const dx = pos[i * 3] - pos[j * 3], dy = pos[i * 3 + 1] - pos[j * 3 + 1],
      dz = pos[i * 3 + 2] - pos[j * 3 + 2]; return dx * dx + dy * dy + dz * dz; };
    while (pick.length < ids.length) {
      const last = pick[pick.length - 1];
      let far = -1, fd = -1;
      for (let i = 0; i < CORTEX; i++) {
        dmin[i] = Math.min(dmin[i], d2(i, last));
        if (dmin[i] > fd) { fd = dmin[i]; far = i; }
      }
      pick.push(far);
    }
    ids.forEach((id, k) => {
      const i = pick[k];
      /* 1.06x out from the centre so the text sits just off the surface, not inside it */
      const local = new T.Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]).multiplyScalar(1.06);
      addMark(local, "is-var", id, "m", 0, implicit);
    });
  }
  /* the wordmark sits ON the crown of the implicit brain, the way each model's name sits
     on its own arrow -- the two labels that name a thing now behave the same way.  the
     caption under it ("one parameter set · every variable") is gone: the ring of variable
     names around the hub already says it, and the standfirst says it in words. */
  addMark(new T.Vector3(HUB_X, HUB_Y + 0.58 * HUB_S, 0), "is-hub", "<b>IBM-1</b>", "c", 4);

  anchors.forEach((a) => {
    /* ON the arrow, not floating above it: at +0.40 the name sat halfway to the lane above
       and it was genuinely ambiguous which brain it belonged to.  anchored bottom-centre at
       the lane's own y, with a 3px nudge, its BASELINE lands on the arrow and the label
       overlaps the brain -- which is what makes it read as that brain's name. */
    addMark(new T.Vector3(MODEL_X, a.y, 0), "is-name", `<b>${a.model.name}</b>`, "c", 3);
    /* the ports themselves sit in a fixed column at the arrow's end rather than riding
       their own site in the brain.  a riding label swung with the brain's own turn and
       two of them crossed the arrow every few seconds; the colour and the side already
       say which end of the trace a port belongs to. */
    const column = (list, cls, x) => {
      const n = (list || []).length;
      (list || []).forEach((port, i) => {
        addMark(new T.Vector3(x, portY(a, n, i), 0), cls, `<b>${port.label}</b>`,
                cls === "is-in" ? "r" : "l");
      });
    };
    column(a.m.inputs, "is-in", TAIL - 0.04);
    column(a.m.outputs, "is-out", HEAD + 0.15);
  });

    spinners = [{ g: implicit, r: HUB_S }].concat(anchors.map((a) => ({ g: a.group, r: SUB })));
    spinners.forEach((s2) => { s2.ry = 0; s2.rx = 0; s2.auto = 1; });
    host.dataset.layout = name;
  }

  /* ---- camera, render ---- */
  const cam = new T.PerspectiveCamera(30, 1, 0.1, 60);
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const canvasHost = document.createElement("div");
  canvasHost.className = "mz-canvas";
  host.insertBefore(canvasHost, layer);
  canvasHost.appendChild(renderer.domElement);

  /* the camera does not move.  each brain turns on its own axis, and a drag turns
     whichever brain it started nearest -- the diagram is four objects, not one scene the
     reader spins. */
  let dist = 10;
  let drag = null;
  function place() {
    cam.position.set(0, L.CAM_Y, dist);
    cam.lookAt(0, L.CAM_Y, 0);
  }
  /* pick by projected distance: a raycast would need colliders on a point cloud */
  function pick(cx, cy) {
    const b = canvasHost.getBoundingClientRect();
    const x = cx - b.left, y = cy - b.top;
    let best = null, bd = Infinity;
    spinners.forEach((s2) => {
      const v = s2.g.position.clone().project(cam);
      const px = (v.x + 1) / 2 * b.width, py = (-v.y + 1) / 2 * b.height;
      const d = Math.hypot(px - x, py - y);
      if (d < bd) { bd = d; best = s2; }
    });
    return bd < 190 ? best : null;
  }
  function resize() {
    const w = host.clientWidth;
    const want = w < TALL_BELOW_PX ? "tall" : "wide";
    if (want !== layoutName) build(want);
    /* the canvas takes the layout's own aspect, so the fit below is height- and
       width-limited at once and neither half-extent is wasted */
    const h = want === "tall" ? Math.round(w * L.HALF_Y / L.HALF_X)
                              : Math.max(400, Math.round(w * 0.59));
    /* the type is in px and the scene is in world units, so a narrow canvas shrinks the
       brains and leaves the labels the size they were.  the trigger is the CANVAS's width,
       not the viewport's -- this figure sits in the second column of a two-column band, so
       at a 1120px viewport it is only 692px wide and a viewport media query never fires. */
    host.classList.toggle("is-tight", w < 820);
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
    /* fit BOTH half-extents.  fitting only the height let a short canvas crop the
       outputs off the right, which is the bug this layout exists to end. */
    dist = Math.max(L.HALF_Y, L.HALF_X / cam.aspect) / Math.tan((cam.fov * Math.PI / 180) / 2);
    measure();
  }
  /* label widths are read once per resize, not per frame: reading offsetWidth inside the
     render loop forces a layout on every mark, every frame. */
  const measure = () => marks.forEach((m) => { m.w = m.el.offsetWidth; });
  canvasHost.addEventListener("pointerdown", (e) => {
    const s2 = pick(e.clientX, e.clientY);
    if (!s2) return;
    drag = { x: e.clientX, y: e.clientY, ry: s2.ry, rx: s2.rx, s: s2 };
    s2.auto = 0;
    canvasHost.setPointerCapture(e.pointerId);
  });
  canvasHost.addEventListener("pointermove", (e) => {
    if (!drag) return;
    drag.s.ry = drag.ry + (e.clientX - drag.x) * 0.009;
    drag.s.rx = Math.max(-0.9, Math.min(0.9, drag.rx + (e.clientY - drag.y) * 0.006));
  });
  const up = () => { drag = null; };
  canvasHost.addEventListener("pointerup", up);
  canvasHost.addEventListener("pointercancel", up);

  let vis = true;
  if (window.IntersectionObserver)
    new IntersectionObserver((es) => { vis = es[0].isIntersecting; }, { rootMargin: "250px" })
      .observe(host);

  /* [x%, y%] applied after the pixel translate: c = centred above the point (a name over
     its arrow), b = centred below it (the hub's caption), m = centred on it, r/l = the
     label's right/left edge hung on it (an input column, an output column). */
  const ANCHOR = { c: [-50, -100], b: [-50, 0], m: [-50, -50], r: [-100, -50], l: [0, -50] };
  const t0 = performance.now();
  function frame() {
    requestAnimationFrame(frame);
    if (!vis) return;
    const tt = (performance.now() - t0) / 1000;
    spinners.forEach((s2, i) => {
      if (s2.auto) s2.ry = Math.sin(tt / 7 + i * 1.7) * 0.5;
      s2.g.rotation.set(s2.rx, s2.ry, 0);
    });
    place();
    renderer.render(scene, cam);
    const w = host.clientWidth, h = renderer.domElement.clientHeight;
    const hubC = implicit.getWorldPosition(new T.Vector3());
    marks.forEach((m) => {
      const wp = m.ride ? m.ride.localToWorld(m.p.clone()) : m.p.clone();
      /* a riding label's opacity follows which way its site FACES: the camera is on +z,
         so the z of (site - brain centre) says front or back.  hard-hiding the back ones
         made labels blink out mid-turn; a ramp lets them fade as they go round. */
      let alpha = 1;
      if (m.ride) {
        const f = wp.clone().sub(hubC).normalize().z;
        alpha = Math.max(0.05, Math.min(0.8, 0.36 + f * 0.8));
      }
      const v = wp.project(cam);
      let x = (v.x + 1) / 2 * w;
      const y = (-v.y + 1) / 2 * h;
      /* clamp into the canvas by the mark's own anchor side, so a long name or a long
         port label shortens its margin instead of leaving the picture */
      const a = ANCHOR[m.anchor], lead = -a[0] / 100 * m.w;
      x = Math.max(lead + 2, Math.min(w - (m.w - lead) - 2, x));
      m.el.style.opacity = v.z < 1 ? alpha.toFixed(2) : "0";
      m.el.style.transform =
        `translate(${x.toFixed(0)}px, ${(y + m.dy).toFixed(0)}px) translate(${a[0]}%, ${a[1]}%)`;
    });
  }
  resize();
  window.addEventListener("resize", resize);
  frame();
  /* one more measure after the fonts land: a label measured before Source Serif 4 has
     loaded is measured in the fallback and clamps against the wrong width */
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);
  host.classList.add("is-ready");
})();
