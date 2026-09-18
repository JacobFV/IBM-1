/* the whole-body figure: every system the atlas ships, nested on the joint tree of the
   driven skeleton, standing in grass, with a pulse running muscle -> nerve -> cortex -> back.

   ------------------------------------------------------------------------------------
   THE ONE THING A READER OF THIS FILE HAS TO KNOW.  the motion below is SYNTHESISED HERE,
   in this file, by hand.  it is a preview of what the station is FOR; it is not a solved
   trajectory and it is not a controller's output.  the honest reason is recorded in
   docs/LOG.md and was the reason the previous version of this figure did not move at all:
   every trajectory this body model has actually solved is outside the model's own declared
   joint ranges -- crawl-best's left knee is past its limit for 95.6% of 1,600 frames, the
   ankles for 79-94% -- so rigid-driving the anatomy with one dislocates the knees and the
   ankles on screen.  a figure that tears itself apart is a worse lie than a figure that is
   openly a preview.

   the admissible motion source is CMU mocap (the station's own feed line names it).  it is
   fetched and NOT yet bound to this skeleton; when it is, `poseFromClip` is the only
   function here that has to change -- it produces a vector of OpenSim coordinate values,
   which is exactly what a retargeted mocap frame would produce.

   what IS real here: the surfaces (Z-Anatomy, CC-BY-SA), the segment binding, the joint
   pivots (found where two segments' bone surfaces meet -- see the exporter), the declared
   joint ranges (read out of IHM-1's own engineering_stance_v1 model.osim and shipped in
   the payload), and the nerves -- Z-Anatomy's own spinal nerves and cord path, with a pulse
   that only runs along routes whose named nerves match the textbook sequence (see below).
   EVERY angle this file commands is clamped to the shipped range before it is applied, and
   a clamp that ever bites logs a warning, because a preview that needed clamping is a
   preview that was authored wrong.
   ------------------------------------------------------------------------------------ */
(function () {
  const host = document.getElementById("body3d");
  const D = window.IBM_BODY;
  if (!host || !D || !window.THREE || !D.meshes) return;
  const T = window.THREE;
  const css = getComputedStyle(document.documentElement);
  const isDark = () => !(document.documentElement.getAttribute("data-theme") === "light" ||
    (!document.documentElement.getAttribute("data-theme") &&
      !window.matchMedia("(prefers-color-scheme: dark)").matches));

  const scene = new T.Scene();
  const cam = new T.PerspectiveCamera(30, 1, 0.05, 40);
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);

  /* three lights rather than one: a key to shape the muscle bellies, a cool fill so the
     shadowed side does not go black against a dark page, and a rim to lift the silhouette
     off the background.  the hemisphere's ground colour is the grass, so the underside of
     the figure picks up a little green and stands ON the patch rather than above it. */
  scene.add(new T.HemisphereLight(0xcfe0ff, 0x35431f, 0.62));
  const key = new T.DirectionalLight(0xfff0dd, 1.32); key.position.set(2.4, 3.2, 2.2); scene.add(key);
  const fill = new T.DirectionalLight(0x93b4ff, 0.40); fill.position.set(-2.6, 0.6, 1.4); scene.add(fill);
  const rim = new T.DirectionalLight(0xffc98a, 0.78); rim.position.set(-1.2, 1.4, -3.0); scene.add(rim);

  const root = new T.Group();                  // recentred once the figure is built
  scene.add(root);
  const body = new T.Group();
  root.add(body);

  /* ---------------------------------------------------------------- the systems

     opacity is not a style choice here, it is the whole readability problem: thirteen
     nested closed surfaces, and the one the eye needs -- the skin -- is the outermost.
     the rule is that a system is as transparent as the number of systems INSIDE it.  bone
     is fully opaque and writes depth, so everything deeper than bone is simply occluded
     and never fights for the blend; muscle is half, because bone has to read through it;
     skin is a quarter, because everything does.  `depthWrite: false` on every transparent
     system is what stops a near layer from culling a far one before it can blend.

     three sorts transparent meshes back-to-front WITHIN a renderOrder, so the ordering
     below is the nesting order of the body and the sorter does the rest.  the skin is the
     exception: it is one closed surface per segment, so it self-occludes, and the two-pass
     back-faces-then-front-faces trick is the only way to blend a closed surface correctly
     without a depth prepass. */
  const SYS = {
    skeletal:      { op: 1.00, ro: 0, rough: 0.58 },
    endocrine:     { op: 0.95, ro: 4, rough: 0.45 },
    urinary:       { op: 0.88, ro: 4, rough: 0.5 },
    cardiac:       { op: 0.90, ro: 5, rough: 0.5 },
    digestive:     { op: 0.82, ro: 5, rough: 0.55 },
    reproductive:  { op: 0.82, ro: 5, rough: 0.55 },
    respiratory:   { op: 0.66, ro: 6, rough: 0.6 },
    arterial:      { op: 0.86, ro: 7, rough: 0.45 },
    venous:        { op: 0.80, ro: 7, rough: 0.45 },
    lymphatic:     { op: 0.55, ro: 8, rough: 0.6 },
    connective:    { op: 0.42, ro: 9, rough: 0.7 },
    muscular:      { op: 0.56, ro: 12, rough: 0.78 },
    integumentary: { op: 0.26, ro: 20, rough: 0.72, skin: true },
  };
  const colourOf = {};
  (D.systems || []).forEach((s) => { colourOf[s.id] = s.color; });

  /* one group per segment, nested on the exporter's parent table.  a mesh's vertices are
     already in its own segment's frame (the exporter subtracts the pivot), so a joint is a
     rotation of a group about its own origin and nothing touches a vertex buffer ever
     again -- 119 merged buffers animate on 22 quaternions. */
  const G = {};
  (D.segments || []).forEach((s) => { G[s] = new T.Group(); });
  (D.segments || []).forEach((s) => {
    const p = D.parents[s], pv = D.pivots[s] || [0, 0, 0], pp = (p && D.pivots[p]) || [0, 0, 0];
    G[s].position.set(pv[0] - pp[0], pv[1] - pp[1], pv[2] - pp[2]);
    (p && G[p] ? G[p] : body).add(G[s]);
  });

  const mats = {};
  function matFor(sysid, side) {
    const k = sysid + "|" + side;
    if (mats[k]) return mats[k];
    const cfg = SYS[sysid] || { op: 0.6, ro: 8, rough: 0.7 };
    const m = new T.MeshStandardMaterial({
      color: new T.Color(colourOf[sysid] || "#999"),
      roughness: cfg.rough, metalness: 0.0,
      transparent: cfg.op < 1, opacity: cfg.op,
      depthWrite: cfg.op >= 1, side: side,
    });
    mats[k] = m;
    return m;
  }

  (D.meshes || []).forEach((m) => {
    const cfg = SYS[m.sys] || { op: 0.6, ro: 8 };
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.Float32BufferAttribute(m.v, 3));
    geo.setIndex(m.i);
    geo.computeVertexNormals();
    const parent = G[m.seg] || body;
    if (cfg.skin) {
      const back = new T.Mesh(geo, matFor(m.sys, T.BackSide));
      back.renderOrder = cfg.ro; parent.add(back);
      const front = new T.Mesh(geo, matFor(m.sys, T.FrontSide));
      front.renderOrder = cfg.ro + 2; parent.add(front);
    } else {
      const mesh = new T.Mesh(geo, matFor(m.sys, T.FrontSide));
      mesh.renderOrder = cfg.ro; parent.add(mesh);
    }
  });

  /* ------------------------------------------------------------ the cortex it reports to

     site/data/graph.js is the mne `sample` subject's pial surface in surface RAS.  it is a
     DIFFERENT HEAD from this atlas's and it is not coregistered with it; what happens below
     is a bounding-box placement into the skull the exporter measured, which is a picture of
     a cortex in the right place and nothing more.  it rides the torso group, so it turns
     with the trunk, and it is the thing the afferent pulse arrives at.

     the units caught me: graph.js is surface RAS in MILLIMETRES and everything else on this
     figure is metres, so the first attempt scaled a 14 cm cortex by a factor meant for
     metres and hung it round the head like a halo.  the fit below is a ratio of two
     measured boxes and is therefore unit-free, which is the only version of this that
     cannot drift. */
  let cortex = null;
  (function buildCortex() {
    const GR = window.IBM_GRAPH, hb = D.headBox;
    if (!GR || !hb || !GR.nodes) return;
    const n = (GR.cortex_range && GR.cortex_range[1]) || 0;
    const step = Math.max(1, Math.round(n / 900));
    const pts = [], cols = [];
    for (let i = 0; i < n; i += step) {
      const nd = GR.nodes[i];
      if (!nd || !nd.p) continue;
      pts.push(-nd.p[0], nd.p[2], nd.p[1]);        // surface RAS -> x left, y up, z anterior
      const c = new T.Color(nd.c || "#8ab");
      cols.push(c.r, c.g, c.b);
    }
    if (pts.length < 30) return;
    const g = new T.BufferGeometry();
    g.setAttribute("position", new T.Float32BufferAttribute(pts, 3));
    g.setAttribute("color", new T.Float32BufferAttribute(cols, 3));
    g.computeBoundingBox();
    const bb = g.boundingBox, sz = bb.getSize(new T.Vector3()), c0 = bb.getCenter(new T.Vector3());
    // the head box is the upper skull the exporter measured; a brain sits inside it with a
    // little clearance for the vault, and lower than its middle, because the box's top is
    // bone and its bottom is roughly the orbits.
    const want = new T.Vector3(hb[1][0] - hb[0][0], hb[1][1] - hb[0][1], hb[1][2] - hb[0][2]);
    const s = Math.min(want.x * 0.92 / sz.x, want.z * 0.90 / sz.z);
    cortex = new T.Points(g, new T.PointsMaterial({
      // the skull is opaque and writes depth, so a depth-tested cortex is simply not
      // there.  drawing it after everything with the test off puts it visibly inside the
      // head, which is the only reading that is both legible and not a lie about occlusion.
      size: 0.004, sizeAttenuation: true, vertexColors: true,
      transparent: true, opacity: 0.34, depthWrite: false, depthTest: false,
    }));
    cortex.scale.setScalar(s);
    const tp = D.pivots.torso || [0, 0, 0];
    const home = new T.Vector3(0, (hb[0][1] + hb[1][1]) / 2 - want.y * 0.12,
                               (hb[0][2] + hb[1][2]) / 2 - want.z * 0.04);
    cortex.position.set(home.x - tp[0] - c0.x * s,
                        home.y - tp[1] - c0.y * s,
                        home.z - tp[2] - c0.z * s);
    cortex.renderOrder = 30;
    (G.torso || body).add(cortex);
  })();

  /* ------------------------------------------------------------------ the joint ranges

     every coordinate the preview commands is clamped to what the model declares.  six of
     them -- the three shoulder angles per side -- are declared [-10, 10] rad in that file,
     which is not a range, it is the absence of one; the exporter flags them and the
     substitutes below are plainly labelled so nothing here claims a bound the model gave.
     CLAUDE.md's standing correction on this: two artefacts that must be compared have to be
     asked what each one DECLARES about the shared quantity, not assumed to agree. */
  const SUBSTITUTE = {           // conventional anatomical limits, NOT from the .osim
    arm_flex: [-1.05, 3.05], arm_add: [-2.60, 0.52], arm_rot: [-1.57, 1.57],
  };
  const RANGE = {};
  for (const k in (D.ranges || {})) {
    const r = D.ranges[k];
    if (r.declared) { RANGE[k] = [r.lo, r.hi, true]; continue; }
    const base = k.replace(/_[lr]$/, "");
    RANGE[k] = SUBSTITUTE[base] ? [SUBSTITUTE[base][0], SUBSTITUTE[base][1], false] : [-3.2, 3.2, false];
  }
  let clipped = 0;
  function gated(name, q) {
    const r = RANGE[name];
    if (!r) return q;
    if (q < r[0] || q > r[1]) {
      if (clipped++ === 0)
        console.warn("body.js: the preview asked for " + name + " = " + q.toFixed(3) +
                     ", outside " + JSON.stringify(r.slice(0, 2)) + " -- the CLIP is wrong, " +
                     "not the clamp; fix the clip rather than widening this.");
      return Math.max(r[0], Math.min(r[1], q));
    }
    return q;
  }

  /* which rotation each coordinate is, in the atlas frame (x to the subject's LEFT, y up,
     z anterior).  the sign column is the part that is easy to get wrong and impossible to
     get wrong quietly -- a flipped knee bends forwards -- and it is why every leg joint was
     checked against a screenshot rather than against the arithmetic.  patella takes half
     the knee angle because the kneecap tracks the joint rather than swinging with the
     shank; it is not a degree of freedom of its own. */
  const DOF = {
    torso: [["lumbar_extension", "x", -1], ["lumbar_bending", "z", 1], ["lumbar_rotation", "y", 1]],
    femur_l: [["hip_flexion_l", "x", -1], ["hip_adduction_l", "z", -1], ["hip_rotation_l", "y", 1]],
    femur_r: [["hip_flexion_r", "x", -1], ["hip_adduction_r", "z", 1], ["hip_rotation_r", "y", -1]],
    tibia_l: [["knee_angle_l", "x", 1]], tibia_r: [["knee_angle_r", "x", 1]],
    patella_l: [["knee_angle_l", "x", 0.5]], patella_r: [["knee_angle_r", "x", 0.5]],
    talus_l: [["ankle_angle_l", "x", -1]], talus_r: [["ankle_angle_r", "x", -1]],
    toes_l: [["mtp_angle_l", "x", -1]], toes_r: [["mtp_angle_r", "x", -1]],
    humerus_l: [["arm_flex_l", "x", -1], ["arm_add_l", "z", -1], ["arm_rot_l", "y", 1]],
    humerus_r: [["arm_flex_r", "x", -1], ["arm_add_r", "z", 1], ["arm_rot_r", "y", -1]],
    ulna_l: [["elbow_flex_l", "x", -1]], ulna_r: [["elbow_flex_r", "x", -1]],
    radius_l: [["pro_sup_l", "y", 1]], radius_r: [["pro_sup_r", "y", -1]],
  };
  const AXIS = { x: new T.Vector3(1, 0, 0), y: new T.Vector3(0, 1, 0), z: new T.Vector3(0, 0, 1) };
  const qtmp = new T.Quaternion(), qacc = new T.Quaternion();

  function applyPose(q) {
    for (const seg in DOF) {
      if (!G[seg]) continue;
      qacc.identity();
      const dofs = DOF[seg];
      for (let i = 0; i < dofs.length; i++) {
        const name = dofs[i][0];
        let v = q[name];
        if (v === undefined) continue;
        v = gated(name, v) * dofs[i][2];
        qtmp.setFromAxisAngle(AXIS[dofs[i][1]], v);
        qacc.multiply(qtmp);
      }
      G[seg].quaternion.copy(qacc);
    }
    // the pelvis is the free body: three angles the model does declare, and a vertical
    // bob.  no forward translation -- the figure walks on the spot, because a figure that
    // strides out of frame is a camera problem, not a locomotion claim.
    qacc.identity();
    qtmp.setFromAxisAngle(AXIS.x, -gated("pelvis_tilt", q.pelvis_tilt || 0)); qacc.multiply(qtmp);
    qtmp.setFromAxisAngle(AXIS.z, gated("pelvis_list", q.pelvis_list || 0)); qacc.multiply(qtmp);
    qtmp.setFromAxisAngle(AXIS.y, gated("pelvis_rotation", q.pelvis_rotation || 0)); qacc.multiply(qtmp);
    body.quaternion.copy(qacc);
    body.position.set(q._tx || 0, q._ty || 0, q._tz || 0);
  }

  /* ------------------------------------------------------------------- the preview clips

     three hand-authored cycles, in OpenSim coordinate values.  the shapes are the textbook
     ones -- the knee flexes in swing and is near-straight through stance, the arms swing
     out of phase with the legs, the reach leads with the trunk -- and their amplitudes were
     chosen to sit comfortably inside the declared ranges rather than at them.  none of this
     is measured and none of it is claimed to be.  a mocap clip retargeted to these same
     coordinate names drops straight in here. */
  const TAU = Math.PI * 2;
  const smooth = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
  const bump = (t) => Math.sin(Math.PI * Math.max(0, Math.min(1, t)));

  function walk(t, q) {
    const P = 1.15;                       // one gait cycle
    const ph = { r: (t / P) * TAU, l: (t / P) * TAU + Math.PI };
    for (const s of ["l", "r"]) {
      const f = ph[s], o = ph[s === "l" ? "r" : "l"];
      const swing = Math.max(0, Math.sin(f + 0.55));
      q["hip_flexion_" + s] = 0.30 * Math.sin(f) + 0.14;
      q["knee_angle_" + s] = 0.12 + 0.98 * swing * swing;
      q["ankle_angle_" + s] = -0.16 * Math.sin(f + 1.1) + 0.04;
      q["mtp_angle_" + s] = 0.22 * Math.max(0, -Math.sin(f + 1.6));
      q["hip_adduction_" + s] = 0.025;
      q["arm_flex_" + s] = 0.52 * Math.sin(o);      // arms swing against the legs
      q["elbow_flex_" + s] = 0.42 + 0.26 * Math.max(0, Math.sin(o));
      q["arm_add_" + s] = -0.09;
    }
    q.lumbar_rotation = 0.06 * Math.sin(ph.r);
    q.lumbar_bending = 0.025 * Math.sin(ph.r);
    q.lumbar_extension = -0.06;
    q.pelvis_list = 0.05 * Math.sin(ph.r);
    q.pelvis_rotation = 0.09 * Math.sin(ph.r);
    q.pelvis_tilt = 0.06;
    // the pelvis rises twice per cycle -- once per step -- and is LOWEST at double support,
    // which is the way round real gait goes.  the first version had it backwards and sank
    // the stance foot through the grass at mid-stance, where it should be at its highest.
    q._ty = -0.013 * Math.cos(2 * ph.r) - 0.008;
    return q;
  }

  function reach(t, q) {
    const s = smooth(t / 1.5) * (1 - smooth((t - 2.6) / 1.4));   // out, hold, back
    q.arm_flex_r = 1.48 * s; q.arm_add_r = -0.38 * s; q.arm_rot_r = -0.25 * s;
    q.elbow_flex_r = 0.95 - 0.78 * s; q.pro_sup_r = 0.35 * s;
    q.arm_flex_l = -0.22 * s; q.arm_add_l = -0.12; q.elbow_flex_l = 0.30 + 0.25 * s;
    q.lumbar_rotation = -0.20 * s; q.lumbar_extension = -0.16 * s; q.lumbar_bending = -0.05 * s;
    q.pelvis_rotation = -0.10 * s; q.pelvis_tilt = 0.05 + 0.10 * s;
    for (const k of ["l", "r"]) {
      q["hip_flexion_" + k] = 0.10 + 0.20 * s;
      q["knee_angle_" + k] = 0.08 + 0.26 * s;
      q["ankle_angle_" + k] = 0.06 + 0.14 * s;
      q["hip_adduction_" + k] = 0.02;
    }
    q._ty = -0.035 * s;
    return q;
  }

  function stand(t, q) {
    const b = Math.sin((t / 2.4) * TAU);          // a slow weight shift
    const br = Math.sin((t / 3.6) * TAU);         // and a breath in the trunk
    q.lumbar_bending = 0.05 * b; q.lumbar_extension = -0.04 + 0.025 * br;
    q.lumbar_rotation = 0.03 * b;
    q.pelvis_list = -0.03 * b; q.pelvis_tilt = 0.04;
    for (const s of ["l", "r"]) {
      const sg = s === "l" ? 1 : -1;
      q["hip_flexion_" + s] = 0.07 + 0.02 * b * sg;
      q["knee_angle_" + s] = 0.09 + 0.05 * Math.max(0, b * sg);
      q["ankle_angle_" + s] = 0.05; q["hip_adduction_" + s] = 0.02 * sg * b;
      q["arm_flex_" + s] = 0.04 * br; q["arm_add_" + s] = -0.10;
      q["elbow_flex_" + s] = 0.22 + 0.05 * br;
    }
    q._ty = 0.004 * br;
    return q;
  }

  const CLIPS = [
    { f: stand, dur: 5.2, nerves: /./ },
    { f: walk, dur: 8.05, nerves: /Tibial|fibular|Femoral/ },
    { f: stand, dur: 3.4, nerves: /./ },
    { f: reach, dur: 4.6, nerves: /(Median|Radial) nerve\.r/ },
  ];
  /* THE CHECK THAT MAKES THE CLAIM ABOVE WORTH ANYTHING.  "every angle is inside the model's
     declared range" is only true if someone looked, so this looks: it sweeps all four clips
     end to end at 60 Hz before the first frame is drawn and tests every value each one emits
     against the range the payload shipped.  it costs about two milliseconds and it prints ONE
     line, which names the coordinates whose bound is a substitute rather than the model's --
     the six shoulder angles the .osim declares as [-10, 10] rad, which is not a range.
     a runtime check is here rather than in a test because the clips are edited here and a
     check that lives somewhere else is a check that gets edited out of step. */
  function verifyClips() {
    const q = {}, bad = [], used = {};
    let n = 0;
    for (let i = 0; i < CLIPS.length; i++)
      for (let t = 0; t <= CLIPS[i].dur; t += 1 / 60) {
        for (const k in q) delete q[k];
        CLIPS[i].f(t, q); n++;
        for (const k in q) {
          if (k.charAt(0) === "_") continue;
          const r = RANGE[k];
          if (!r) continue;
          used[k] = r[2];
          if (q[k] < r[0] - 1e-9 || q[k] > r[1] + 1e-9)
            bad.push(k + "=" + q[k].toFixed(3) + " outside " + r[0].toFixed(2) + ".." + r[1].toFixed(2));
        }
      }
    const sub = Object.keys(used).filter((k) => !used[k]);
    console.log("body.js: " + n + " preview frames over " + Object.keys(used).length +
      " coordinates, " + (bad.length ? bad.length + " OUT OF RANGE: " + bad.slice(0, 4).join("; ")
                                     : "all inside the ranges " + D.rangeSource + " declares") +
      (sub.length ? " (" + sub.length + " of them on a substituted bound, because the model " +
        "declares no real one: " + sub.join(", ") + ")" : ""));
  }
  verifyClips();

  const FADE = 0.75;
  /* `#clip=2` in the URL pins one clip and stops the cycle.  it stays because the walk is
     five seconds into a twenty-one second loop and two rounds of screenshots went by before
     anyone realised they had all landed on the standing clip. */
  const PIN = (location.hash.match(/clip=(\d)/) || [])[1];
  let cur = { i: Math.min(CLIPS.length - 1, +(PIN || 0)), t: 0 }, prev = null, fade = 0;
  const qa = {}, qb = {}, qm = {};

  function poseFromClip(dt) {
    cur.t += dt;
    if (prev) { prev.t += dt; fade -= dt; if (fade <= 0) prev = null; }
    if (cur.t >= CLIPS[cur.i].dur && !prev) {
      prev = cur; fade = FADE;
      cur = { i: PIN ? cur.i : (cur.i + 1) % CLIPS.length, t: 0 };
    }
    for (const k in qa) delete qa[k];
    for (const k in qb) delete qb[k];
    for (const k in qm) delete qm[k];
    CLIPS[cur.i].f(cur.t, qa);
    if (!prev) return qa;
    CLIPS[prev.i].f(prev.t, qb);
    const w = 1 - fade / FADE, u = smooth(w);
    for (const k in qb) qm[k] = qb[k];
    for (const k in qa) qm[k] = (qm[k] === undefined ? 0 : qm[k]) * (1 - u) + qa[k] * u;
    for (const k in qm) if (qa[k] === undefined) qm[k] = qm[k] * (1 - u);
    return qm;
  }

  /* ------------------------------------------------------- the nerves, and what runs in them

     these are Z-Anatomy's own spinal nerves -- the centrelines the atlas authors them as --
     and the spinal cord drawn along the centreline of the atlas's spinal dura, extracted from
     the staged source by scripts/blender_zanatomy_nerves.py.  the exporter's header says why
     IHM-1's display export had left them out (a licence exclusion scoped to the inner ear and
     the kidney, applied to the whole collection) and why taking the spinal nerves stays inside
     it.  the tube around each centreline is a DISPLAY radius, at or below the real calibre;
     the source's own bevel is half a millimetre and would not be visible at this size.

     WHAT THIS REPLACED, because it was worse than nothing: glowing beads riding straight
     three-point lines from IHM-1's peripheral_display.json -- cortex, a spinal relay, a
     muscle -- which that file itself labels `schematic_anatomical_prior`.  on screen they
     cut across the body in straight chords that no nerve follows, and read as wiring.

     a nerve crosses joints, so it cannot ride one segment the way a bone does.  every
     centreline point is bound to its nearest bone surface, blended across a joint with the
     adjacent segment, and the whole network is ONE SkinnedMesh on 22 bones that mirror the
     segment groups -- built once at rest, bent by the GPU every frame, so the only per-frame
     upload is the colour.

     the pulse is a band of vertex colour moving along the nerve's own arc length, not an
     object on top of it.  it only runs along pathways the exporter GATED: a shortest path
     through the atlas's network, from a named nerve's far end to the top of the cord, kept
     only if the named nerves it passes are the textbook sequence (tibial -> sciatic -> cauda
     -> cord, median -> plexus -> roots -> cord, and so on).  the ulnar and musculocutaneous
     routes failed that gate -- the geometry sends them to the cord through the long thoracic
     nerve -- so they are drawn and never pulsed.  afferent runs up in the site's input
     colour, the cortex brightens when it arrives, and the efferent answer runs back down in
     the output colour.  the speed is a depiction: real conduction is 50-70 m/s and would cross
     the body in a frame.  it ran at 1.15 m/s (slowed ~50x) and read as sap rising, not as a
     nerve firing; at 5 m/s (~12x slow) a leg nerve is crossed in ~0.2 s, which is fast enough
     to read as a signal and slow enough that the band is still seen to TRAVEL. */
  const AFF = new T.Color(css.getPropertyValue("--fig-in").trim() || "#bd2f3b");
  const EFF = new T.Color(css.getPropertyValue("--fig-out").trim() || "#6d34c0");
  const NV = D.nerves;
  const K = 6;                                   // vertices round each ring
  let nerveMesh = null, nerveCol = null, nerveHot = null, nerveBase = null, bones = null;
  const npts = NV ? NV.p.length / 3 : 0;
  const paths = [];
  (function buildNerves() {
    if (!NV || !npts) return;
    const P = NV.p, pos = new Float32Array(npts * K * 3), nrm = new Float32Array(npts * K * 3);
    const si = new Uint16Array(npts * K * 4), sw = new Float32Array(npts * K * 4);
    const idx = [];
    const t = new T.Vector3(), n = new T.Vector3(), b = new T.Vector3(), q = new T.Vector3(),
          a0 = new T.Vector3(), a1 = new T.Vector3(), up = new T.Vector3();
    for (const [start, count, r] of NV.chains) {
      n.set(0, 0, 0);
      for (let i = 0; i < count; i++) {
        const g = start + i, lo = start + Math.max(0, i - 1), hi = start + Math.min(count - 1, i + 1);
        a0.fromArray(P, lo * 3); a1.fromArray(P, hi * 3);
        t.subVectors(a1, a0).normalize();
        // parallel transport: carry the previous ring's normal onto this ring's plane, so the
        // tube does not twist; seed it from whichever axis is least parallel to the nerve.
        if (i === 0 || n.lengthSq() < 1e-8) {
          up.set(Math.abs(t.y) < 0.9 ? 0 : 1, Math.abs(t.y) < 0.9 ? 1 : 0, 0);
          n.crossVectors(t, up).normalize();
        } else {
          n.addScaledVector(t, -n.dot(t)).normalize();
        }
        b.crossVectors(t, n);
        q.fromArray(P, g * 3);
        for (let k = 0; k < K; k++) {
          const th = (k / K) * Math.PI * 2, c = Math.cos(th), sn = Math.sin(th), v = g * K + k;
          const rx = c * n.x + sn * b.x, ry = c * n.y + sn * b.y, rz = c * n.z + sn * b.z;
          pos[v * 3] = q.x + r * rx; pos[v * 3 + 1] = q.y + r * ry; pos[v * 3 + 2] = q.z + r * rz;
          nrm[v * 3] = rx; nrm[v * 3 + 1] = ry; nrm[v * 3 + 2] = rz;
          si[v * 4] = NV.sa[g]; si[v * 4 + 1] = NV.sb[g];
          sw[v * 4] = NV.w[g] / 100; sw[v * 4 + 1] = 1 - NV.w[g] / 100;
        }
        if (i < count - 1)
          for (let k = 0; k < K; k++) {
            const a = g * K + k, bq = g * K + (k + 1) % K, c2 = a + K, d2 = bq + K;
            idx.push(a, c2, bq, bq, c2, d2);
          }
      }
    }
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.BufferAttribute(pos, 3));
    geo.setAttribute("normal", new T.BufferAttribute(nrm, 3));
    geo.setAttribute("skinIndex", new T.Uint16BufferAttribute(si, 4));
    geo.setAttribute("skinWeight", new T.Float32BufferAttribute(sw, 4));
    nerveCol = new T.BufferAttribute(new Uint8Array(npts * K * 3), 3, true);
    nerveCol.setUsage(T.DynamicDrawUsage);
    geo.setAttribute("color", nerveCol);
    geo.setIndex(idx);
    nerveHot = new T.BufferAttribute(new Uint8Array(npts * K), 1, true);
    nerveHot.setUsage(T.DynamicDrawUsage);
    geo.setAttribute("aHot", nerveHot);
    nerveBase = new Uint8Array(npts * K * 3);
    const mat = new T.MeshLambertMaterial({
      vertexColors: true, skinning: true, transparent: true, opacity: 0.92,
      depthWrite: false, depthTest: false,
    });
    /* the first build only RECOLOURED the vertices, and on a 2-3 mm tube behind two
       translucent layers a red stretch of a yellow nerve was invisible in six frames out of
       six.  so where the pulse is, the vertices also move: `aHot` pushes each ring out along
       its own normal (before skinning, so the swelling bends with the limb), and the lit
       colour gives way to the vertex colour itself, so the band reads as lit from inside
       rather than shaded.  it is the nerve's own vertices doing both -- nothing is drawn on
       top of it. */
    mat.onBeforeCompile = (sh) => {
      sh.vertexShader = "attribute float aHot;\nvarying float vHot;\n" + sh.vertexShader.replace(
        "#include <begin_vertex>",
        "vec3 transformed = vec3( position ) + normal * aHot * 0.0082;\n\tvHot = aHot;");
      sh.fragmentShader = "varying float vHot;\n" + sh.fragmentShader.replace(
        "gl_FragColor = vec4( outgoingLight, diffuseColor.a );",
        "outgoingLight = mix( outgoingLight, diffuseColor.rgb, vHot );\n\tgl_FragColor = vec4( outgoingLight, diffuseColor.a );");
    };
    nerveMesh = new T.SkinnedMesh(geo, mat);
    // drawn last and WITHOUT the depth test.  depth-tested, the cord vanished inside the
    // opaque vertebrae and the sciatic behind the femur -- which is most of every pulse's
    // journey -- and a pulse that disappears for two thirds of its run reads as a flicker.
    // the nerves are the system this figure is about, so they are drawn over it, the way an
    // atlas highlights one system through the others.
    nerveMesh.renderOrder = 25;
    nerveMesh.frustumCulled = false;
    root.add(nerveMesh);
    bones = (D.segments || []).map(() => new T.Bone());

    // each pathway's arc length from its far end, measured at rest along its own points
    for (const pth of NV.paths || []) {
      const cum = new Float32Array(pth.idx.length);
      for (let k = 1; k < pth.idx.length; k++) {
        a0.fromArray(P, pth.idx[k - 1] * 3); a1.fromArray(P, pth.idx[k] * 3);
        cum[k] = cum[k - 1] + a0.distanceTo(a1);
      }
      paths.push({ name: pth.name, idx: pth.idx, cum: cum, L: cum[cum.length - 1] });
    }
  })();

  /* bones mirror the segment groups: the skeleton reads nothing but each bone's matrixWorld,
     so copying the group's into it IS the pose.  the bones are deliberately not in the scene
     graph, where updateMatrixWorld would overwrite what is copied in. */
  function syncBones() {
    if (!bones) return;
    (D.segments || []).forEach((s, i) => { if (G[s]) bones[i].matrixWorld.copy(G[s].matrixWorld); });
  }
  function bindNerves() {                      // once, in the neutral pose, after the fit
    if (!nerveMesh) return;
    root.updateMatrixWorld(true);
    syncBones();
    nerveMesh.bind(new T.Skeleton(bones), nerveMesh.matrixWorld);
  }

  const baseColour = new T.Color();
  function paintBase() {
    if (!nerveBase) return;
    // the atlas's own convention: nerves are yellow.  deeper on bone paper so they hold
    // against it, lighter at night so they are not a dark line on a dark figure.
    baseColour.set(isDark() ? "#e7c55a" : "#c9981c");
    const r = baseColour.r * 255, g = baseColour.g * 255, b = baseColour.b * 255;
    for (let v = 0; v < nerveBase.length; v += 3) { nerveBase[v] = r; nerveBase[v + 1] = g; nerveBase[v + 2] = b; }
  }

  /* the band widens with the speed: at 5 m/s a 0.10 m band is on any one vertex for ~40 ms,
     under three frames, and it flickered instead of moving */
  const SPEED = 5.0, BAND = 0.15;                // m/s on screen; band half-width, metres
  const active = [];
  let fireAt = 0.8, cortexGlow = 0;
  function fire(path, dir) { active.push({ p: path, dir: dir, t: 0 }); }
  function schedulePulses(dt) {
    fireAt -= dt;
    if (fireAt > 0 || !paths.length || active.length > 6) return;
    fireAt = 0.35 + Math.random() * 0.35;        /* a pulse now lasts ~0.3 s, so fire more often */
    const want = CLIPS[cur.i].nerves;
    const pool = paths.filter((p) => want.test(p.name));
    const p = (pool.length ? pool : paths)[Math.floor(Math.random() * (pool.length || paths.length))];
    fire(p, -1);                                 // the muscle reports
  }
  const ctmp = new T.Color();
  function updatePulses(dt) {
    if (!nerveCol) return;
    cortexGlow = Math.max(0, cortexGlow - dt * 3.2);
    if (cortex) cortex.material.opacity = 0.34 + 0.5 * cortexGlow;
    const col = nerveCol.array, hotA = nerveHot.array;
    col.set(nerveBase);
    hotA.fill(0);
    for (let a = active.length - 1; a >= 0; a--) {
      const u = active[a];
      u.t += dt;
      const run = u.t * SPEED, L = u.p.L;
      // afferent (-1) travels far end -> cord top; efferent (+1) cord top -> far end
      const front = u.dir < 0 ? run : L - run;
      if (run > L + 0.5) {
        active.splice(a, 1);
        if (u.dir < 0) { cortexGlow = 1; window.setTimeout(() => fire(u.p, 1), 110); }
        continue;
      }
      ctmp.copy(u.dir < 0 ? AFF : EFF);
      const cum = u.p.cum, idx = u.p.idx;
      for (let k = 0; k < idx.length; k++) {
        const d = cum[k] - front;
        // a band at the front and a short fading wake behind it, on the side already passed
        const behind = u.dir < 0 ? -d : d;
        let s = Math.exp(-(d * d) / (BAND * BAND));
        if (behind > 0 && behind < 0.55) s = Math.max(s, 0.5 * (1 - behind / 0.55));
        if (s < 0.02) continue;
        const hot = Math.min(1, s * 1.2), lift = s > 0.5 ? (s - 0.5) * 1.1 : 0;
        const rr = (ctmp.r + lift) * 255, gg = (ctmp.g + lift) * 255, bb = (ctmp.b + lift) * 255;
        const h8 = Math.round(Math.min(1, s) * 255);
        for (let j = 0; j < K; j++) {
          const hv = idx[k] * K + j;
          if (h8 > hotA[hv]) hotA[hv] = h8;
          const v = hv * 3;
          col[v] = Math.min(255, col[v] + (rr - col[v]) * hot);
          col[v + 1] = Math.min(255, col[v + 1] + (gg - col[v + 1]) * hot);
          col[v + 2] = Math.min(255, col[v + 2] + (bb - col[v + 2]) * hot);
        }
      }
    }
    nerveCol.needsUpdate = true;
    nerveHot.needsUpdate = true;
  }

  /* -------------------------------------------------------------------------- the grass

     the patch has to end without ending: a disc with a hard rim is a diagram of a disc.
     so the ground is one CanvasTexture -- procedural, no binary asset in the repo -- whose
     ALPHA falls to zero before its geometry does, and the blades thin and shorten on the
     same falloff.  it feathers in material space rather than by masking the canvas,
     because the canvas is the whole figure: mask it and the feet go too. */
  /* the disc has to be SMALLER than the frame, not larger: a patch whose feather falls
     outside the canvas is cropped by the canvas and ends in a hard straight line at the
     left and right edges, which is exactly what the first run produced.  the camera frames
     about 1.8 m across at the figure, so the falloff has to be done by 0.9. */
  const GROUND_R = 0.78;
  function grassTexture(dark) {
    const S = 512, cv = document.createElement("canvas");
    cv.width = cv.height = S;
    const x = cv.getContext("2d");
    x.fillStyle = dark ? "#31402a" : "#58703f";
    x.fillRect(0, 0, S, S);
    for (let i = 0; i < 9000; i++) {                     // clumps, so it is not flat felt
      const a = Math.random() * TAU, r = Math.sqrt(Math.random()) * S * 0.5;
      const px = S / 2 + Math.cos(a) * r, py = S / 2 + Math.sin(a) * r;
      const l = (30 + Math.random() * 30) * (dark ? 0.6 : 1), h = 74 + Math.random() * 26;
      x.strokeStyle = "hsla(" + h + "," + (12 + Math.random() * 16) + "%," + l + "%,0.40)";
      x.lineWidth = 0.8 + Math.random() * 1.8;
      x.beginPath(); x.moveTo(px, py);
      x.lineTo(px + (Math.random() - 0.5) * 13, py + (Math.random() - 0.5) * 13);
      x.stroke();
    }
    // a soft contact shadow, so the figure sits IN the grass rather than on top of it
    const sh = x.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S * 0.15);
    sh.addColorStop(0, "rgba(0,0,0,.42)"); sh.addColorStop(1, "rgba(0,0,0,0)");
    x.fillStyle = sh; x.fillRect(0, 0, S, S);
    // and the feather: keep only what is inside the falloff.  `destination-in` multiplies
    // the alpha already there, so this is the one operation that fades the grass rather
    // than painting page-coloured paint over it -- which would only work on one theme.
    const fe = x.createRadialGradient(S / 2, S / 2, S * 0.1, S / 2, S / 2, S * 0.5);
    fe.addColorStop(0, "rgba(0,0,0,.9)"); fe.addColorStop(0.34, "rgba(0,0,0,.8)");
    fe.addColorStop(0.62, "rgba(0,0,0,.45)"); fe.addColorStop(0.84, "rgba(0,0,0,.13)");
    fe.addColorStop(1, "rgba(0,0,0,0)");
    x.globalCompositeOperation = "destination-in";
    x.fillStyle = fe; x.fillRect(0, 0, S, S);
    const tex = new T.CanvasTexture(cv);
    tex.anisotropy = 4;
    return tex;
  }

  const ground = new T.Mesh(
    new T.CircleGeometry(GROUND_R, 72).rotateX(-Math.PI / 2),
    new T.MeshStandardMaterial({ transparent: true, roughness: 0.95, metalness: 0,
                                 depthWrite: false, side: T.DoubleSide })
  );
  ground.renderOrder = -1;
  scene.add(ground);        // world space: the floor is found once, after the figure is fitted

  /* the blades are one InstancedMesh, so the whole lawn is a single draw call.  their
     density and height follow the same radial falloff as the texture's alpha -- a blade
     standing where the ground has already faded out is exactly the hard edge this was
     meant to avoid -- and each one carries its own colour, because a lawn of one green is
     a carpet. */
  const NB = 2200;
  const blade = new T.InstancedMesh(
    new T.PlaneGeometry(0.0055, 1, 1, 2).translate(0, 0.5, 0),
    new T.MeshStandardMaterial({ side: T.DoubleSide, roughness: 0.85, metalness: 0 }),
    NB
  );
  const bladeSeed = [];
  (function seedBlades() {
    const c = new T.Color();
    for (let i = 0; i < NB; i++) {
      const a = Math.random() * TAU;
      const r = Math.pow(Math.random(), 0.55) * GROUND_R;
      const fall = Math.max(0, 1 - Math.pow(r / GROUND_R, 1.7));
      bladeSeed.push({ x: Math.cos(a) * r, z: Math.sin(a) * r, yaw: Math.random() * TAU,
                       h: (0.011 + Math.random() * 0.026) * (0.15 + 0.85 * fall),
                       ph: Math.random() * TAU, keep: fall > 0.05 });
      c.setHSL(0.215 + Math.random() * 0.05, 0.14 + Math.random() * 0.16,
               0.36 + Math.random() * 0.20);
      blade.setColorAt(i, c);
    }
    if (blade.instanceColor) blade.instanceColor.needsUpdate = true;
  })();
  blade.frustumCulled = false;
  scene.add(blade);

  function themeScene() {
    const dark = isDark();
    if (ground.material.map) ground.material.map.dispose();
    ground.material.map = grassTexture(dark);
    ground.material.color.setHex(dark ? 0xb2c09c : 0xf6f8ee);
    ground.material.needsUpdate = true;
    blade.material.color.setHex(dark ? 0x93a682 : 0xeef3e2);   // multiplies the per-blade colour
    // the figure is lit for bone paper; at night the fill has to come up or the shadowed
    // half of the silhouette disappears into the page entirely.
    fill.intensity = dark ? 0.62 : 0.40;
    rim.intensity = dark ? 1.05 : 0.78;
    AFF.set(css.getPropertyValue("--fig-in").trim() || "#bd2f3b");
    EFF.set(css.getPropertyValue("--fig-out").trim() || "#6d34c0");
    paintBase();
  }
  themeScene();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", themeScene);
  new MutationObserver(themeScene).observe(document.documentElement,
    { attributes: true, attributeFilter: ["data-theme"] });

  /* --------------------------------------------------------------- fit, and then place

     the figure is fitted in its NEUTRAL pose (every coordinate zero, which is the pose the
     atlas surfaces were authored in and is inside every declared range).  fitting to a
     moving pose makes the frame breathe with the animation. */
  const dummy = new T.Object3D();
  let floorY = 0, dist = 3.2;
  (function fit() {
    applyPose({});
    body.position.set(0, 0, 0);
    root.position.set(0, 0, 0);
    ground.visible = blade.visible = false;
    root.updateMatrixWorld(true);
    const box = new T.Box3();
    root.traverse((o) => { if (o.isMesh) box.expandByObject(o); });
    ground.visible = blade.visible = true;
    const c = box.getCenter(new T.Vector3()), sz = box.getSize(new T.Vector3());
    root.position.set(-c.x, -c.y, -c.z);
    floorY = box.min.y - c.y;
    ground.position.y = floorY + 0.004;
    blade.position.y = floorY;
    const h = Math.max(sz.y, sz.x * 1.05);
    dist = (h / 2) / Math.tan((cam.fov * Math.PI / 180) / 2) * 1.12;
  })();
  bindNerves();
  paintBase();

  let az = -0.75, el = 0.10, drag = null, spin = true, last = performance.now();
  function place() {
    cam.position.set(Math.sin(az) * Math.cos(el) * dist, Math.sin(el) * dist + 0.04,
                     Math.cos(az) * Math.cos(el) * dist);
    cam.lookAt(0, 0.0, 0);
  }
  function resize() {
    const w = host.clientWidth, h = Math.max(340, Math.round(w * 1.06));
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
  }

  host.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, az: az, el: el };
    host.setPointerCapture(e.pointerId); spin = false;
  });
  host.addEventListener("pointermove", (e) => {
    if (!drag) return;
    az = drag.az - (e.clientX - drag.x) * 0.008;
    el = Math.max(-0.45, Math.min(0.65, drag.el + (e.clientY - drag.y) * 0.005));
  });
  const stop = () => { drag = null; };
  host.addEventListener("pointerup", stop);
  host.addEventListener("pointercancel", stop);

  let visible = true;
  if (window.IntersectionObserver)
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { rootMargin: "200px" }).observe(host);

  /* the sway re-uploads 2,200 instance matrices, so it runs at a third of the frame rate.
     grass at 20 Hz is indistinguishable from grass at 60 and the upload is 140 kB a go. */
  let swayT = 0, swayN = 0;
  function swayBlades(dt) {
    swayT += dt;
    if (swayN++ % 3) return;
    for (let i = 0; i < NB; i++) {
      const b = bladeSeed[i];
      if (!b.keep) { dummy.scale.set(0, 0, 0); dummy.position.set(0, -99, 0); }
      else {
        dummy.position.set(b.x, 0, b.z);
        dummy.rotation.set(0, b.yaw, 0.16 * Math.sin(swayT * 1.35 + b.ph) + 0.08);
        dummy.scale.set(1, b.h, 1);
      }
      dummy.updateMatrix();
      blade.setMatrixAt(i, dummy.matrix);
    }
    blade.instanceMatrix.needsUpdate = true;
  }

  function frame() {
    requestAnimationFrame(frame);
    const now = performance.now();
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (!visible) return;
    applyPose(poseFromClip(dt));
    swayBlades(dt);
    if (spin) az += dt * 0.075;
    place();
    root.updateMatrixWorld(true);
    syncBones();
    schedulePulses(dt);
    updatePulses(dt);
    renderer.render(scene, cam);
  }
  resize();
  window.addEventListener("resize", resize);
  frame();
  host.classList.add("is-ready");
})();
