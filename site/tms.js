/* the stimulation figure: a 7-axis cobot placing a TMS coil on the scalp.

   the arm is a real 7-DOF cobot, not a drawing of one: the link geometry is built from
   the Franka Emika Panda's published modified-DH parameters, so the joint axes, the link
   offsets and the 855 mm reach are the actual machine's.  the pose is not keyframed
   either -- a damped-least-squares IK solves the seven joint angles that put the coil
   face on the scalp target with its normal along the scalp normal, which is the thing a
   neuronavigation rig actually has to do.

   the target is the motor hotspot: the digitised EEG contact nearest the left precentral
   gyrus, taken from the same graph the rest of the site draws. */
(function () {
  const host = document.getElementById("tms3d");
  const G = window.IBM_GRAPH;
  if (!host || !G || !window.THREE) return;
  const T = window.THREE;
  const MM = 0.001;                                  /* the graph is in mm, we work in m */

  /* ---------- the head: the same point cloud the rest of the site draws ----------
     an opaque scalp shell hides the thing the figure is about.  the cortex is drawn as its
     own sites, and the pulse is shown as a RIPPLE travelling outward from the stimulated
     point through those sites -- brightness by distance from the target, delayed so the
     wave spreads rather than flashing everywhere at once. */
  const ctr = new T.Vector3().fromArray(G.center || [0, 0, 0]).multiplyScalar(MM);
  const scene = new T.Scene();
  const world = new T.Group(); scene.add(world);

  /* THE GRAPH IS SURFACE RAS -- +x right, +y anterior, +z SUPERIOR -- and three.js is
     y-up.  this figure used the raw triple, so the head lay on its back looking at the
     ceiling while a cobot reached down the side of it: the whole scene was a quarter turn
     out.  brain.js line 15 already owned the mapping and this did not use it.

     (-x, z, y) mirrors x as well as swapping y and z, which keeps the determinant at +1.
     that matters here: a reflection would reverse the pial surface's triangle winding and
     invert every normal on the one lit mesh in the figure. */
  const toY = (p, s) => new T.Vector3(-(p[0] * s - ctr.x), p[2] * s - ctr.z, p[1] * s - ctr.y);

  const NODES = G.nodes.length;
  const CORTEX = G.cortex_range ? G.cortex_range[1] : NODES;
  const npos = new Float32Array(NODES * 3);
  for (let i = 0; i < NODES; i++) {
    const v = toY(G.nodes[i].p, MM);
    npos[i * 3] = v.x; npos[i * 3 + 1] = v.y; npos[i * 3 + 2] = v.z;
  }
  const ncol = new Float32Array(NODES * 3);
  const restCol = G.nodes.map((n) => new T.Color(n.c || "#7a8290"));
  const brainGeo = new T.BufferGeometry();
  brainGeo.setAttribute("position", new T.BufferAttribute(npos, 3));
  brainGeo.setAttribute("color", new T.BufferAttribute(ncol, 3));
  /* additive, so the cloud GLOWS against a dark page.  the atlas colours are mostly
     dark blues and purples: drawn normally at this size they vanish into the background,
     which is exactly what made the first pass look empty. */
  /* the WRINKLED pial surface, when data/cortex.js is present.  the node cloud alone is a
     sampling of an oct5 source space and reads as a smooth blob; this is the subject's own
     folded cortex, same frame, so it drops straight in beside the scalp and the contacts. */
  const CX = window.IBM_CORTEX;
  let cortexMesh = null, cxPos = null, cxCol = null, cxBase = null, cxN = 0;
  if (CX) {
    const un = (b64) => { const s2 = atob(b64), u = new Uint8Array(s2.length);
      for (let i = 0; i < s2.length; i++) u[i] = s2.charCodeAt(i); return u; };
    const q = new Uint16Array(un(CX.pos).buffer);
    const idx = new Uint16Array(un(CX.idx).buffer);
    const rgb = un(CX.col);
    cxN = CX.n_vertices;
    cxPos = new Float32Array(cxN * 3);
    for (let i = 0; i < cxN; i++) {
      const v = toY([q[i * 3] * CX.scale + CX.origin[0],
                     q[i * 3 + 1] * CX.scale + CX.origin[1],
                     q[i * 3 + 2] * CX.scale + CX.origin[2]], MM);
      cxPos[i * 3] = v.x; cxPos[i * 3 + 1] = v.y; cxPos[i * 3 + 2] = v.z;
    }
    /* the aparc LUT is FreeSurfer's, and raw it is a bag of fully saturated primaries that
       fights everything else on the page.  keep a third of the hue -- enough that a region
       stays identifiable -- over a warm grey at the page's own value. */
    cxBase = new Float32Array(cxN * 3);
    for (let i = 0; i < cxN; i++) {
      const r = rgb[i * 3] / 255, g = rgb[i * 3 + 1] / 255, b = rgb[i * 3 + 2] / 255;
      const lum = 0.3 * r + 0.59 * g + 0.11 * b;
      const tint = [lum * 0.92 + 0.10, lum * 0.88 + 0.085, lum * 0.82 + 0.075];
      cxBase[i * 3] = (tint[0] + (r - lum) * 0.34) * 0.8;
      cxBase[i * 3 + 1] = (tint[1] + (g - lum) * 0.34) * 0.8;
      cxBase[i * 3 + 2] = (tint[2] + (b - lum) * 0.34) * 0.8;
    }
    cxCol = new Float32Array(cxBase);
    const g2 = new T.BufferGeometry();
    g2.setAttribute("position", new T.BufferAttribute(cxPos, 3));
    g2.setAttribute("color", new T.BufferAttribute(cxCol, 3));
    g2.setIndex(new T.BufferAttribute(idx, 1));
    g2.computeVertexNormals();
    cortexMesh = new T.Mesh(g2, new T.MeshStandardMaterial({
      vertexColors: true, roughness: 0.92, metalness: 0.0,
      flatShading: false, side: T.DoubleSide,
    }));
    world.add(cortexMesh);
  }

  const brain = new T.Points(brainGeo, new T.PointsMaterial({
    size: 0.012, vertexColors: true, transparent: true, opacity: 1,
    sizeAttenuation: true, depthWrite: false, blending: T.AdditiveBlending,
  }));
  if (!CX) world.add(brain);

  /* the scalp stays, as a wireframe hint of where the coil actually sits */
  const scalpGeo = new T.BufferGeometry();
  const sxyz = new Float32Array(G.scalp.xyz.length * 3);
  G.scalp.xyz.forEach((p, i) => {
    const v = toY(p, MM);
    sxyz[i * 3] = v.x; sxyz[i * 3 + 1] = v.y; sxyz[i * 3 + 2] = v.z;
  });
  scalpGeo.setAttribute("position", new T.BufferAttribute(sxyz, 3));
  scalpGeo.setIndex(G.scalp.faces.flat());
  const scalp = new T.LineSegments(new T.WireframeGeometry(scalpGeo),
    new T.LineBasicMaterial({ color: 0x9aa4b4, transparent: true, opacity: 0.11 }));
  world.add(scalp);
  /* and a skin over it, barely there.  a wireframe at 7% is invisible against a lit room,
     so the figure read as a cortex floating in mid air beside a robot -- there was no HEAD
     for the coil to be placed on, which is the whole claim of the picture.  16% is enough
     to give a silhouette and a highlight and still show the ripple through it. */
  scalpGeo.computeVertexNormals();
  const skin = new T.Mesh(scalpGeo, new T.MeshStandardMaterial({
    color: 0xd9b49c, roughness: 0.78, metalness: 0.0, transparent: true, opacity: 0.24,
    side: T.DoubleSide, depthWrite: false,
  }));
  world.add(skin);

  /* ---------- the protocol: four sites, and what each train is FOR ----------

     one site, pulsed over and over, was a demo of a robot.  a course of TMS is a
     PROTOCOL: a site chosen because of the loop it sits in, a band that loop runs at, and
     a pair whose coupling the train is meant to move.  the figure cycles four of the
     targets that are actually used clinically and names, for each, the coupling it is
     aimed at and the meso clique it belongs to -- which is the thing this model claims to
     be able to predict and the reason the site is worth hitting at all.

     DECLARED, NOT MEASURED -- said here and in the caption's data-status, not on the page
     (the site states the programme's end state, by the user's rule).  nothing here has been
     run: no coupling has been predicted, no course simulated, no before/after spectrum
     compared.  what would have to exist to drop the "declared" chip is, in order: a coil
     pose streaming into the field solve so the stimulated set is computed rather than
     assumed; a per-subject motor threshold so the amplitude means something; the paired
     spectra of the named pair before and after a train, against a sham arm; and the same
     across sessions, since the week-four claim is about reorganisation and not about the
     minutes after a pulse.  the cliques below are anatomy -- real aparc labels, the loops
     as the literature describes them -- and the BANDS are the declared priors, which is
     exactly the kind of number docs/LOG.md has moved before (the thalamo-cortical prior
     said alpha and the data said 13.45 Hz).  treat every band here as a prior. */
  const PROTOCOLS = [
    { site: ["precentral", "lh"], name: "left M1", area: "precentral",
      band: "beta 13–30 Hz", couple: "M1 ↔ SMA",
      clique: [["precentral", "lh"], ["paracentral", "lh"],
               ["postcentral", "lh"], ["superiorfrontal", "lh"]] },
    { site: ["rostralmiddlefrontal", "lh"], name: "left DLPFC", area: "rostral middle frontal",
      band: "theta 4–8 Hz", couple: "DLPFC ↔ ACC",
      clique: [["rostralmiddlefrontal", "lh"], ["caudalmiddlefrontal", "lh"],
               ["rostralanteriorcingulate", "lh"], ["caudalanteriorcingulate", "lh"]] },
    { site: ["superiorparietal", "lh"], name: "left IPS", area: "superior parietal",
      band: "alpha 8–13 Hz", couple: "IPS ↔ FEF",
      clique: [["superiorparietal", "lh"], ["inferiorparietal", "lh"],
               ["caudalmiddlefrontal", "lh"], ["precuneus", "lh"]] },
    { site: ["supramarginal", "lh"], name: "left TPJ", area: "supramarginal",
      band: "gamma 30–80 Hz", couple: "TPJ ↔ STG",
      clique: [["supramarginal", "lh"], ["superiortemporal", "lh"],
               ["transversetemporal", "lh"], ["bankssts", "lh"]] },
  ];

  /* the 60 digitised contacts; which one is hot depends on the protocol */
  const eeg = G.nodes.filter((n) => n.g === "eeg");
  const dots = new T.Group(); world.add(dots);
  const dotGeo = new T.SphereGeometry(0.0045, 10, 8);
  const dotMat = new T.MeshBasicMaterial({ color: 0xe9e7e0, transparent: true, opacity: 0.65 });
  const hotMat = new T.MeshBasicMaterial({ color: 0xf0b34a });   /* basic: no emissive */
  const contacts = eeg.map((n) => {
    const p = toY(n.p, MM);
    const m = new T.Mesh(dotGeo, dotMat); m.position.copy(p); dots.add(m);
    return { p, mesh: m };
  });
  if (!contacts.length) { console.warn("tms: no digitised contacts"); return; }

  const nodesIn = (r, h) => G.nodes.filter((n) => n.r === r && (!h || n.h === h));
  const centroidOf = (r, h) => {
    const ns = nodesIn(r, h);
    if (!ns.length) { console.warn("tms: no nodes for region", r, h); return null; }
    const m = ns.reduce((a, n) => a.add(new T.Vector3().fromArray(n.p)), new T.Vector3())
      .multiplyScalar(1 / ns.length);
    return toY([m.x, m.y, m.z], MM);
  };

  /* ---- one prepared scene per protocol: the hot contact, the coil pose, and the
          clique drawn as its own sites plus an arc from the stimulated one to each ---- */
  const cliqueMat = new T.PointsMaterial({ color: 0x8ef0ff, size: 0.011, transparent: true,
    opacity: 0, sizeAttenuation: true, depthWrite: false, blending: T.AdditiveBlending });
  const arcMat = new T.MeshBasicMaterial({ color: 0x63d3e6, transparent: true, opacity: 0,
    depthWrite: false });
  PROTOCOLS.forEach((P) => {
    const aim = centroidOf(P.site[0], P.site[1]);
    P.target = contacts.reduce((b, c) =>
      (!b || c.p.distanceTo(aim) < b.p.distanceTo(aim)) ? c : b, null);
    P.nrm = P.target.p.clone().normalize();
    P.coilAt = P.target.p.clone().addScaledVector(P.nrm, 0.012);
    P.away = P.coilAt.clone().addScaledVector(P.nrm, 0.10).add(new T.Vector3(0.03, 0.02, 0));

    const g = new T.Group(); g.visible = false; world.add(g); P.group = g;
    const pts = [];
    P.clique.forEach(([r, h]) => nodesIn(r, h).forEach((n) => {
      const v = toY(n.p, MM); pts.push(v.x, v.y, v.z);
    }));
    const cg = new T.BufferGeometry();
    cg.setAttribute("position", new T.Float32BufferAttribute(pts, 3));
    g.add(new T.Points(cg, cliqueMat));
    /* a STAR from the stimulated area, not a ring round all four: a ring says every pair
       in the clique is coupled, which is a stronger claim than the one being made. */
    const cs = P.clique.map(([r, h]) => centroidOf(r, h)).filter(Boolean);
    P.centroids = cs;
    cs.slice(1).forEach((c) => {
      const mid = cs[0].clone().lerp(c, 0.5);
      mid.addScaledVector(mid.clone().normalize(), 0.038);   /* bow it clear of the surface */
      const curve = new T.QuadraticBezierCurve3(cs[0], mid, c);
      g.add(new T.Mesh(new T.TubeGeometry(curve, 22, 0.0024, 6, false), arcMat));
    });
  });

  /* the ripple is measured from the stimulated site, so both distance tables are rebuilt
     when the protocol changes -- 3k nodes and 13k vertices, once every seven seconds */
  const dist0 = new Float32Array(CORTEX);
  let dmax = 0, cxDist = CX ? new Float32Array(cxN) : null, cxMax = 0;
  let live = null;
  function setProtocol(P) {
    if (live) { live.group.visible = false; live.target.mesh.material = dotMat;
                live.target.mesh.scale.setScalar(1); }
    live = P;
    P.group.visible = true;
    P.target.mesh.material = hotMat;
    P.target.mesh.scale.setScalar(1.7);
    dmax = 0;
    for (let i = 0; i < CORTEX; i++) {
      const dx = npos[i * 3] - P.target.p.x, dy = npos[i * 3 + 1] - P.target.p.y,
            dz = npos[i * 3 + 2] - P.target.p.z;
      dist0[i] = Math.hypot(dx, dy, dz);
      if (dist0[i] > dmax) dmax = dist0[i];
    }
    if (CX) {
      cxMax = 0;
      for (let i = 0; i < cxN; i++) {
        const dx = cxPos[i * 3] - P.target.p.x, dy = cxPos[i * 3 + 1] - P.target.p.y,
              dz = cxPos[i * 3 + 2] - P.target.p.z;
        cxDist[i] = Math.hypot(dx, dy, dz);
        if (cxDist[i] > cxMax) cxMax = cxDist[i];
      }
    }
    /* ONE LINE, under the render: "target: <site> · <band>".  it was a frosted-glass card
       inside the figure -- site, area, band, the coupled pair, the clique's labels -- and it
       covered a third of the frame to say what the lit clique already shows.  the pair and
       the clique list are in PROTOCOLS above; the status ("declared, not measured") is in
       its header comment and on the element, per the site's rule that status lives in the
       source and docs/LOG.md, not on the page. */
    card.textContent = `target: ${P.name} \u00b7 ${P.band}`;
    card.dataset.status = "declared-not-measured";
  }

  const HOT = new T.Color(0xffe6b0), LIFT = new T.Color(0xaebbd4);
  function ripple(phase) {
    /* phase < 0 means no pulse in flight: everything sits at its resting colour */
    const c = new T.Color();
    for (let i = 0; i < NODES; i++) {
      let amp = 0;
      if (phase >= 0 && i < CORTEX) {
        const front = phase * dmax * 1.35;
        const d = Math.abs(dist0[i] - front);
        amp = Math.exp(-(d * d) / (2 * 0.026 * 0.026)) * (1 - phase * 0.75) * 1.5;
      }
      c.copy(restCol[i]).lerp(LIFT, 0.5).multiplyScalar(0.5).lerp(HOT, Math.min(1, amp));
      ncol[i * 3] = c.r; ncol[i * 3 + 1] = c.g; ncol[i * 3 + 2] = c.b;
    }
    brainGeo.attributes.color.needsUpdate = true;
  }
  /* the same wave, over the surface vertices */
  function rippleCortex(phase) {
    if (!CX) return;
    const front = phase * cxMax * 1.35;
    for (let i = 0; i < cxN; i++) {
      let amp = 0;
      if (phase >= 0) {
        const d = Math.abs(cxDist[i] - front);
        amp = Math.exp(-(d * d) / (2 * 0.028 * 0.028)) * (1 - phase * 0.7) * 1.6;
      }
      amp = Math.min(1, amp);
      cxCol[i * 3] = cxBase[i * 3] + (1 - cxBase[i * 3]) * amp;
      cxCol[i * 3 + 1] = cxBase[i * 3 + 1] + (0.86 - cxBase[i * 3 + 1]) * amp;
      cxCol[i * 3 + 2] = cxBase[i * 3 + 2] + (0.48 - cxBase[i * 3 + 2]) * amp;
    }
    cortexMesh.geometry.attributes.color.needsUpdate = true;
  }

  /* the one-line target caption, OUTSIDE the canvas box: a sibling after #tms3d, so it sits
     under the render and is never painted over by the masked canvas (see CLAUDE.md) */
  const card = document.createElement("p");
  card.className = "tms-target";
  card.setAttribute("aria-live", "polite");
  host.insertAdjacentElement("afterend", card);

  setProtocol(PROTOCOLS[0]);
  ripple(-1); rippleCortex(-1);

  /* ---------- the arm: Franka Panda, modified DH [a, d, alpha] ---------- */
  const DH = [
    [0, 0.333, 0], [0, 0, -Math.PI / 2], [0, 0.316, Math.PI / 2],
    [0.0825, 0, Math.PI / 2], [-0.0825, 0.384, -Math.PI / 2],
    [0, 0, Math.PI / 2], [0.088, 0, Math.PI / 2],
  ];
  const FLANGE = 0.107;
  const LIM = [[-2.897, 2.897], [-1.763, 1.763], [-2.897, 2.897], [-3.072, -0.07],
               [-2.897, 2.897], [-0.0175, 3.752], [-2.897, 2.897]];
  const q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8];

  /* the arm's first DH link runs along its own LOCAL +Z, so the column is z-up in base
     coordinates while the scene is now y-up.  rotation.x = -pi/2 stands it upright (three
     composes an 'XYZ' euler as Rx·Ry·Rz, so rotation.z still spins it about its own
     column, which is the one adjustment a real installation has).  the base sits on the
     cart to the patient's right, below and in front of the head; q[0] finds the azimuth
     that reaches the target, so the yaw here is cosmetic. */
  const base = new T.Group();
  /* BEHIND the subject's left shoulder, not in front of it.  mounted forward of the head
     (z > 0) the arm has to fold back over itself and the elbow lands between the camera
     and the scalp -- the coil was visible and the head it was on was not.  from behind,
     the whole chain rises on the far side and reaches forward into frame. */
  base.position.set(0.36, -0.36, -0.26);
  base.rotation.set(-Math.PI / 2, 0, 0.5);
  world.add(base);

  function dhMat(a, d, al, th) {
    const ct = Math.cos(th), st = Math.sin(th), ca = Math.cos(al), sa = Math.sin(al);
    return new T.Matrix4().set(
      ct, -st, 0, a,
      st * ca, ct * ca, -sa, -d * sa,
      st * sa, ct * sa, ca, d * ca,
      0, 0, 0, 1);
  }
  function fk(qq) {
    const out = [];
    let M = new T.Matrix4();
    for (let i = 0; i < 7; i++) {
      M = M.clone().multiply(dhMat(DH[i][0], DH[i][1], DH[i][2], qq[i]));
      out.push(M.clone());
    }
    out.push(M.clone().multiply(new T.Matrix4().makeTranslation(0, 0, FLANGE)));
    return out;
  }
  const posOf = (M) => new T.Vector3().setFromMatrixPosition(M);
  const zOf = (M) => new T.Vector3(M.elements[8], M.elements[9], M.elements[10]).normalize();

  /* damped least squares over position and tool-axis alignment.  the damping is what
     keeps it stable near a singularity -- an undamped pseudo-inverse throws the elbow */
  function ik(goalP, goalZ, iters) {
    for (let it = 0; it < iters; it++) {
      const Ms = fk(q), tip = posOf(Ms[7]), tz = zOf(Ms[7]);
      const ep = goalP.clone().sub(tip);
      const er = new T.Vector3().crossVectors(tz, goalZ);
      if (ep.length() < 4e-4 && er.length() < 4e-3) break;
      const e = [ep.x, ep.y, ep.z, er.x * 0.22, er.y * 0.22, er.z * 0.22];
      const J = [];
      for (let i = 0; i < 7; i++) {
        const zi = zOf(Ms[i]), pi = posOf(Ms[i]);
        const lin = new T.Vector3().crossVectors(zi, tip.clone().sub(pi));
        J.push([lin.x, lin.y, lin.z, zi.x * 0.22, zi.y * 0.22, zi.z * 0.22]);
      }
      const lam = 0.055;
      for (let i = 0; i < 7; i++) {
        let g = 0;
        for (let k = 0; k < 6; k++) g += J[i][k] * e[k];
        q[i] += g / (1 + lam) * 0.85;
        q[i] = Math.max(LIM[i][0], Math.min(LIM[i][1], q[i]));
      }
    }
  }

  /* ---------- arm geometry: one segment per link, rebuilt from FK each frame ---------- */
  const shell = new T.MeshStandardMaterial({ color: 0xe8e6e1, roughness: 0.42, metalness: 0.06 });
  const joint = new T.MeshStandardMaterial({ color: 0x2c2f38, roughness: 0.5, metalness: 0.35 });
  const trim = new T.MeshStandardMaterial({ color: 0xf0b34a, roughness: 0.35, metalness: 0.5 });
  const coilMat = new T.MeshStandardMaterial({ color: 0xf0b34a, roughness: 0.3, metalness: 0.4,
                                               emissive: 0x3a2400 });
  /* a cobot reads as a cobot because of its PROPORTIONS: pale tapered shells that get
     slimmer toward the wrist, a dark collar at every axis, and a cast base.  bare
     equal-radius cylinders read as a stick figure however correct the kinematics are. */
  const RAD = [0.052, 0.049, 0.046, 0.043, 0.039, 0.034, 0.030, 0.026];
  const links = [], hubs = [], collars = [], bands = [];
  for (let i = 0; i < 8; i++) {
    /* a segmented shell rather than one tube: a slim waist between two thicker ends is
       what a cobot link actually looks like, and it catches the key light in two places */
    const m = new T.Mesh(new T.CylinderGeometry(RAD[i] * 0.8, RAD[i] * 0.86, 1, 36, 1), shell);
    m.geometry.translate(0, 0.5, 0);
    links.push(m); base.add(m);
    if (i < 7) {
      const h = new T.Mesh(new T.SphereGeometry(RAD[i] * 1.08, 28, 20), shell);
      hubs.push(h); base.add(h);
      const c = new T.Mesh(new T.CylinderGeometry(RAD[i] * 1.14, RAD[i] * 1.14, RAD[i] * 0.62, 36), joint);
      collars.push(c); base.add(c);
      /* a thin bright band on each axis: real arms have one, and it reads the rotation */
      const bd = new T.Mesh(new T.TorusGeometry(RAD[i] * 1.16, RAD[i] * 0.075, 8, 30), trim);
      bands.push(bd); base.add(bd);
    }
  }
  /* the pedestal is a cylinder about its own Y and the column rises along the base's Z,
     so it has to be laid down to be a base plate rather than a stub beside one */
  const pedestal = new T.Mesh(new T.CylinderGeometry(0.082, 0.115, 0.055, 32), joint);
  pedestal.rotation.x = Math.PI / 2;
  pedestal.position.z = 0.027; base.add(pedestal);
  /* the coil: two windings side by side, the figure-of-eight a real TMS coil is */
  const coil = new T.Group();
  [-0.026, 0.026].forEach((dx) => {
    const t = new T.Mesh(new T.TorusGeometry(0.026, 0.0085, 10, 26), coilMat);
    t.position.x = dx; coil.add(t);
  });
  base.add(coil);
  const tipDot = new T.Mesh(new T.SphereGeometry(0.006, 12, 10), coilMat); base.add(tipDot);

  function layout() {
    const Ms = fk(q);
    const pts = [new T.Vector3(0, 0, 0), ...Ms.map(posOf)];
    for (let i = 0; i < 8; i++) {
      const a = pts[i], b = pts[i + 1], d = b.clone().sub(a), L = d.length();
      links[i].position.copy(a);
      links[i].scale.set(1, Math.max(L, 1e-4), 1);
      links[i].quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), d.clone().normalize());
      links[i].scale.x = links[i].scale.z = i > 5 ? 0.62 : 1;
    }
    hubs.forEach((h, i) => h.position.copy(pts[i + 1]));
    /* a collar sits ON each axis, aligned to the joint's own z, which is what makes the
       seven axes legible as axes rather than as bends in a tube */
    collars.forEach((c, i) => {
      c.position.copy(pts[i + 1]);
      c.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), zOf(Ms[i]));
      bands[i].position.copy(pts[i + 1]);
      bands[i].quaternion.copy(c.quaternion).multiply(
        new T.Quaternion().setFromAxisAngle(new T.Vector3(1, 0, 0), Math.PI / 2));
    });
    const M = Ms[7];
    coil.position.copy(posOf(M));
    coil.quaternion.setFromRotationMatrix(M);
    tipDot.position.copy(posOf(M));
  }

  /* the three fixed "spectra read here" markers are gone: which regions matter depends
     on which site is being hit, and the protocol's own clique now says it -- in the card,
     and as the lit sites and arcs in the model. */

  /* ---------- the ground the rig stands on ----------
     the rig used to float on the page, which read as a rendering of a coil rather than as
     a procedure happening somewhere.  it went through a whole treatment room -- two walls,
     a skirting line, a ceiling panel, a console on a stand -- and came back down to a floor
     and the cart the arm is bolted to, because every prop past those competed with the
     head for attention and every wall was an edge.  the canvas box is still feathered in
     CSS (see #tms3d's mask) so the frame itself never shows a hard border.

     PLANNED, NOT BUILT: none of this is a model of a clinic.  the neuronavigation loop
     these props stand for -- a tracked coil pose streaming into the field solve, a stored
     MEP threshold per subject, a session ledger over a four-week course -- is what the
     stimulation arm is for, and none of it exists yet.  the geometry here is set dressing
     at a scale that happens to be right (cart top at the arm's base, head one metre off
     the floor, seated) so that when a real coil pose arrives it can be dropped in without
     moving the furniture.

     EVERY PIECE OF IT FADES WITH DISTANCE FROM THE HEAD.  masking the canvas box was not
     enough: inside the mask the room still had four hard architectural edges -- a skirting
     line, a wall corner, the lip of the floor -- and a room that ends in a corner reads as
     a screenshot of a game.  feather() below patches each room material's shader so its
     ALPHA falls off radially in world space about the subject, which means the page shows
     through rather than a colour being blended in, and it means the falloff is the same
     whatever the camera is doing.  the head, the cortex and the arm are never feathered:
     they are the figure, and everything else is the room dissolving around them. */
  const FLOOR_Y = -1.02;
  const room = new T.Group(); world.add(room);

  /* feather(mat, at, r0, r1, planar): alpha is 1 inside r0 of `at`, 0 beyond r1.  with
     `planar` the distance is taken in the FLOOR PLANE (x,z) only.  the floor needs that: a
     spherical falloff about the head measures every floor point as at least a metre away,
     because the floor is a metre down, so the patch directly underfoot could never be both
     solid and small -- it was 89% opaque and ran out to the frame edge.  measured flat, the
     floor is a soft disc under the rig that goes fully transparent well inside the frame. */
  const feather = (mat, at, r0, r1, planar) => {
    mat.transparent = true;
    mat.depthWrite = false;      /* a fading plane must not occlude what is behind it */
    const c = `vec3(${at[0].toFixed(3)}, ${at[1].toFixed(3)}, ${at[2].toFixed(3)})`;
    const d = planar ? `distance(vFeatherPos.xz, ${c}.xz)` : `distance(vFeatherPos, ${c})`;
    mat.onBeforeCompile = (sh) => {
      sh.vertexShader = "varying vec3 vFeatherPos;\n" + sh.vertexShader.replace(
        "#include <begin_vertex>",
        "#include <begin_vertex>\n  vFeatherPos = (modelMatrix * vec4(transformed, 1.0)).xyz;");
      /* dithering_fragment is the last include in both meshphysical and meshbasic in
         r128, so this runs after gl_FragColor is final in either */
      sh.fragmentShader = "varying vec3 vFeatherPos;\n" + sh.fragmentShader.replace(
        "#include <dithering_fragment>",
        `#include <dithering_fragment>\n  gl_FragColor.a *= 1.0 - smoothstep(${r0.toFixed(3)}, ${r1.toFixed(3)}, ${d});`);
    };
    mat.needsUpdate = true;
    return mat;
  };
  const HEAD_AT = [0, -0.05, 0];
  const roomMat = (hex, rough) => feather(new T.MeshStandardMaterial({
    color: hex, roughness: rough, metalness: 0.0 }), HEAD_AT, 0.55, 1.6, false);
  const plate = (w, h, mat, pos, rot) => {
    const m = new T.Mesh(new T.PlaneGeometry(w, h), mat);
    m.position.set(pos[0], pos[1], pos[2]);
    if (rot) m.rotation.set(rot[0], rot[1], rot[2]);
    room.add(m);
    return m;
  };
  /* NO WALLS.  the corner, the skirting line and the ceiling panel made it a room, and a
     room has edges however far they are feathered; what the figure needs is a GROUND, so
     the rig reads as standing somewhere, and nothing else.  the floor is centred between
     the head and the cart, solid only in a small disc under them, and gone by 1.05 m. */
  const FLOOR_AT = [(base.position.x) * 0.5, FLOOR_Y, (base.position.z) * 0.5];
  plate(4, 4, feather(new T.MeshStandardMaterial({ color: 0x7c8385, roughness: 0.95, metalness: 0 }),
                      FLOOR_AT, 0.12, 1.05, true),
        [FLOOR_AT[0], FLOOR_Y, FLOOR_AT[2]], [-Math.PI / 2, 0, 0]);

  /* the cart the arm is bolted to: its top face is exactly the arm's base plane */
  const cart = new T.Mesh(new T.BoxGeometry(0.44, 0.62, 0.44), roomMat(0xa4abab, 0.6));
  cart.position.set(base.position.x, base.position.y - 0.31, base.position.z); room.add(cart);
  const cartTop = new T.Mesh(new T.BoxGeometry(0.48, 0.022, 0.48), roomMat(0x8b9294, 0.5));
  cartTop.position.set(base.position.x, base.position.y - 0.006, base.position.z); room.add(cartTop);

  /* NO MANNEQUIN.  two goes at giving the head a body -- a neck-shoulder-torso stack of
     cylinders, then a single lathed silhouette -- both came out looking like laboratory
     glassware with a brain in the top of it, because a translucent torso at this scale has
     no features to read as anatomy and every smooth revolved solid reads as a vessel.  the
     rest of this site draws a brain in the air and is legible; a half-modelled person is
     not, and it stole attention from the one thing the figure is about.  so the head is the
     subject, the room fades around it, and nothing pretends to be a patient. */

  /* no console either: its stand ran straight up the label column and its screen sat
     behind the protocol card.  the card is the operator's view now. */

  /* ---------- lighting, camera ---------- */
  scene.add(new T.HemisphereLight(0xdfe9f2, 0x2b2b28, 0.62));
  const k1 = new T.DirectionalLight(0xfff6e8, 0.78); k1.position.set(1.4, 2.6, 1.2); scene.add(k1);
  const k2 = new T.DirectionalLight(0x9fc0ff, 0.38); k2.position.set(-1.5, 0.6, 0.9); scene.add(k2);
  const k3 = new T.DirectionalLight(0xffc98a, 0.55); k3.position.set(-0.6, 0.9, -1.8); scene.add(k3);

  const cam = new T.PerspectiveCamera(34, 1, 0.01, 20);
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);

  /* frame the whole rig -- head AND the reach of the arm.  fitting to the head alone
     runs the arm off the bottom of the canvas, and the arm is most of the picture.

     a bounding SPHERE of the rig, and the field of view taken as the NARROWER of the
     vertical and horizontal ones.  the old fit used the largest half-extent against the
     vertical fov only, so on a canvas wider than it is tall -- which is every canvas this
     figure gets -- the horizontal fov was never checked and the arm ran out of frame.
     the ROOM is excluded on purpose: its walls are metres across and fitting to them
     would leave the head a dozen pixels wide. */
  let az = 0.58, el = 0.18, dist = 1.25, drag = null;
  const focus = new T.Vector3();
  function fit() {
    world.updateMatrixWorld(true);
    const box = new T.Box3();
    /* the SUBJECT, not the rig: the head, the contacts, the coil and the wrist end of the
       arm.  fitting the whole 855 mm chain is what the frame did next, and it is correct
       and useless -- the arm is two thirds of that sphere and the head came out at 16% of
       the picture height.  the elbow and the base run off the bottom-right corner, which
       is exactly where the room is fading out, so the crop never reads as a cut edge. */
    [cortexMesh || brain, skin, dots, coil].concat(links.slice(5), hubs.slice(5))
      .forEach((o) => { if (o) box.expandByObject(o); });
    if (box.isEmpty()) return;
    const sph = box.getBoundingSphere(new T.Sphere());
    focus.copy(sph.center);
    const vfov = cam.fov * Math.PI / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * (cam.aspect || 1));
    dist = sph.radius / Math.sin(Math.min(vfov, hfov) / 2) * 1.55;
  }
  function place() {
    cam.position.set(focus.x + Math.sin(az) * Math.cos(el) * dist,
                     focus.y + Math.sin(el) * dist,
                     focus.z + Math.cos(az) * Math.cos(el) * dist);
    cam.lookAt(focus);
  }
  function resize() {
    const w = host.clientWidth, h = Math.max(340, Math.round(w * 0.66));
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
    /* refit after the aspect changes -- fitting once at load framed the rig for whatever
       width the column happened to have before layout settled, and cropped it after */
    fit();
  }
  host.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, az, el }; host.setPointerCapture(e.pointerId); });
  host.addEventListener("pointermove", (e) => {
    if (!drag) return;
    az = drag.az - (e.clientX - drag.x) * 0.008;
    el = Math.max(-0.35, Math.min(0.85, drag.el + (e.clientY - drag.y) * 0.005));
  });
  const up = () => { drag = null; };
  host.addEventListener("pointerup", up); host.addEventListener("pointercancel", up);

  /* the approach: hold off the scalp, close on it, pulse, retract, move to the next site.
     the protocol swaps at the TOP of a cycle, with the coil at its retract pose, so the
     arm slews to the new standoff over the approach rather than jumping -- which is what a
     navigated rig actually looks like between targets. */
  const CYCLE = 7;
  const toLocal = (p) => base.worldToLocal(p.clone());
  let vis = true;
  if (window.IntersectionObserver)
    new IntersectionObserver((es) => { vis = es[0].isIntersecting; }, { rootMargin: "200px" }).observe(host);

  const t0 = performance.now();
  function frame() {
    requestAnimationFrame(frame);
    if (!vis) return;
    const el2 = (performance.now() - t0) / 1000;
    const want = PROTOCOLS[Math.floor(el2 / CYCLE) % PROTOCOLS.length];
    if (want !== live) setProtocol(want);
    const t = el2 % CYCLE;
    const s = t < 2.4 ? t / 2.4 : t < 4.8 ? 1 : 1 - (t - 4.8) / 2.2;
    const e = Math.max(0, Math.min(1, s)) ** 2 * (3 - 2 * Math.max(0, Math.min(1, s)));
    const goalZ = live.nrm.clone().negate();
    const goal = live.away.clone().lerp(live.coilAt, e);
    ik(toLocal(goal), base.worldToLocal(base.localToWorld(new T.Vector3()).add(goalZ)).normalize(), 9);
    layout();
    /* the pulse fires once the coil is seated, and the ripple runs from there */
    const T0 = 2.6, TR = 1.7;
    const phase = (t > T0 && t < T0 + TR) ? (t - T0) / TR : -1;
    ripple(phase); rippleCortex(phase);
    /* the clique is always faintly lit -- it is a standing property of the substrate, not
       something the pulse creates -- and the pulse drives it up.  the arcs lag the sites a
       little so the coupling reads as following the stimulation rather than preceding it. */
    const glow = phase < 0 ? 0 : Math.sin(Math.min(1, phase * 1.15) * Math.PI);
    cliqueMat.opacity = 0.16 + 0.74 * glow;
    arcMat.opacity = 0.10 + 0.5 * Math.max(0, Math.sin(Math.min(1, phase * 0.95) * Math.PI));
    coilMat.emissive.setHex(phase >= 0 && phase < 0.18 ? 0xf0b34a : 0x3a2400);
    hotMat.color.setHex(phase >= 0 ? 0xfff0cf : 0xf0b34a);
    place();
    renderer.render(scene, cam);
    /* NO projected labels at all any more.  the card names the site, and the contact
       being hit is already the one amber sphere on the head and the origin the ripple
       starts from -- a floating "left IPS" beside it repeated the card and landed on the
       console, which is where the last three of these landed too. */
  }
  resize();
  window.addEventListener("resize", resize);
  ik(toLocal(live.away),
     base.worldToLocal(base.localToWorld(new T.Vector3()).add(live.nrm.clone().negate())).normalize(), 40);
  layout();
  fit();
  frame();
  host.classList.add("is-ready");
})();
