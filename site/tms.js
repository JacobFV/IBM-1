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

  /* ---------- the head ---------- */
  const ctr = new T.Vector3().fromArray(G.center || [0, 0, 0]).multiplyScalar(MM);
  const scalpGeo = new T.BufferGeometry();
  scalpGeo.setAttribute("position", new T.Float32BufferAttribute(
    G.scalp.xyz.flat().map((v) => v * MM), 3));
  scalpGeo.setIndex(G.scalp.faces.flat());
  scalpGeo.computeVertexNormals();

  const scene = new T.Scene();
  const head = new T.Mesh(scalpGeo, new T.MeshStandardMaterial({
    color: 0xd9c2ae, roughness: 0.82, metalness: 0.02,
    transparent: true, opacity: 0.5, side: T.DoubleSide,
  }));
  head.position.sub(ctr);
  const world = new T.Group(); world.add(head); scene.add(world);

  /* the 60 digitised contacts, and the one the pulse is aimed at */
  const eeg = G.nodes.filter((n) => n.g === "eeg");
  const pre = G.nodes.filter((n) => n.r === "precentral" && n.h === "lh");
  const pc = pre.length
    ? pre.reduce((a, n) => a.add(new T.Vector3().fromArray(n.p)), new T.Vector3()).multiplyScalar(1 / pre.length)
    : new T.Vector3(-40, 0, 90);
  const pcm = pc.clone().multiplyScalar(MM).sub(ctr);   /* pc is a Vector3 already */
  let target = null, best = Infinity;
  const dots = new T.Group(); world.add(dots);
  const dotGeo = new T.SphereGeometry(0.0045, 10, 8);
  const dotMat = new T.MeshStandardMaterial({ color: 0xe9e7e0, roughness: 0.5 });
  const hotMat = new T.MeshStandardMaterial({ color: 0xf0b34a, emissive: 0x6b4405, roughness: 0.35 });
  eeg.forEach((n) => {
    const p = new T.Vector3().fromArray(n.p).multiplyScalar(MM).sub(ctr);
    const d = p.distanceTo(pcm);
    const m = new T.Mesh(dotGeo, dotMat); m.position.copy(p); dots.add(m);
    if (d < best) { best = d; target = { p, mesh: m }; }
  });
  if (!target) { console.warn("tms: no target contact found"); return; }
  target.mesh.material = hotMat;
  target.mesh.scale.setScalar(1.6);
  const nrm = target.p.clone().normalize();           /* scalp is convex enough here */
  const coilAt = target.p.clone().addScaledVector(nrm, 0.012);

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

  const base = new T.Group();
  base.position.set(0.30, -0.34, 0.16);
  base.rotation.y = -Math.PI * 0.62;
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
  const joint = new T.MeshStandardMaterial({ color: 0x2c2f38, roughness: 0.55, metalness: 0.25 });
  const coilMat = new T.MeshStandardMaterial({ color: 0xf0b34a, roughness: 0.3, metalness: 0.4,
                                               emissive: 0x3a2400 });
  const links = [], hubs = [];
  for (let i = 0; i < 8; i++) {
    const m = new T.Mesh(new T.CylinderGeometry(0.036, 0.036, 1, 18), shell);
    m.geometry.translate(0, 0.5, 0);
    links.push(m); base.add(m);
    if (i < 7) { const h = new T.Mesh(new T.SphereGeometry(0.046, 18, 14), joint); hubs.push(h); base.add(h); }
  }
  base.add(new T.Mesh(new T.CylinderGeometry(0.075, 0.095, 0.05, 24), joint));
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
      if (i < 7) hubs[i].position.copy(pts[i + 1]).addScaledVector(new T.Vector3(), 0);
    }
    hubs.forEach((h, i) => h.position.copy(pts[i + 1]));
    const M = Ms[7];
    coil.position.copy(posOf(M));
    coil.quaternion.setFromRotationMatrix(M);
    tipDot.position.copy(posOf(M));
  }

  /* ---------- lighting, camera ---------- */
  scene.add(new T.HemisphereLight(0xcfe0ff, 0x1a1208, 0.6));
  const k1 = new T.DirectionalLight(0xfff2e0, 1.3); k1.position.set(1.6, 1.8, 1.4); scene.add(k1);
  const k2 = new T.DirectionalLight(0x8fb2ff, 0.4); k2.position.set(-1.5, 0.4, 0.9); scene.add(k2);
  const k3 = new T.DirectionalLight(0xffc98a, 0.7); k3.position.set(-0.6, 0.9, -1.8); scene.add(k3);

  const cam = new T.PerspectiveCamera(34, 1, 0.01, 20);
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);

  /* frame the whole rig -- head AND the reach of the arm.  fitting to the head alone
     runs the arm off the bottom of the canvas, and the arm is most of the picture. */
  let az = 0.78, el = 0.2, dist = 1.25, drag = null;
  const focus = new T.Vector3();
  function fit() {
    const box = new T.Box3();
    world.updateMatrixWorld(true);
    world.traverse((o) => { if (o.isMesh) box.expandByObject(o); });
    if (box.isEmpty()) return;
    box.getCenter(focus);
    const sz = box.getSize(new T.Vector3());
    const r = Math.max(sz.x, sz.y, sz.z) * 0.5;
    dist = r / Math.tan((cam.fov * Math.PI / 180) / 2) * 1.5;
  }
  function place() {
    cam.position.set(focus.x + Math.sin(az) * Math.cos(el) * dist,
                     focus.y + Math.sin(el) * dist,
                     focus.z + Math.cos(az) * Math.cos(el) * dist);
    cam.lookAt(focus);
  }
  function resize() {
    const w = host.clientWidth, h = Math.max(300, Math.round(w * 0.84));
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
  }
  host.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, az, el }; host.setPointerCapture(e.pointerId); });
  host.addEventListener("pointermove", (e) => {
    if (!drag) return;
    az = drag.az - (e.clientX - drag.x) * 0.008;
    el = Math.max(-0.35, Math.min(0.85, drag.el + (e.clientY - drag.y) * 0.005));
  });
  const up = () => { drag = null; };
  host.addEventListener("pointerup", up); host.addEventListener("pointercancel", up);

  /* the approach: hold off the scalp, close on it, pulse, retract */
  const goalZ = nrm.clone().negate();
  const away = coilAt.clone().addScaledVector(nrm, 0.16).add(new T.Vector3(0.05, 0.03, 0));
  const toLocal = (p) => base.worldToLocal(p.clone());
  let vis = true;
  if (window.IntersectionObserver)
    new IntersectionObserver((es) => { vis = es[0].isIntersecting; }, { rootMargin: "200px" }).observe(host);

  const t0 = performance.now();
  function frame() {
    requestAnimationFrame(frame);
    if (!vis) return;
    const t = ((performance.now() - t0) / 1000) % 6;
    const s = t < 2 ? t / 2 : t < 4 ? 1 : 1 - (t - 4) / 2;
    const e = s * s * (3 - 2 * s);
    const goal = away.clone().lerp(coilAt, e);
    ik(toLocal(goal), base.worldToLocal(base.localToWorld(new T.Vector3()).add(goalZ)).normalize(), 9);
    layout();
    const pulse = t > 2.5 && t < 3.6;
    coilMat.emissive.setHex(pulse ? 0xf0b34a : 0x3a2400);
    hotMat.emissive.setHex(pulse ? 0xf0b34a : 0x6b4405);
    place();
    renderer.render(scene, cam);
  }
  resize();
  window.addEventListener("resize", resize);
  ik(toLocal(away), base.worldToLocal(base.localToWorld(new T.Vector3()).add(goalZ)).normalize(), 40);
  layout();
  fit();
  frame();
  host.classList.add("is-ready");
})();
