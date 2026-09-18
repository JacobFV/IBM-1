/* the musculoskeletal body: real Z-Anatomy surfaces, rigidly driven per segment by a
   trajectory the body model actually solved.  see scripts/export_site_body.py for how
   the meshes (za-* structures) are bridged to the trajectory (bp3d entities): each mesh
   rides the segment of the nearest bound centroid, and each segment's per-frame rigid
   motion is recovered from its own members by Kabsch.

   every mesh is emitted in its segment's REST frame, so playback is one matrix per
   segment per frame and no per-vertex work -- 130 meshes animate without touching a
   single vertex buffer after upload. */
(function () {
  const host = document.getElementById("body3d");
  const D = window.IBM_BODY;
  if (!host || !D || !window.THREE) return;
  const T = window.THREE;

  const scene = new T.Scene();
  const cam = new T.PerspectiveCamera(32, 1, 0.05, 40);
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);

  /* three lights rather than one: a key to shape the muscle bellies, a cool fill so the
     shadowed side does not go black against a dark page, and a rim to lift the
     silhouette off the background */
  scene.add(new T.HemisphereLight(0xbfd4ff, 0x22150f, 0.55));
  const key = new T.DirectionalLight(0xfff0dd, 1.45); key.position.set(2.4, 3.2, 2.2); scene.add(key);
  const fill = new T.DirectionalLight(0x93b4ff, 0.42); fill.position.set(-2.6, 0.6, 1.4); scene.add(fill);
  const rim = new T.DirectionalLight(0xffc98a, 0.85); rim.position.set(-1.2, 1.4, -3.0); scene.add(rim);

  const centre = new T.Vector3().fromArray(D.centre || [0, 0.9, 0]);
  const root = new T.Group();
  root.position.set(-centre.x, -centre.y, -centre.z);
  scene.add(root);

  /* one group per segment; the exporter gives us a matrix per segment per frame */
  const groups = {};
  (D.segments || []).forEach((s) => { const g = new T.Group(); g.matrixAutoUpdate = false; groups[s] = g; root.add(g); });

  const mats = {};
  (D.meshes || []).forEach((m) => {
    const geo = new T.BufferGeometry();
    geo.setAttribute("position", new T.Float32BufferAttribute(m.v, 3));
    geo.setIndex(m.i);
    geo.computeVertexNormals();
    if (!mats[m.col]) {
      mats[m.col] = new T.MeshStandardMaterial({
        color: new T.Color(m.col), roughness: m.sys === "skeletal" ? 0.62 : 0.78,
        metalness: 0.0, flatShading: false,
      });
    }
    const mesh = new T.Mesh(geo, mats[m.col]);
    (groups[m.seg] || root).add(mesh);
  });

  const MB = D.motionBySegment || {};
  const times = D.times || [0];
  const tmp = new T.Matrix4();
  /* THE REST POSE, not a trajectory.  every stored trajectory for this body is outside
     the model's own declared joint ranges (docs/LOG.md: crawl-best's left knee is past its
     limit for 95.6% of 1,600 frames, the ankles for 79-94%), so driving the anatomy with
     one tears the skeleton apart at the joints.  the binding and the geometry are right --
     rendered at rest this is a correctly articulated standing figure -- and it is the
     MOTION that is not admissible.  so the figure turns rather than walks. */
  function pose(fi) {
    for (const s in groups) groups[s].matrix.identity();
    return;
    for (const s in groups) {
      const f = MB[s] && MB[s][fi];
      if (!f) continue;
      /* the exporter writes row-major R then t; three wants column-major elements */
      tmp.set(f[0], f[1], f[2], f[9],
              f[3], f[4], f[5], f[10],
              f[6], f[7], f[8], f[11],
              0, 0, 0, 1);
      groups[s].matrix.copy(tmp);
    }
  }

  /* fit the camera to the POSED body rather than to hand-tuned numbers: the meshes are
     stored in each segment's rest frame, so their raw bounds say nothing about where the
     figure actually stands once frame 0 is applied. */
  function fit() {
    pose(0);
    root.position.set(0, 0, 0);
    root.updateMatrixWorld(true);
    const box = new T.Box3();
    root.traverse((o) => { if (o.isMesh) box.expandByObject(o); });
    if (box.isEmpty()) return 3.2;
    const c = box.getCenter(new T.Vector3()), sz = box.getSize(new T.Vector3());
    root.position.set(-c.x, -c.y, -c.z);
    const h = Math.max(sz.y, sz.x * 1.1);
    return (h / 2) / Math.tan((cam.fov * Math.PI / 180) / 2) * 1.16;
  }

  let az = -0.4, el = 0.06, dist = 3.2, drag = null, spin = true, t0 = performance.now();
  dist = fit();
  function place() {
    cam.position.set(Math.sin(az) * Math.cos(el) * dist, Math.sin(el) * dist + 0.05,
                     Math.cos(az) * Math.cos(el) * dist);
    cam.lookAt(0, 0.02, 0);
  }
  function resize() {
    const w = host.clientWidth, h = Math.max(320, Math.round(w * 1.02));
    renderer.setSize(w, h, false);
    cam.aspect = w / h; cam.updateProjectionMatrix();
  }

  host.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, az, el }; host.setPointerCapture(e.pointerId); spin = false; });
  host.addEventListener("pointermove", (e) => {
    if (!drag) return;
    az = drag.az - (e.clientX - drag.x) * 0.008;
    el = Math.max(-0.5, Math.min(0.7, drag.el + (e.clientY - drag.y) * 0.005));
  });
  const stop = () => { drag = null; };
  host.addEventListener("pointerup", stop);
  host.addEventListener("pointercancel", stop);

  const dur = (D.duration_s || 1.5) * 1000;
  let visible = true;
  if (window.IntersectionObserver)
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; }, { rootMargin: "200px" }).observe(host);

  function frame() {
    requestAnimationFrame(frame);
    if (!visible) return;
    if (spin) az = -0.4 + Math.sin((performance.now() - t0) / 11000) * 0.62;
    place();
    renderer.render(scene, cam);
  }
  resize();
  window.addEventListener("resize", () => { resize(); });
  pose(0);
  frame();
  host.classList.add("is-ready");
})();
