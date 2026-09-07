/* a slow field behind the page: three planes of domain-warped noise at
   different depths, offset by scroll and by the opener's zoom so the whole
   thing parallaxes like a space rather than a wallpaper.  expects three.js. */
(function () {
  'use strict';
  const canvas = document.getElementById('bg');
  if (!canvas || !window.THREE) return;
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true, powerPreference: 'low-power' }); } catch (e) { return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25) * 0.6);
  renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const isLight = () => {
    const t = document.documentElement.dataset.theme;
    if (t === 'dark') return 0; if (t === 'light') return 1;
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 0 : 1;
  };
  const u = {
    time: { value: 0 }, res: { value: new THREE.Vector2(1, 1) }, scroll: { value: 0 },
    zoom: { value: 1 }, light: { value: isLight() }, heroFrac: { value: 1 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms: u,
    vertexShader: 'void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: `
      precision highp float;
      uniform float time, scroll, zoom, light, heroFrac; uniform vec2 res;
      float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float noise(vec2 p){
        vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
        return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
      }
      float fbm(vec2 p){
        float v = 0.0, a = 0.5; mat2 r = mat2(0.8, 0.6, -0.6, 0.8);
        for (int i = 0; i < 4; i++) { v += a * noise(p); p = r * p * 2.03 + 1.7; a *= 0.5; }
        return v;
      }
      vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d){ return a + b * cos(6.28318 * (c * t + d)); }
      // one plane at depth d: parallax by scroll/d, scale by zoom^(1/d)
      vec4 plane(vec2 uv, float d, float seed){
        float s = pow(zoom, 1.0 / d);
        vec2 p = (uv - 0.5) * vec2(res.x / res.y, 1.0) * s * (1.1 + 0.6 * d);
        p.y += scroll / res.y / d * 0.9;
        float t = time * 0.03;
        vec2 q = vec2(fbm(p + vec2(0.0, t) + seed), fbm(p + vec2(5.2, 1.3) - t * 0.7 + seed));
        vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2) + t * 0.5), fbm(p + 4.0 * q + vec2(8.3, 2.8) - t * 0.3));
        float f = fbm(p + 3.0 * r);
        float hue = f * 0.9 + seed * 0.13 + t * 0.4;
        vec3 col = pal(hue, vec3(0.5), vec3(0.5), vec3(1.0, 0.9, 0.8), vec3(0.62, 0.45, 0.25));
        float ridge = smoothstep(0.35, 0.75, f) * (1.0 - smoothstep(0.75, 0.95, f));
        float a = (0.35 + 0.65 * ridge) / (0.6 + d * 0.5);
        return vec4(col, a);
      }
      void main(){
        vec2 uv = gl_FragCoord.xy / res;
        vec4 acc = vec4(0.0);
        vec4 c3 = plane(uv, 3.4, 2.0); acc = mix(acc, vec4(c3.rgb, 1.0), c3.a * 0.7);
        vec4 c2 = plane(uv, 1.9, 1.0); acc = mix(acc, vec4(c2.rgb, 1.0), c2.a * 0.6);
        vec4 c1 = plane(uv, 1.0, 0.0); acc = mix(acc, vec4(c1.rgb, 1.0), c1.a * 0.5);
        vec3 col = acc.rgb;
        // ink or paper: the field is tinted toward the page's ground, never louder than the page
        vec3 dark = mix(vec3(0.03, 0.035, 0.06), col * 0.55, 0.9);
        vec3 pale = mix(vec3(0.95, 0.95, 0.93), col, 0.22);
        vec3 outc = mix(dark, pale, light);
        // the opener sits on ink in either theme
        outc = mix(outc, dark, heroFrac * light);
        gl_FragColor = vec4(outc, 1.0);
      }`,
    depthTest: false, depthWrite: false,
  });
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));

  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h, false);
    u.res.value.set(renderer.domElement.width, renderer.domElement.height);
  }
  window.addEventListener('resize', resize);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { u.light.value = isLight(); });
  new MutationObserver(() => { u.light.value = isLight(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  resize();

  const hero = document.getElementById('hero');
  let t0 = performance.now(), frozen = reduceMotion, visible = true, last = 0;
  document.addEventListener('visibilitychange', () => { visible = !document.hidden; if (visible) requestAnimationFrame(frame); });
  function frame(now) {
    if (!visible) return;
    if (!frozen) u.time.value = (now - t0) / 1000;
    const sy = window.scrollY || 0;
    u.scroll.value += (sy * renderer.getPixelRatio() - u.scroll.value) * 0.15;
    const vv = window.visualViewport ? window.visualViewport.scale : 1;
    const target = (window.IBM_ZOOM || 1) * vv;
    u.zoom.value += (target - u.zoom.value) * 0.08;
    if (hero) { const hh = hero.offsetHeight || 1; u.heroFrac.value = Math.max(0, Math.min(1, 1.15 - sy / hh)); }
    // the field is slow: half rate is plenty, and cheaper
    if (now - last > 33 || Math.abs(sy * renderer.getPixelRatio() - u.scroll.value) > 0.5) { renderer.render(scene, camera); last = now; }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
