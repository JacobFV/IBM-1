// Capture the site hero with one materialization selected, once per brain
// network shown in the reel's "networks of the brain" pile.
//
//   python3 -m http.server 8766 --bind 127.0.0.1   (from site/)
//   node scripts/capture_network_tiles.js [playwright-module-path]
//
// The hero selects `#m=<id>` on load; the lit set blooms over ~1.6 s and the
// camera glides to the model's focus, so each capture waits well past both.
// What is lit is exactly the materialization's declared `hot` set from
// site/data/graph.js -- the captions in site/index.html name those regions.
const path = require('path');
const pw = require(process.argv[2] || 'playwright');

const IDS = ['eeg_to_image', 'speech_envelope', 'meg_to_text', 'invasive_bci',
  'plasticity_learning', 'pharmaco', 'sleep_dynamics', 'anesthesia'];
const OUT = path.join(__dirname, '..', 'site', 'media');

(async () => {
  const browser = await pw.chromium.launch({
    executablePath: process.env.CHROME || '/usr/bin/google-chrome',
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
  for (const id of IDS) {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
    await page.goto(`http://127.0.0.1:8766/#m=${id}`);
    await page.waitForFunction((m) => document.querySelector(`.ring-item.is-active[data-id="${m}"]`), id);
    await page.waitForTimeout(9000);
    const file = path.join(OUT, `net-${id}.jpg`);
    await page.locator('#hero').screenshot({ path: file, type: 'jpeg', quality: 88 });
    console.log('wrote', file);
    await page.close();
  }
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
