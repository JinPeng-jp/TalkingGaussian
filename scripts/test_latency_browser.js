/* Real-browser + real-GPU smoke test. Service must be idle (single audio user).
 * NODE_PATH=/path/to/node_modules node scripts/test_latency_browser.js
 * Uses a test browser with a synthetic microphone, never a physical device.
 */
const {chromium} = require('playwright');
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const base = process.env.TG_TEST_URL || 'https://192.168.167.5:8443';
const output = process.env.TG_TEST_OUTPUT || path.resolve('logs/latency-trace-test');

async function main() {
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true, args: [
    '--no-sandbox', '--autoplay-policy=no-user-gesture-required',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding'
  ]});
  const context = await browser.newContext({ignoreHTTPSErrors: true,
    viewport: {width: 1500, height: 1100}, permissions: ['microphone']});
  const page = await context.newPage();
  const errors = [];
  const report = {runs: [], errors};
  let lastSession = null;
  page.on('pageerror', error => errors.push(String(error)));
  page.on('dialog', async dialog => { errors.push(dialog.message()); await dialog.dismiss(); });
  async function collect(label) {
    await page.waitForFunction(previous => traceSession !== previous &&
      inputMode !== 'none' && measuredSamples.length >= 65, lastSession, {timeout: 30000});
    const result = await page.evaluate(() => ({
      avatar: selectedAvatar, transport: videoTransport, totalText: document.getElementById('latencyTotal').textContent,
      summary: document.getElementById('latencySummary').textContent,
      firstDrivenFrameMs, session: traceSession, samples: measuredSamples,
      fps: document.getElementById('displayFps').textContent
    }));
    assert(result.samples.every(s => s.trace.session === result.session && s.trace.avatar === result.avatar));
    assert.equal(new Set(result.samples.map(s => s.featureId)).size, result.samples.length);
    assert.notEqual(result.session, lastSession);
    lastSession = result.session;
    result.samples.forEach(s => {
      const sum = ['prepare', 'context', 'asr', 'featureWait', 'render', 'publishWait', 'transport', 'browser']
        .reduce((n, key) => n + s[key], 0);
      assert(Math.abs(sum - s.total) < 0.001);
      assert.equal(s.total, s.paintedMs - s.trace.capture_ms);
      assert(s.context > 0);
    });
    report.runs.push({label, ...result});
    console.log(label, result.summary, result.fps);
    await page.screenshot({path: path.join(output, `${label}.png`), fullPage: true});
    return result;
  }
  try {
    await page.goto(base);
    await page.waitForFunction(() => latestStatus.state === 'ready' && latestStatus.asr && latestStatus.asr_alive);
    assert.equal(await page.locator('#latencyTotal').innerText(), '-- ms');
    await page.click('[data-avatar="may"]');
    await page.click('#startSample');
    const may = await collect('may-websocket');
    const observer = await context.newPage();
    await observer.goto(base);
    await observer.waitForTimeout(1000);
    assert.equal(await observer.locator('#latencyTotal').innerText(), '-- ms');
    assert.equal(await observer.evaluate(() => measuredCount), 0);
    await observer.close();
    await page.bringToFront();
    const downloadPromise = page.waitForEvent('download');
    await page.click('#exportLatency');
    await (await downloadPromise).saveAs(path.join(output, 'may-export.json'));
    await page.click('#stop');
    await page.waitForTimeout(750);
    assert.equal(await page.locator('#latencyTotal').innerText(), '-- ms');
    assert(await page.evaluate(() => measuredCount > 0));
    await page.click('[data-avatar="macron"]');
    await page.waitForFunction(() => selectedAvatar === 'macron' && !avatarChangeBusy);
    assert.equal(await page.evaluate(() => measuredCount), 0);
    await page.click('#startSample');
    const macron = await collect('macron-websocket');
    assert.notEqual(may.session, macron.session);
    await page.click('#stop');
    await page.click('#startMic');
    await collect('macron-synthetic-microphone');
    await page.click('#stop');
    // Force the real browser through the independent HTTP fallback path.
    await page.evaluate(() => {
      connectVideoWebSocket = () => {};
      const old = videoSocket;
      videoSocket = null;
      latestWebSocketFrame = null;
      videoTransport = 'HTTP 备用';
      if (old) old.close();
    });
    await page.waitForTimeout(600);
    await page.click('#startSample');
    const fallback = await collect('macron-http-fallback');
    assert.equal(fallback.transport, 'HTTP 备用');
    await page.click('#stop');
    await page.click('[data-avatar="may"]');
    await page.waitForFunction(() => selectedAvatar === 'may' && !avatarChangeBusy);
    assert.deepEqual(errors, []);
    report.passed = true;
  } finally {
    await page.evaluate(() => stopAudioInput()).catch(() => {});
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
