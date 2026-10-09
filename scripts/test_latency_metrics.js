/* Run with node scripts/test_latency_metrics.js. No browser dependencies. */
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const metrics = require('../realtime/static/latency_metrics.js');
const trace = {session: 'a', avatar: 'may', seq: 7, feature_id: 10,
  capture_ms: 100, sent_ms: 102, received_at: 10000,
  asr_started: 10000.480, asr_finished: 10000.525,
  render_started: 10000.540, render_finished: 10000.567, sending_at: 10000.570};
const sample = metrics.measure(trace, 'a', 'may', 690, 700);
assert.equal(sample.total, 600);
const components = ['prepare', 'context', 'asr', 'featureWait', 'render', 'publishWait', 'transport', 'browser'];
assert(Math.abs(components.reduce((sum, key) => sum + sample[key], 0) - sample.total) < 1e-6);
assert(Math.abs(sample.transport - 18) < 1e-6);
const delayed = metrics.measure(trace, 'a', 'may', 890, 900);
assert.equal(delayed.total - sample.total, 200);
assert.equal(delayed.transport - sample.transport, 200);
// Arbitrary server clock offset changes no displayed value.
const shifted = {...trace};
['received_at', 'asr_started', 'asr_finished', 'render_started', 'render_finished', 'sending_at']
  .forEach(key => shifted[key] += 1234567);
const shiftedSample = metrics.measure(shifted, 'a', 'may', 690, 700);
components.forEach(key => assert(Math.abs(shiftedSample[key] - sample[key]) < 1e-5));
assert.equal(metrics.measure(trace, 'another-tab', 'may', 690, 700), null);
assert.equal(metrics.measure(trace, 'a', 'macron', 690, 700), null);
assert.equal(metrics.measure({}, 'a', 'may', 690, 700), null);
assert.equal(metrics.measure({...trace, asr_started: 9999}, 'a', 'may', 690, 700), null);
assert.equal(metrics.measure(trace, 'a', 'may', 400, 410), null);
assert.equal(metrics.percentile([4, 3, 2, 1], .5), 2);
assert.equal(metrics.percentile([4, 3, 2, 1], .95), 4);
const html = fs.readFileSync(require('path').join(__dirname, '../realtime/static/index.html'), 'utf8');
const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(inline); // Check the actual inline browser script too.
assert(!inline.includes('latestEstimatedLatencyMs'));
console.log('PASS: sums, clock-offset independence, invalid/session/model rejection, percentiles, inline JS syntax');
