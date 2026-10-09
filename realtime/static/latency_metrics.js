/* Pure functions shared by the browser and Node regression tests. */
(function(root) {
  'use strict';
  function measure(trace, session, avatar, receivedMs, paintedMs) {
    if (!trace || !session || trace.session !== session || trace.avatar !== avatar) return null;
    const fields = ['capture_ms', 'sent_ms', 'received_at', 'asr_started',
      'asr_finished', 'render_started', 'render_finished', 'sending_at', 'feature_id'];
    if (!fields.every(key => typeof trace[key] === 'number' && Number.isFinite(trace[key]))) return null;
    if (![receivedMs, paintedMs].every(Number.isFinite)) return null;
    const server = (trace.sending_at - trace.received_at) * 1000;
    const result = {
      featureId: trace.feature_id, audioSeq: trace.seq,
      total: paintedMs - trace.capture_ms,
      prepare: trace.sent_ms - trace.capture_ms,
      context: (trace.asr_started - trace.received_at) * 1000,
      asr: (trace.asr_finished - trace.asr_started) * 1000,
      featureWait: (trace.render_started - trace.asr_finished) * 1000,
      render: (trace.render_finished - trace.render_started) * 1000,
      publishWait: (trace.sending_at - trace.render_finished) * 1000,
      // Round-trip residual includes WS serialization, transport and dispatch;
      // it is deliberately NOT labelled one-way network latency.
      transport: (receivedMs - trace.sent_ms) - server,
      browser: paintedMs - receivedMs,
      receivedMs, paintedMs, trace
    };
    if (['total', 'prepare', 'context', 'asr', 'featureWait', 'render',
      'publishWait', 'transport', 'browser'].some(key => result[key] < 0)) return null;
    return result;
  }
  function percentile(values, fraction) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.max(0, Math.ceil(sorted.length * fraction) - 1)];
  }
  const api = {measure, percentile};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LatencyMetrics = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
