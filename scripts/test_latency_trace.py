"""CPU-only tracing regressions: python -m unittest scripts.test_latency_trace."""
import json
import struct
import unittest

from realtime.latency_trace import AudioTraceTimeline, frame_metadata, unpack_audio_packet


def packet(meta, pcm=b"\x01\x00" * 160):
    raw = json.dumps(meta).encode("utf-8")
    return b"TGA1" + struct.pack("!I", len(raw)) + raw + pcm


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(session="browser-session-1", seq=3, capture_ms=100.5, sent_ms=101.1)

    def test_packet_roundtrip(self):
        pcm, trace = unpack_audio_packet(packet(self.meta), 555.0)
        self.assertEqual(len(pcm), 320)
        self.assertEqual(trace["capture_ms"], 100.5)
        self.assertEqual(trace["received_at"], 555.0)

    def test_reject_invalid_packets(self):
        for raw in (b"", b"TGA1", b"TGA1\xff\xff\xff\xff", packet(self.meta, b"\x00"),
                    packet(dict(self.meta, sent_ms=float("nan"))),
                    packet(dict(self.meta, capture_ms=200)), packet([]),
                    packet(dict(self.meta, seq=-1))):
            with self.assertRaises(ValueError):
                unpack_audio_packet(raw, 555.0)

    def test_trace_packet_containing_anchor_not_latest_packet(self):
        timeline = AudioTraceTimeline(64000)
        for seq in range(15):
            timeline.append(1365, dict(self.meta, seq=seq, capture_ms=seq * 85.3,
                                      received_at=555 + seq * .0853))
        # Eight-window attention selects index 4; centre is 480 ms behind
        # the latest block in this constructed 50-logit example.
        consumed, rolling, latest_start = 16640, 16000, 24
        centers = [consumed - rolling + (s + 8) * 320
                   for s in range(latest_start - 14, latest_start + 1, 2)]
        trace = timeline.feature_trace(centers[4], 9, 557, 557.05)
        self.assertEqual(trace["seq"], int(centers[4] // 1365))
        self.assertNotEqual(trace["seq"], 14)
        self.assertEqual(trace["feature_id"], 9)

    def test_boundaries_untraced_and_reset(self):
        timeline = AudioTraceTimeline(64000)
        timeline.append(100, self.meta)
        timeline.append(100, None)
        self.assertIsNotNone(timeline.feature_trace(99, 0, 1, 2))
        self.assertIsNone(timeline.feature_trace(100, 0, 1, 2))
        self.assertIsNone(timeline.feature_trace(200, 0, 1, 2))
        self.assertIsNone(AudioTraceTimeline(64000).feature_trace(99, 0, 1, 2))

    def test_bounded_history(self):
        timeline = AudioTraceTimeline(300)
        for _ in range(100):
            timeline.append(100, self.meta)
        self.assertLessEqual(len(timeline.spans), 5)
        self.assertIsNone(timeline.feature_trace(0, 0, 1, 2))

    def test_sending_does_not_mutate_trace(self):
        original = dict(self.meta, render_finished=3)
        self.assertEqual(frame_metadata(original, 4)["sending_at"], 4)
        self.assertNotIn("sending_at", original)
        self.assertEqual(frame_metadata(None, 4), {})


if __name__ == "__main__":
    unittest.main()
