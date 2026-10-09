"""Application-layer tracing. Never subtract browser and server clocks.

An audio anchor is the input packet containing the nominal centre of attention
window 4 (the training-time current frame in [t-4, ..., t+3]). Its *callback*
timestamp is echoed, not an estimated ADC/sample timestamp or lip-sync label.
"""

import json
import math
import struct
from collections import deque


def unpack_audio_packet(packet, received_at):
    if len(packet) < 8 or packet[:4] != b"TGA1":
        raise ValueError("Missing TGA1 audio header")
    length = struct.unpack("!I", packet[4:8])[0]
    if not 0 < length <= 2048 or 8 + length >= len(packet):
        raise ValueError("Invalid audio metadata length")
    meta = json.loads(packet[8:8 + length].decode("utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Invalid audio metadata")
    session = meta.get("session")
    seq = meta.get("seq")
    capture = meta.get("capture_ms")
    sent = meta.get("sent_ms")
    if not isinstance(session, str) or not 1 <= len(session) <= 100:
        raise ValueError("Invalid trace session")
    if type(seq) is not int or seq < 0:
        raise ValueError("Invalid audio sequence")
    if any(type(v) not in (float, int) or not math.isfinite(v)
           for v in (capture, sent)) or not 0 <= capture <= sent:
        raise ValueError("Invalid browser timestamps")
    pcm = packet[8 + length:]
    if len(pcm) % 2:
        raise ValueError("PCM must contain complete int16 samples")
    return pcm, {"session": session, "seq": seq, "capture_ms": capture,
                 "sent_ms": sent, "received_at": received_at}


class AudioTraceTimeline:
    """Track received PCM spans through reblocking and rolling inference."""

    def __init__(self, retain_samples):
        self.spans = deque()
        self.samples = 0
        self.retain_samples = retain_samples

    def append(self, count, metadata):
        self.spans.append((self.samples, self.samples + count, metadata))
        self.samples += count
        while self.spans and self.spans[0][1] < self.samples - self.retain_samples:
            self.spans.popleft()

    def feature_trace(self, center_sample, feature_id, started, finished):
        for left, right, metadata in self.spans:
            if left <= center_sample < right:
                if metadata is None:
                    return None
                return dict(metadata, feature_id=feature_id,
                            anchor_sample=int(center_sample),
                            asr_started=started, asr_finished=finished)
        return None


def frame_metadata(trace, sending_at):
    if not trace:
        return {}
    # All server endpoints use the host monotonic clock, including the worker.
    return dict(trace, sending_at=sending_at)
