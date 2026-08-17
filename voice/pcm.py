"""Raw PCM <-> WAV helpers.

voice/stt.py decodes audio via faster-whisper, which uses PyAV under the
hood and needs a container/header to know the sample rate and bit depth —
it can't take headerless raw PCM. The WebSocket mic stream (voice/vad.py)
deals in raw int16 frames, so this wraps a finalized utterance's PCM bytes
in a minimal WAV header before handing it to stt.transcribe.
"""

from __future__ import annotations

import io
import wave

from voice.vad import SAMPLE_RATE


def pcm16_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buf.getvalue()
