"""Week 0 sanity check: transcribe a recording of your own voice with faster-whisper.

Record a few seconds of yourself talking (Windows Voice Recorder is fine, any
format works - faster-whisper decodes via ffmpeg/PyAV) and run:

  .venv/Scripts/python.exe voice/stt_check.py path/to/recording.m4a
"""

import sys

from faster_whisper import WhisperModel

if len(sys.argv) != 2:
    sys.exit("Usage: python voice/stt_check.py <path-to-audio-file>")

audio_path = sys.argv[1]

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_path)

print(f"Detected language: {info.language} (p={info.language_probability:.2f})")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
