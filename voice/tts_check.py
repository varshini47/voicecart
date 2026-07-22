"""Week 0 sanity check: synthesize one WAV with Piper to confirm local TTS works.

Run: .venv/Scripts/python.exe voice/tts_check.py
"""

import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL = MODEL_DIR / "en_US-lessac-medium.onnx"
CONFIG = MODEL_DIR / "en_US-lessac-medium.onnx.json"
OUT_WAV = Path(__file__).parent / "tts_check_out.wav"
TEXT = "Hello, this is a test of the VoiceCart text to speech pipeline."

if not MODEL.exists():
    sys.exit(
        f"Voice model not found at {MODEL}. Run:\n"
        f"  python -m piper.download_voices --download-dir voice/models en_US-lessac-medium"
    )

result = subprocess.run(
    [
        sys.executable, "-m", "piper",
        "-m", str(MODEL),
        "-c", str(CONFIG),
        "-f", str(OUT_WAV),
    ],
    input=TEXT,
    text=True,
    capture_output=True,
)

print(result.stdout)
print(result.stderr)

if result.returncode == 0 and OUT_WAV.exists():
    print(f"OK: wrote {OUT_WAV} ({OUT_WAV.stat().st_size} bytes). Play it to confirm audio.")
else:
    sys.exit("Piper synthesis failed, see output above.")
