# Deployed (text-mode) image — Week 4, Milestone 4.2. Deliberately excludes
# voice/ (faster-whisper, piper-tts) and their model weights: voice runs
# locally in demos, and Whisper's RAM footprint doesn't fit a free-tier
# instance. See agent/main_text.py for the voice-free entrypoint this runs.
#
# Build from the repo root: docker build -f infra/Dockerfile -t voicecart .
FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY agent/ agent/
COPY mcp_commerce/ mcp_commerce/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "agent.main_text:app", "--host", "0.0.0.0", "--port", "8000"]
