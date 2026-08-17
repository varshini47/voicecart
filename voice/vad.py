"""Streaming speech end-pointing on top of webrtcvad.

Milestone 3.1: turns continuous mic audio into discrete utterances without a
manual stop button. The client streams fixed-size raw PCM frames over the
WebSocket; Endpointer runs frame-level voice activity detection and decides
when an utterance has started and, after enough trailing silence, when it's
finished — that's the "VAD (silence detection)" piece from CLAUDE.md's
Week 3 plan. faster-whisper itself has no notion of streaming, so the actual
transcription still happens once, on the finalized utterance (see
agent/ws_stream.py) — this module only draws the utterance boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono PCM

START_SPEECH_FRAMES = 3  # ~90ms of speech before an utterance is considered started
# ~9s of trailing silence before automatic VAD finalization fires. This used
# to be 1.2s, but even at that length, automatic end-pointing kept firing on
# ordinary mid-sentence pauses (or brief mic/room noise) *while the user was
# still talking* — chopping real speech into meaningless fragments and, once
# a manual "Done talking" control (agent/ws_stream.py's "finalize" message)
# was added, actively fighting it: by the time the user clicked the button,
# VAD had already auto-finalized most of what they'd said as junk fragments,
# leaving only a tiny leftover. VAD auto-detection is now a rare fallback
# for an abandoned/forgotten mic, not something normal speech should ever
# trigger — the button is the real control.
END_SILENCE_FRAMES = 300
TRAILING_SILENCE_KEEP_FRAMES = 8  # ~240ms of trailing silence kept in the finalized clip
MAX_UTTERANCE_FRAMES = 1000  # ~30s safety cap so a stuck-open mic can't buffer forever
MIN_SPEECH_FRAMES = 10  # ~300ms of actual detected speech required to finalize at all.
# START_SPEECH_FRAMES (90ms) is short enough that a noise blip can be
# misclassified as speech starting; if nothing real follows, auto-VAD still
# finalizes ~9s later on a clip that's almost entirely silence, and Whisper
# hallucinates plausible boilerplate ("Thank you.") on such clips — the same
# failure mode as the short-utterance bug this trims for, just triggered by
# noise instead of real speech. Below this threshold, treat it as a false
# start and discard rather than transcribing it.


class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: bytes) -> bool: ...


class WebRtcVad:
    """Wraps webrtcvad so Endpointer doesn't depend on its (frame, sample_rate) call shape."""

    def __init__(self, aggressiveness: int = 3) -> None:
        # Level 3 is webrtcvad's most conservative-about-flagging-speech
        # setting (0-3 scale). Level 2 was classifying enough real-mic
        # background/room noise as "speech" that live utterances never saw
        # enough consecutive silence to finalize, running to the 30s safety
        # cap instead of a normal few-second utterance — a real-audio issue
        # that clean synthesized test audio (Piper output) never surfaced.
        import webrtcvad

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        return self._vad.is_speech(frame, SAMPLE_RATE)


@dataclass
class Endpointer:
    """Feed it one fixed-size PCM frame at a time; it returns the finalized
    utterance's PCM bytes once enough trailing silence has been seen, or
    None if the utterance isn't finished (or hasn't started) yet.
    """

    vad: VoiceActivityDetector = field(default_factory=WebRtcVad)
    _in_speech: bool = field(default=False, init=False)
    _consecutive_speech: int = field(default=0, init=False)
    _consecutive_silence: int = field(default=0, init=False)
    _speech_frame_total: int = field(default=0, init=False)
    _buffer: list[bytes] = field(default_factory=list, init=False)

    def accept_frame(self, frame: bytes) -> bytes | None:
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"expected {FRAME_BYTES}-byte frames, got {len(frame)}")

        speech = self.vad.is_speech(frame)

        if not self._in_speech:
            if speech:
                self._consecutive_speech += 1
                self._speech_frame_total += 1
                self._buffer.append(frame)
                if self._consecutive_speech >= START_SPEECH_FRAMES:
                    self._in_speech = True
                    self._consecutive_silence = 0
            else:
                # Leading silence/noise blips before real speech starts: drop them.
                self._consecutive_speech = 0
                self._speech_frame_total = 0
                self._buffer.clear()
            return None

        self._buffer.append(frame)
        if speech:
            self._speech_frame_total += 1
        self._consecutive_silence = 0 if speech else self._consecutive_silence + 1

        if self._consecutive_silence >= END_SILENCE_FRAMES or len(self._buffer) >= MAX_UTTERANCE_FRAMES:
            return self._finalize()
        return None

    def _finalize(self) -> bytes | None:
        # The buffer's tail is however much trailing silence was needed to
        # detect end-of-speech (up to END_SILENCE_FRAMES) — trim most of it
        # off before returning. Whisper hallucinates plausible-sounding
        # boilerplate ("Thank you.", "Thanks for watching!") when a short
        # utterance is dominated by silence rather than speech; keeping only
        # a small buffer of trailing silence fixes this without touching
        # longer utterances, where the untrimmed tail was a small fraction
        # of the total anyway.
        trim = max(0, min(self._consecutive_silence, END_SILENCE_FRAMES) - TRAILING_SILENCE_KEEP_FRAMES)
        frames = self._buffer[: len(self._buffer) - trim] if trim else self._buffer
        enough_speech = self._speech_frame_total >= MIN_SPEECH_FRAMES
        audio = b"".join(frames)
        self._reset()
        return audio if enough_speech else None

    def _reset(self) -> None:
        self._buffer.clear()
        self._in_speech = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._speech_frame_total = 0

    def force_finalize(self) -> bytes | None:
        """Finalize whatever's buffered right now, regardless of silence —
        for a manual "done talking" signal instead of waiting on VAD. No
        silence to trim here (the user, not a pause, marked the end), so the
        raw buffer is returned as-is. None if nothing was buffered, or if
        what was buffered doesn't clear MIN_SPEECH_FRAMES (e.g. the button
        was clicked without saying anything).
        """
        if not self._buffer:
            return None
        enough_speech = self._speech_frame_total >= MIN_SPEECH_FRAMES
        audio = b"".join(self._buffer)
        self._reset()
        return audio if enough_speech else None
