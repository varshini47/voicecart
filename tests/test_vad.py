"""Tests for voice.vad.Endpointer's state machine. A fake VAD (a fixed
queue of speech/silence booleans) drives it deterministically instead of
real audio, so these tests exercise the start/end-pointing logic itself,
not webrtcvad's classifier.
"""

from __future__ import annotations

import pytest

from voice.vad import (
    END_SILENCE_FRAMES,
    FRAME_BYTES,
    MIN_SPEECH_FRAMES,
    START_SPEECH_FRAMES,
    TRAILING_SILENCE_KEEP_FRAMES,
    Endpointer,
)

FRAME = b"\x00" * FRAME_BYTES

# A speech run long enough to clear MIN_SPEECH_FRAMES, for tests exercising
# behavior (trailing-silence trim, reset, etc.) other than the minimum-speech
# gate itself. START_SPEECH_FRAMES alone is too short for that now.
SPEECH_FRAMES = MIN_SPEECH_FRAMES


class FakeVad:
    def __init__(self, pattern: list[bool]) -> None:
        self._pattern = list(pattern)

    def is_speech(self, frame: bytes) -> bool:
        return self._pattern.pop(0)


def _feed(endpointer: Endpointer, n: int) -> bytes | None:
    result = None
    for _ in range(n):
        result = endpointer.accept_frame(FRAME)
    return result


def test_leading_silence_is_dropped_and_never_finalizes() -> None:
    endpointer = Endpointer(vad=FakeVad([False] * 50))

    for _ in range(50):
        assert endpointer.accept_frame(FRAME) is None


def test_short_noise_blip_does_not_start_an_utterance() -> None:
    # One speech frame, then back to silence — short of START_SPEECH_FRAMES.
    pattern = [True] * (START_SPEECH_FRAMES - 1) + [False] * 50
    endpointer = Endpointer(vad=FakeVad(pattern))

    result = _feed(endpointer, len(pattern))

    assert result is None


def test_utterance_finalizes_after_trailing_silence() -> None:
    pattern = [True] * SPEECH_FRAMES + [False] * END_SILENCE_FRAMES
    endpointer = Endpointer(vad=FakeVad(pattern))

    result = _feed(endpointer, len(pattern) - 1)
    assert result is None  # not enough trailing silence yet

    last = endpointer.accept_frame(FRAME)
    # Trailing silence beyond TRAILING_SILENCE_KEEP_FRAMES is trimmed off —
    # see test_trailing_silence_is_trimmed for why.
    expected_frames = SPEECH_FRAMES + TRAILING_SILENCE_KEEP_FRAMES
    assert last == FRAME * expected_frames


def test_trailing_silence_is_trimmed() -> None:
    # A short utterance with the full END_SILENCE_FRAMES tail shouldn't come
    # back dominated by silence — that's exactly what made faster-whisper
    # hallucinate boilerplate text ("Thank you.") for short live utterances.
    pattern = [True] * SPEECH_FRAMES + [False] * END_SILENCE_FRAMES
    endpointer = Endpointer(vad=FakeVad(pattern))

    result = _feed(endpointer, len(pattern))

    assert len(result) == (SPEECH_FRAMES + TRAILING_SILENCE_KEEP_FRAMES) * FRAME_BYTES


def test_speech_resets_silence_counter_before_threshold() -> None:
    # Speech starts, a brief pause short of the threshold, then speech again,
    # then a full silence run — should only finalize once, after the second run.
    pattern = (
        [True] * SPEECH_FRAMES
        + [False] * (END_SILENCE_FRAMES - 1)
        + [True]
        + [False] * END_SILENCE_FRAMES
    )
    endpointer = Endpointer(vad=FakeVad(pattern))

    result = _feed(endpointer, len(pattern) - 1)
    assert result is None

    last = endpointer.accept_frame(FRAME)
    # SPEECH_FRAMES initial speech + the mid-utterance blip that reset
    # the silence counter (1 speech frame + END_SILENCE_FRAMES - 1 silence
    # frames before it) + TRAILING_SILENCE_KEEP_FRAMES of the final tail.
    expected_frames = SPEECH_FRAMES + (END_SILENCE_FRAMES - 1) + 1 + TRAILING_SILENCE_KEEP_FRAMES
    assert last == FRAME * expected_frames


def test_noise_blip_that_auto_finalizes_is_discarded() -> None:
    # A blip just long enough to start an "utterance" (START_SPEECH_FRAMES)
    # but short of MIN_SPEECH_FRAMES, followed by enough silence to trigger
    # auto-finalization, should be discarded rather than returned — this is
    # almost always background noise misclassified as speech, and Whisper
    # hallucinates boilerplate text on the resulting silence-dominated clip.
    assert START_SPEECH_FRAMES < MIN_SPEECH_FRAMES
    pattern = [True] * START_SPEECH_FRAMES + [False] * END_SILENCE_FRAMES
    endpointer = Endpointer(vad=FakeVad(pattern))

    result = _feed(endpointer, len(pattern))

    assert result is None


def test_wrong_frame_size_raises() -> None:
    endpointer = Endpointer(vad=FakeVad([True]))

    with pytest.raises(ValueError):
        endpointer.accept_frame(b"\x00" * (FRAME_BYTES - 1))


def test_force_finalize_returns_buffered_audio_without_waiting_for_silence() -> None:
    # Still speaking (no trailing silence at all) — accept_frame alone
    # would never finalize this, but force_finalize should return it as-is.
    pattern = [True] * (SPEECH_FRAMES + 5)
    endpointer = Endpointer(vad=FakeVad(pattern))

    for _ in range(len(pattern)):
        assert endpointer.accept_frame(FRAME) is None

    result = endpointer.force_finalize()
    assert result == FRAME * len(pattern)


def test_force_finalize_with_nothing_buffered_returns_none() -> None:
    endpointer = Endpointer(vad=FakeVad([]))

    assert endpointer.force_finalize() is None


def test_force_finalize_with_too_little_speech_is_discarded() -> None:
    # e.g. the "Done talking" button clicked without having said anything.
    pattern = [True] * (MIN_SPEECH_FRAMES - 1)
    endpointer = Endpointer(vad=FakeVad(pattern))
    _feed(endpointer, len(pattern))

    assert endpointer.force_finalize() is None


def test_force_finalize_resets_state_for_next_utterance() -> None:
    pattern = [True] * SPEECH_FRAMES
    endpointer = Endpointer(vad=FakeVad(pattern))
    _feed(endpointer, len(pattern))
    endpointer.force_finalize()

    assert endpointer.force_finalize() is None  # buffer was cleared


def test_resets_after_finalizing_for_a_second_utterance() -> None:
    pattern = [True] * SPEECH_FRAMES + [False] * END_SILENCE_FRAMES
    endpointer = Endpointer(vad=FakeVad(pattern * 2))

    first = _feed(endpointer, len(pattern))
    second = _feed(endpointer, len(pattern))

    expected_frames = SPEECH_FRAMES + TRAILING_SILENCE_KEEP_FRAMES
    assert first == FRAME * expected_frames
    assert second == FRAME * expected_frames
