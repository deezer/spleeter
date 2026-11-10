"""Tests for the lifetime of the process pool used for audio export."""

from multiprocessing import Pool, active_children
from pathlib import Path
from time import monotonic, sleep
from wave import open as open_wave

import numpy as np
import pytest

from spleeter.separator import Separator


def _wait_for(condition):
    deadline = monotonic() + 5
    while not condition():
        if monotonic() >= deadline:
            return False
        sleep(0.01)
    return True


class _GatedAudioAdapter:
    def save(self, path, *_args):
        output = Path(path)
        output.with_suffix(".started").touch()
        if not _wait_for((output.parent / "release").exists):
            raise TimeoutError("The test did not release the writer")
        output.write_bytes(b"complete")


@pytest.fixture
def one_worker_pool(monkeypatch):
    monkeypatch.setattr("spleeter.separator.Pool", lambda: Pool(1))


def test_unused_pool_is_released(one_worker_pool):
    before = set(active_children())
    for _ in range(3):
        separator = Separator("spleeter:2stems")
        assert set(active_children()) - before
        del separator

    assert _wait_for(lambda: set(active_children()) <= before)


def test_pending_exports_finish_after_separator_is_released(tmp_path, one_worker_pool):
    before = set(active_children())
    separator = Separator("spleeter:2stems")
    waveform = np.zeros((16, 2), dtype=np.float32)
    separator.save_to_file(
        {"vocals": waveform, "accompaniment": waveform},
        "input.wav",
        str(tmp_path),
        audio_adapter=_GatedAudioAdapter(),
        synchronous=False,
    )
    output = tmp_path / "input"
    assert _wait_for((output / "vocals.started").exists)
    del separator
    (output / "release").touch()

    assert _wait_for(lambda: set(active_children()) <= before)
    for stem in ("vocals", "accompaniment"):
        assert (output / f"{stem}.wav").read_bytes() == b"complete"


def test_default_pool_exports_wav_files(tmp_path, one_worker_pool):
    separator = Separator("spleeter:2stems")
    separator.separate_to_file("audio_example.mp3", str(tmp_path), duration=2)

    for stem in ("vocals", "accompaniment"):
        with open_wave(str(tmp_path / "audio_example" / f"{stem}.wav"), "rb") as wav:
            assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (
                2,
                44100,
                88200,
            )
