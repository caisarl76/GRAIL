from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_WRITER_PATH = REPO_ROOT / "imports/SONIC/gear_sonic/data/video_writer.py"


class _FakePacket:
    pass


class _FakeVideoFrame:
    @staticmethod
    def from_ndarray(frame, format):
        return {"frame": frame, "format": format}


class _FakeStream:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.encode_started = threading.Event()
        self.allow_encode_finish = threading.Event()
        self.frame_encode_done = threading.Event()
        self.flush_called = threading.Event()

    def encode(self, frame=None):
        if frame is None:
            if self.encode_started.is_set() and not self.frame_encode_done.is_set():
                raise RuntimeError("flush called while frame encode is active")
            self.flush_called.set()
            return [_FakePacket()]

        self.encode_started.set()
        self.allow_encode_finish.wait(timeout=2)
        self.frame_encode_done.set()
        return [_FakePacket()]


class _FakeContainer:
    def __init__(self):
        self.stream = _FakeStream()
        self.closed = False

    def add_stream(self, codec, rate):
        return self.stream

    def mux(self, packet):
        pass

    def close(self):
        self.closed = True


def _load_video_writer_with_fake_av(monkeypatch):
    container = _FakeContainer()
    fake_av = ModuleType("av")
    fake_av.VideoFrame = _FakeVideoFrame
    fake_av.open = lambda *args, **kwargs: container
    monkeypatch.setitem(sys.modules, "av", fake_av)

    spec = importlib.util.spec_from_file_location("test_video_writer_module", VIDEO_WRITER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.VideoWriter, container


def test_video_writer_stop_waits_for_worker_before_flushing(monkeypatch, tmp_path):
    VideoWriter, container = _load_video_writer_with_fake_av(monkeypatch)
    writer = VideoWriter(str(tmp_path / "out.mp4"), width=8, height=8, fps=25)
    writer._first_frame = False
    writer.add_frame(np.zeros((8, 8, 3), dtype=np.uint8))

    assert container.stream.encode_started.wait(timeout=1)
    errors = []

    def stop_writer():
        try:
            writer.stop()
        except Exception as exc:
            errors.append(exc)

    stop_thread = threading.Thread(target=stop_writer)
    stop_thread.start()
    time.sleep(0.05)

    assert stop_thread.is_alive()
    assert not container.stream.flush_called.is_set()

    container.stream.allow_encode_finish.set()
    stop_thread.join(timeout=1)

    assert not stop_thread.is_alive()
    assert errors == []
    assert container.stream.flush_called.is_set()
    assert container.closed is True
