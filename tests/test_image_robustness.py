"""Unreadable or vanishing frames must fail one pass cleanly, not crash it.

Cameras write the newest frame while we look for it, and cleanup scripts delete
old ones. cv2.imread returns None for a truncated file, which used to reach
cv2.resize as an opaque cv2.error and end the monitor thread.
"""

import os
import threading

import cv2
import numpy as np
import pytest

import synthetic_roofstatus as srs


class Var:
    """Minimal stand-in for a Tk variable."""

    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value


def write_png(path, value=128):
    cv2.imwrite(str(path), np.full((48, 64), value, dtype=np.uint8))
    return path


# ── prep_image ────────────────────────────────────────────────────────────────

def test_prep_image_returns_the_model_input_size(app, tmp_path):
    img = app.prep_image(str(write_png(tmp_path / "ok.png")))
    assert img.shape == (srs.IMG_SIZE, srs.IMG_SIZE)


def test_prep_image_raises_value_error_for_a_truncated_file(app, tmp_path):
    good = write_png(tmp_path / "ok.png").read_bytes()
    bad = tmp_path / "half.png"
    bad.write_bytes(good[:20])

    with pytest.raises(ValueError, match="Could not read image"):
        app.prep_image(str(bad))


def test_prep_image_raises_value_error_for_a_missing_file(app, tmp_path):
    with pytest.raises(ValueError):
        app.prep_image(str(tmp_path / "gone.png"))


# ── a full pass over an unreadable frame ──────────────────────────────────────

@pytest.fixture
def pipeline(app, tmp_path):
    class Model:
        def predict(self, _x):
            return [1]

    app._classify_lock = threading.RLock()
    app._last_classification = None
    app._last_good_pass_at = None
    app._failsafe_active = False
    app.model = Model()
    app.last_image_hash = None
    app.samples = []
    app.save_sample_if_needed = lambda path, config=None: app.samples.append(path)
    app._capture_preview = lambda *a, **k: None
    app.writes = []
    app._write_status_file = lambda *a, **k: app.writes.append(a) or True
    app.config = {
        'camera_url': '', 'monitor_path': str(tmp_path),
        'output_path': str(tmp_path / "status.txt"),
        'save_on_disagreement': False, 'save_on_toggle': False,
    }
    return app


def test_unreadable_newest_frame_fails_the_pass_without_raising(pipeline, tmp_path):
    (tmp_path / "frame.png").write_bytes(b"\x89PNG\r\n\x1a\n truncated")

    filename, error = pipeline.classify_latest_png(pipeline.config, max_cache_age=0)

    assert filename is None
    assert "Could not read image" in error
    assert pipeline.writes == [], "status file written for an unreadable frame"
    assert pipeline.samples == [], "unreadable frame sampled into the training set"
    assert pipeline._last_classification is None


def test_temporary_download_is_deleted_when_the_frame_is_unreadable(pipeline, tmp_path):
    download = tmp_path / "download.jpg"
    download.write_bytes(b"<html>502 Bad Gateway</html>")
    pipeline._resolve_latest_image = lambda url, folder: (
        str(download), "http://cam/latest.jpg", True, None)

    filename, _error = pipeline.classify_latest_png(pipeline.config, max_cache_age=0)

    assert filename is None
    assert not download.exists(), "temp download leaked"


def test_temporary_download_is_deleted_when_classification_raises(pipeline, tmp_path):
    download = write_png(tmp_path / "download.png")
    pipeline._resolve_latest_image = lambda url, folder: (
        str(download), "http://cam/latest.png", True, None)

    class Broken:
        def predict(self, _x):
            raise RuntimeError("model file corrupt")
    pipeline.model = Broken()

    with pytest.raises(RuntimeError):
        pipeline.classify_latest_png(pipeline.config, max_cache_age=0)
    assert not download.exists(), "temp download leaked"


# ── _resolve_latest_image ─────────────────────────────────────────────────────

def test_newest_frame_is_chosen(app, tmp_path):
    old = write_png(tmp_path / "old.png")
    new = write_png(tmp_path / "new.jpg")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    (tmp_path / "notes.txt").write_text("ignored")

    path, caption, is_temp, error = app._resolve_latest_image("", str(tmp_path))

    assert caption == "new.jpg"
    assert path == str(new)
    assert (is_temp, error) == (False, None)


def test_frame_deleted_during_the_scan_is_skipped(app, tmp_path, monkeypatch):
    keep = write_png(tmp_path / "keep.png")
    write_png(tmp_path / "vanishing.png")
    real_getmtime = os.path.getmtime

    def getmtime(path):
        if os.path.basename(path) == "vanishing.png":
            raise FileNotFoundError(path)
        return real_getmtime(path)

    monkeypatch.setattr(srs.os.path, "getmtime", getmtime)

    path, caption, _is_temp, error = app._resolve_latest_image("", str(tmp_path))

    assert error is None
    assert path == str(keep)


def test_folder_with_only_vanished_frames_reports_no_images(app, tmp_path, monkeypatch):
    write_png(tmp_path / "vanishing.png")

    def getmtime(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(srs.os.path, "getmtime", getmtime)

    path, _caption, _is_temp, error = app._resolve_latest_image("", str(tmp_path))

    assert path is None
    assert error == "No image files found"


# ── training data ─────────────────────────────────────────────────────────────

def test_training_skips_unreadable_images(app, tmp_path):
    app.training_data_folder = Var(str(tmp_path))
    (tmp_path / "open").mkdir()
    (tmp_path / "closed").mkdir()
    write_png(tmp_path / "open" / "a.png", 200)
    write_png(tmp_path / "closed" / "b.png", 20)
    (tmp_path / "closed" / "broken.png").write_bytes(b"nope")

    X, y, skipped = app._load_training_data()

    assert sorted(y) == [0, 1]
    assert len(X) == 2
    assert skipped == ["closed/broken.png"]


def test_duplicate_hashing_tolerates_unreadable_images(app, tmp_path):
    write_png(tmp_path / "a.png")
    (tmp_path / "broken.png").write_bytes(b"nope")

    assert len(app.get_existing_hashes(str(tmp_path))) == 1
