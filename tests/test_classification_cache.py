"""Classification is serialised and its result is shared between callers.

The monitor thread and the ASCOM server thread both want the current roof state.
Letting both run the pipeline meant two writers on the status file and a shared
toggle/disagreement baseline being advanced twice per cycle.
"""

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import synthetic_roofstatus as srs


@pytest.fixture
def classifier(app):
    """App fixture with a stubbed classification pass that records its calls."""
    app._classify_lock = threading.RLock()
    app._last_classification = None
    app.calls = []

    def fake_pass(config=None):
        app.calls.append(config)
        status = "OPEN" if len(app.calls) % 2 else "CLOSED"
        filename = f"frame{len(app.calls)}.png"
        app._last_classification = (filename, status, datetime.now(timezone.utc))
        return filename, status

    app._classify_latest_png_uncached = fake_pass
    return app


def test_first_call_runs_the_pipeline(classifier):
    filename, status = classifier.classify_latest_png({'k': 'v'})

    assert (filename, status) == ("frame1.png", "OPEN")
    assert classifier.calls == [{'k': 'v'}]


def test_second_call_reuses_the_cached_result(classifier):
    first = classifier.classify_latest_png({})
    second = classifier.classify_latest_png({})

    assert first == second
    assert len(classifier.calls) == 1, "pipeline ran twice for one cycle"


def test_zero_max_age_forces_a_fresh_pass(classifier):
    """The monitor loop and a post-override refresh must not see a stale answer."""
    classifier.classify_latest_png({})
    _filename, status = classifier.classify_latest_png({}, max_cache_age=0)

    assert len(classifier.calls) == 2
    assert status == "CLOSED"


def test_expired_cache_triggers_a_fresh_pass(classifier):
    classifier.classify_latest_png({})
    filename, status, taken_at = classifier._last_classification
    classifier._last_classification = (
        filename, status, taken_at - timedelta(seconds=srs.CLASSIFICATION_CACHE_SECONDS + 5)
    )

    classifier.classify_latest_png({})

    assert len(classifier.calls) == 2


def test_failed_fresh_pass_clears_the_cached_result(classifier):
    """A monitor cycle that cannot classify must not leave the old OPEN standing."""
    classifier.classify_latest_png({})
    classifier._classify_latest_png_uncached = lambda config=None: (None, "No image")

    result = classifier.classify_latest_png({}, max_cache_age=0)

    assert result == (None, "No image")
    assert classifier._last_classification is None
    assert classifier.get_cached_status() == (None, None)


def test_raising_fresh_pass_clears_the_cached_result(classifier):
    classifier.classify_latest_png({})

    def boom(config=None):
        raise RuntimeError("model exploded")

    classifier._classify_latest_png_uncached = boom

    with pytest.raises(RuntimeError):
        classifier.classify_latest_png({}, max_cache_age=0)
    assert classifier._last_classification is None


def test_concurrent_callers_do_not_interleave(app):
    """Two threads asking at once produce exactly one pipeline run at a time."""
    app._classify_lock = threading.RLock()
    app._last_classification = None
    overlaps = []
    active = []
    lock = threading.Lock()

    def slow_pass(config=None):
        with lock:
            active.append(1)
            if len(active) > 1:
                overlaps.append(len(active))
        time.sleep(0.05)
        with lock:
            active.pop()
        app._last_classification = ("f.png", "OPEN", datetime.now(timezone.utc))
        return "f.png", "OPEN"

    app._classify_latest_png_uncached = slow_pass

    threads = [
        threading.Thread(target=lambda: app.classify_latest_png({}, max_cache_age=0))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert overlaps == [], "classification passes overlapped"


# ── get_cached_status: what the ASCOM safety monitor consumes ─────────────────

def test_cached_status_is_none_before_anything_is_classified(app):
    app._last_classification = None
    assert app.get_cached_status() == (None, None)


def test_cached_status_returns_a_fresh_result(app):
    app._last_classification = ("f.png", "OPEN", datetime.now(timezone.utc))

    status, age = app.get_cached_status()

    assert status == "OPEN"
    assert age < 5


def test_cached_status_refuses_a_stale_result(app):
    """A stale classification must not keep being reported as authoritative."""
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=srs.CLASSIFICATION_MAX_AGE_SECONDS + 60)
    app._last_classification = ("f.png", "OPEN", stale)

    status, age = app.get_cached_status()

    assert status is None
    assert age > srs.CLASSIFICATION_MAX_AGE_SECONDS


def test_cached_status_honours_a_custom_max_age(app):
    app._last_classification = (
        "f.png", "OPEN", datetime.now(timezone.utc) - timedelta(seconds=30))

    assert app.get_cached_status(60)[0] == "OPEN"
    assert app.get_cached_status(10)[0] is None


# ── a pass whose status file write fails ──────────────────────────────────────

def _stub_pipeline(app, tmp_path, write_ok):
    """Stub everything a real pass touches except the result bookkeeping."""
    import numpy as np

    image = tmp_path / "frame.png"
    image.write_bytes(b"not really a png")

    class Model:
        def predict(self, _x):
            return [1]  # OPEN

    app._classify_lock = threading.RLock()
    app._last_classification = None
    app._last_good_pass_at = None
    app._failsafe_active = False
    app.model = Model()
    app.last_image_hash = None
    app.previous_classified_status = None
    app._in_disagreement = False
    app._resolve_latest_image = lambda url, path: (str(image), "frame.png", False, None)
    app.save_sample_if_needed = lambda *a, **k: None
    app._capture_preview = lambda *a, **k: None
    app.read_secondary_source = lambda config: (None, None)
    app.prep_image = lambda path: np.zeros((2, 2))
    app.is_sun_safe_for_open = lambda config=None: True
    app.get_manual_override = lambda: None
    app.calculate_sun_angle = lambda config=None: -30.0
    app._save_frame_for_review = lambda *a, **k: None
    app._write_status_file = lambda *a, **k: write_ok
    return {
        'camera_url': '', 'monitor_path': str(tmp_path),
        'output_path': str(tmp_path / "status.txt"),
        'save_on_disagreement': False, 'save_on_toggle': False,
    }


def test_successful_write_publishes_the_result(app, tmp_path):
    config = _stub_pipeline(app, tmp_path, write_ok=True)

    assert app.classify_latest_png(config, max_cache_age=0) == ("frame.png", "OPEN")
    assert app.get_cached_status()[0] == "OPEN"


def test_failed_write_does_not_publish_the_result(app, tmp_path):
    """ASCOM must not report a status the roof status file never received."""
    config = _stub_pipeline(app, tmp_path, write_ok=False)
    app._last_classification = ("old.png", "CLOSED", datetime.now(timezone.utc))

    filename, error = app.classify_latest_png(config, max_cache_age=0)

    assert filename is None
    assert "status file" in error
    assert app.get_cached_status() == (None, None)


def test_successful_pass_lifts_the_failsafe(app, tmp_path):
    """Once classification recovers, the fail-safe grace period restarts."""
    config = _stub_pipeline(app, tmp_path, write_ok=True)
    app._failsafe_active = True

    app.classify_latest_png(config, max_cache_age=0)

    assert app._failsafe_active is False
    assert app._last_good_pass_at == app._last_classification[2]


def test_failed_pass_does_not_extend_the_failsafe_grace(app, tmp_path):
    config = _stub_pipeline(app, tmp_path, write_ok=False)

    app.classify_latest_png(config, max_cache_age=0)

    assert app._last_good_pass_at is None
