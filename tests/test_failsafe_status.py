"""When classification keeps failing, the status file must not keep saying OPEN.

A failed pass leaves the file alone. That is fine for one glitch, but when the
camera feed dies the last good line - possibly OPEN - used to stay in the file
all night with a timestamp that only a careful reader would notice was stale.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

import synthetic_roofstatus as srs


@pytest.fixture
def failsafe(app, tmp_path):
    app._classify_lock = threading.RLock()
    app._last_monitor_config = None
    app._failsafe_active = False
    app.override = None
    app.get_manual_override = lambda: app.override
    app.output = tmp_path / "RoofStatusFile.txt"
    app.config = {"output_path": str(app.output)}
    return app


def ago(seconds):
    return datetime.now(timezone.utc) - timedelta(seconds=seconds)


def test_recent_good_pass_leaves_the_file_alone(failsafe):
    failsafe._last_good_pass_at = ago(srs.FAILSAFE_AFTER_SECONDS - 30)

    assert failsafe._apply_failsafe_status(failsafe.config) is False
    assert not failsafe.output.exists()


def test_overdue_classification_writes_closed(failsafe):
    failsafe.output.write_text("???2026-09-20 01:00:00AM Roof Status: OPEN\n")
    failsafe._last_good_pass_at = ago(srs.FAILSAFE_AFTER_SECONDS + 1)

    assert failsafe._apply_failsafe_status(failsafe.config) is True

    line = failsafe.output.read_text()
    assert "Roof Status: CLOSED" in line
    assert "failsafe" in line
    assert "OPEN" not in line
    assert failsafe._failsafe_active is True


def test_never_classified_is_overdue(failsafe):
    failsafe._last_good_pass_at = None

    assert failsafe._apply_failsafe_status(failsafe.config) is True
    assert "Roof Status: CLOSED" in failsafe.output.read_text()


def test_manual_override_still_wins(failsafe):
    """An override is the tool for "the camera is obscured and I know better"."""
    failsafe._last_good_pass_at = ago(srs.FAILSAFE_AFTER_SECONDS + 60)
    failsafe.override = "OPEN"

    assert failsafe._apply_failsafe_status(failsafe.config) is True
    assert "Roof Status: OPEN (Manual override: OPEN)" in failsafe.output.read_text()


def test_falls_back_to_the_last_known_output_path(failsafe):
    """A pass that failed to get a config snapshot still knows where to write."""
    failsafe._last_good_pass_at = None
    failsafe._last_monitor_config = failsafe.config

    assert failsafe._apply_failsafe_status(None) is True
    assert failsafe.output.exists()


def test_no_output_path_means_no_write(failsafe):
    failsafe._last_good_pass_at = None

    assert failsafe._apply_failsafe_status(None) is False


def test_failed_write_is_reported(failsafe):
    failsafe._last_good_pass_at = None
    failsafe._write_status_file = lambda *a, **k: False

    assert failsafe._apply_failsafe_status(failsafe.config) is False
    assert failsafe._failsafe_active is False
