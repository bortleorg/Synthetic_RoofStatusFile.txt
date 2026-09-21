"""A manual override must reach the ASCOM IsSafe flag straight away.

ASCOM read the cached classification, which only picks an override up on the
next monitoring pass. Force CLOSED therefore left IsSafe True for up to a minute
and a half, and indefinitely while monitoring was stopped.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

from ascom_alpaca_safety import AscomAlpacaSafetyMonitor


class Var:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


@pytest.fixture
def app(app, monkeypatch):
    import synthetic_roofstatus as srs

    monkeypatch.setattr(srs.messagebox, "showinfo", lambda *a, **k: None)
    app._classify_lock = threading.RLock()
    app._last_classification = ("frame.png", "OPEN", datetime.now(timezone.utc))
    app.override_mode = Var("AUTO")
    app.override_duration = Var("1 hour")
    app.save_settings = lambda: None
    app._update_override_display = lambda: None
    app._write_status_file = lambda *a, **k: True
    app.refreshes = 0
    app._refresh_status_now = lambda: setattr(app, "refreshes", app.refreshes + 1)
    app.is_sun_safe_for_open = lambda config=None: True
    return app


@pytest.fixture
def server(app):
    server = AscomAlpacaSafetyMonitor(port=0, device_number=0, roof_classifier_app=app,
                                      start_background=False)
    server.connected = True
    server.refresh_safety_status()
    app.ascom_server = server
    return server


def test_force_closed_makes_issafe_false_immediately(app, server):
    assert server.is_safe is True

    app.override_mode.set("CLOSED")
    app.apply_manual_override()

    assert server.is_safe is False


def test_force_closed_is_reported_while_monitoring_is_stopped(app):
    app._last_classification = None
    app.override_active = "CLOSED"

    assert app.get_cached_status() == ("CLOSED", 0.0)


def test_force_open_is_reported_while_monitoring_is_stopped(app):
    """README: ASCOM clients see the forced status (the sun guard still applies)."""
    app._last_classification = None
    app.override_active = "OPEN"

    assert app.get_cached_status() == ("OPEN", 0.0)


def test_forced_open_still_respects_the_sun(app, server):
    app._last_classification = None
    app.is_sun_safe_for_open = lambda config=None: False
    app.override_mode.set("OPEN")

    app.apply_manual_override()

    assert server.is_safe is False


def test_clearing_does_not_leave_the_forced_status_standing(app, server):
    """The cached result carries the forced status; it must not outlive the override."""
    app.override_mode.set("OPEN")
    app.apply_manual_override()
    app._last_classification = ("frame.png", "OPEN", datetime.now(timezone.utc))

    app.clear_manual_override()

    assert app._last_classification is None
    assert server.is_safe is False
    assert app.refreshes == 1, "no re-classification requested after clearing"


def test_expired_override_drops_the_cached_forced_status(app):
    app._defer_to_ui = lambda func: None
    app.override_active = "OPEN"
    app.override_expiry = datetime.now() - timedelta(seconds=1)

    assert app.get_cached_status() == (None, None)
    assert app.override_active is None


def test_refresh_is_a_no_op_without_a_server(app):
    app.ascom_server = None
    app._refresh_ascom_safety()


def test_refresh_failure_is_logged_not_raised(app):
    class Broken:
        def refresh_safety_status(self):
            raise RuntimeError("boom")

    app.ascom_server = Broken()
    app._refresh_ascom_safety()
