"""The monitor loop must survive a bad pass, and a stopped run must stay stopped.

Before this, any exception in a pass (a half-written frame, a file deleted
mid-scan) propagated out of monitor_loop and silently ended the thread. The UI
kept saying "Monitoring: Active" while the roof status file froze on its last
line. Separately, Stop followed quickly by Start could let the old loop's exit
reset the new run's state.
"""

import threading
import time

import pytest

import synthetic_roofstatus as srs


class FakeRoot:
    """Stands in for Tk: after(0, ...) runs the callback straight away."""

    def after(self, ms, func=None, *args):
        if ms == 0 and func is not None:
            func(*args)
        return "after#1"

    def after_cancel(self, _after_id):
        pass


@pytest.fixture
def looper(app, monkeypatch):
    monkeypatch.setattr(srs, "MONITOR_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(srs, "_COUNTDOWN_TICK_SECONDS", 0.005)

    app.root = FakeRoot()
    app._monitor_stop_event = threading.Event()
    app.updates = []
    app.failsafe_calls = []
    app.notifications = []
    app.cleared = 0

    app._request_monitor_config = lambda: {"output_path": "status.txt"}
    app.update_monitoring_status = lambda f, s: app.updates.append((f, s))
    app.update_countdown = lambda remaining: None
    app._apply_failsafe_status = lambda config=None: app.failsafe_calls.append(config)
    app._check_and_send_notifications = lambda status, config=None: app.notifications.append(status)

    def clear():
        app.cleared += 1
    app.clear_monitoring_status = clear
    return app


def run_loop(app, stop_event=None, timeout=5):
    thread = threading.Thread(target=app.monitor_loop, args=(stop_event,), daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "monitor loop did not stop"


def test_loop_survives_an_exception_in_a_pass(looper):
    calls = []

    def classify(config, max_cache_age=None):
        calls.append(config)
        if len(calls) == 1:
            raise RuntimeError("cv2.error: !ssize.empty()")
        if len(calls) == 3:
            looper._monitor_stop_event.set()
        return "frame.png", "OPEN"

    looper.classify_latest_png = classify

    run_loop(looper)

    assert len(calls) == 3, "loop ended after the failing pass"
    assert looper.updates[0][0] is None
    assert "cv2.error" in looper.updates[0][1]
    assert looper.updates[1] == ("frame.png", "OPEN")
    assert looper.notifications == ["OPEN"], "notifications sent for the stopped pass"


def stop_after_updates(app, count):
    """Make update_monitoring_status stop the loop after *count* reports."""
    record = app.update_monitoring_status

    def update(f, s):
        record(f, s)
        if len(app.updates) >= count:
            app._monitor_stop_event.set()
    app.update_monitoring_status = update


def test_failed_pass_triggers_the_failsafe_check(looper):
    looper.classify_latest_png = lambda config, max_cache_age=None: (
        None, "Failed to fetch image from URL")
    stop_after_updates(looper, 2)

    run_loop(looper)

    assert looper.failsafe_calls == [{"output_path": "status.txt"}] * 2


def test_successful_pass_does_not_touch_the_failsafe(looper):
    looper.classify_latest_png = lambda config, max_cache_age=None: ("frame.png", "CLOSED")
    stop_after_updates(looper, 2)

    run_loop(looper)

    assert looper.failsafe_calls == []


def test_exception_in_notifications_does_not_end_the_loop(looper):
    passes = []

    def classify(config, max_cache_age=None):
        passes.append(1)
        if len(passes) == 2:
            looper._monitor_stop_event.set()
        return "frame.png", "OPEN"

    def explode(status, config=None):
        raise ValueError("bad webhook config")

    looper.classify_latest_png = classify
    looper._check_and_send_notifications = explode

    run_loop(looper)

    assert len(passes) == 2


def test_pass_that_finishes_after_stop_reports_nothing(looper):
    """A pass still running when Stop is pressed must not write into the UI."""
    def classify(config, max_cache_age=None):
        looper._monitor_stop_event.set()
        return "frame.png", "OPEN"

    looper.classify_latest_png = classify

    run_loop(looper)

    assert looper.updates == []
    assert looper.notifications == []


def test_stop_interrupts_the_countdown_promptly(looper, monkeypatch):
    monkeypatch.setattr(srs, "MONITOR_INTERVAL_SECONDS", 60)
    monkeypatch.setattr(srs, "_COUNTDOWN_TICK_SECONDS", 1.0)
    first_pass = threading.Event()

    def classify(config, max_cache_age=None):
        first_pass.set()
        return "frame.png", "OPEN"

    looper.classify_latest_png = classify
    thread = threading.Thread(target=looper.monitor_loop, daemon=True)
    thread.start()
    assert first_pass.wait(5)

    looper._monitor_stop_event.set()
    thread.join(2)

    assert not thread.is_alive(), "stop waited for the full 60s interval"


def test_loop_exit_resets_the_ui_for_its_own_run(looper):
    looper._monitor_stop_event.set()

    run_loop(looper)

    assert looper.cleared == 1


def test_old_loop_exit_does_not_reset_a_newer_run(looper):
    """Stop then Start: the old loop finishing must not clear the new run."""
    old_run = threading.Event()
    old_run.set()
    looper._monitor_stop_event = threading.Event()  # the new run

    run_loop(looper, stop_event=old_run)

    assert looper.cleared == 0


def test_old_loop_cannot_be_revived_by_a_new_start(looper):
    """With a shared stop flag, Start used to un-stop a loop still winding down."""
    old_run = looper._monitor_stop_event
    old_run.set()                                   # Stop
    looper._monitor_stop_event = threading.Event()  # Start creates a fresh event
    looper.classify_latest_png = lambda *a, **k: pytest.fail("old loop ran a pass")

    run_loop(looper, stop_event=old_run)


def test_stop_monitoring_sets_only_the_current_event(looper):
    current = looper._monitor_stop_event

    looper.stop_monitoring()

    assert current.is_set()
    assert looper.cleared == 1


def test_loop_without_a_run_returns_instead_of_crashing(looper):
    """monitor_loop called before any Start has no stop event to honour."""
    looper._monitor_stop_event = None
    looper.classify_latest_png = lambda *a, **k: pytest.fail("loop ran without a run")

    looper.monitor_loop()

    assert looper.cleared == 0


def test_interval_holds_when_the_tick_is_not_one_second(looper, monkeypatch):
    """The wait is driven by a deadline, not by counting ticks as seconds."""
    monkeypatch.setattr(srs, "MONITOR_INTERVAL_SECONDS", 0.3)
    monkeypatch.setattr(srs, "_COUNTDOWN_TICK_SECONDS", 0.05)
    started = []

    def classify(config, max_cache_age=None):
        started.append(time.monotonic())
        if len(started) == 2:
            looper._monitor_stop_event.set()
        return "frame.png", "OPEN"

    looper.classify_latest_png = classify

    run_loop(looper)

    gap = started[1] - started[0]
    assert 0.3 <= gap < 1.0, f"passes {gap:.3f}s apart, expected ~0.3s"


def test_countdown_reports_whole_seconds_down_to_one(looper, monkeypatch):
    monkeypatch.setattr(srs, "MONITOR_INTERVAL_SECONDS", 2.5)
    monkeypatch.setattr(srs, "_COUNTDOWN_TICK_SECONDS", 0.5)
    shown = []
    looper.update_countdown = shown.append
    passes = []

    def classify(config, max_cache_age=None):
        passes.append(1)
        if len(passes) == 2:
            looper._monitor_stop_event.set()
        return "frame.png", "OPEN"

    looper.classify_latest_png = classify

    run_loop(looper, timeout=10)

    assert all(isinstance(r, int) for r in shown)
    assert shown[0] == 3 and shown[-1] == 1
    assert shown == sorted(shown, reverse=True)


# ── what the Monitoring tab shows for a failed pass ───────────────────────────

class FakeLabel:
    def __init__(self):
        self.text = None
        self.fg = None

    def config(self, text=None, fg=None, **_kwargs):
        self.text, self.fg = text, fg


@pytest.fixture
def display(app):
    app.status_label = FakeLabel()
    app.statusbar_label = FakeLabel()
    app._failsafe_active = False
    return app


def test_failed_pass_shows_its_reason(display):
    display.update_monitoring_status(None, "Could not read image frame.png")

    assert "Could not read image frame.png" in display.status_label.text
    assert display.status_label.fg == srs.COLOR_ERROR


def test_failed_pass_mentions_an_active_failsafe(display):
    display._failsafe_active = True

    display.update_monitoring_status(None, "Failed to fetch image from URL")

    assert "Failed to fetch image from URL" in display.status_label.text
    assert "fail-safe" in display.status_label.text


def test_failed_pass_without_a_reason_falls_back_to_a_generic_message(display):
    display.update_monitoring_status(None, None)

    assert display.status_label.text == "Monitoring: Error checking files"
