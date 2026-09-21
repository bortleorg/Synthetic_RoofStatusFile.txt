"""ASCOM safety monitor: the IsSafe flag and the discovery responder."""

import socket
import threading

import pytest

from ascom_alpaca_safety import (
    MAX_STATUS_AGE_SECONDS,
    AscomAlpacaSafetyMonitor,
    compute_safety,
)


# ── compute_safety ────────────────────────────────────────────────────────────

def test_open_roof_in_the_dark_is_safe():
    assert compute_safety("OPEN", True, 10) == (True, "")


def test_closed_roof_is_not_safe():
    is_safe, error = compute_safety("CLOSED", True, 10)
    assert is_safe is False
    assert error == ""


def test_open_roof_in_daylight_is_not_safe():
    assert compute_safety("OPEN", False, 10)[0] is False


def test_unknown_roof_status_is_not_safe():
    is_safe, error = compute_safety(None, True, None)
    assert is_safe is False
    assert "No roof status" in error


def test_stale_roof_status_is_not_safe():
    """An old classification must not keep a client believing it is safe."""
    is_safe, error = compute_safety(None, True, MAX_STATUS_AGE_SECONDS + 60)
    assert is_safe is False
    assert "stale" in error


def test_status_older_than_the_limit_is_rejected_even_if_supplied():
    is_safe, error = compute_safety("OPEN", True, MAX_STATUS_AGE_SECONDS + 1)
    assert is_safe is False
    assert "stale" in error


def test_age_right_at_the_limit_is_still_accepted():
    assert compute_safety("OPEN", True, MAX_STATUS_AGE_SECONDS) == (True, "")


# ── refresh_safety_status ─────────────────────────────────────────────────────

class FakeApp:
    def __init__(self, status="OPEN", age=5, sun_safe=True):
        self.status = status
        self.age = age
        self.sun_safe = sun_safe
        self.classify_calls = 0

    def get_cached_status(self, max_age_seconds=MAX_STATUS_AGE_SECONDS):
        return self.status, self.age

    def is_sun_safe_for_open(self, config=None):
        return self.sun_safe

    def classify_latest_png(self, config=None, max_cache_age=None):
        self.classify_calls += 1
        return "f.png", self.status


@pytest.fixture
def monitor():
    """A server with no sockets, no threads and no Flask run loop."""
    server = AscomAlpacaSafetyMonitor(port=0, device_number=0, start_background=False)
    server.discovery_enabled = False
    return server


def test_refresh_reports_safe_for_a_fresh_open_roof(monitor):
    monitor.roof_classifier_app = FakeApp("OPEN", 5, True)
    monitor.connected = True

    monitor.refresh_safety_status()

    assert monitor.is_safe is True
    assert monitor.last_error == ""


def test_refresh_never_classifies_itself(monitor):
    """The safety thread used to race the monitor thread for the status file."""
    app = FakeApp("OPEN", 5, True)
    monitor.roof_classifier_app = app
    monitor.connected = True

    monitor.refresh_safety_status()

    assert app.classify_calls == 0


def test_refresh_reports_unsafe_when_the_status_is_stale(monitor):
    monitor.roof_classifier_app = FakeApp(None, MAX_STATUS_AGE_SECONDS + 30, True)
    monitor.connected = True
    monitor.is_safe = True

    monitor.refresh_safety_status()

    assert monitor.is_safe is False
    assert "stale" in monitor.last_error


def test_refresh_reports_unsafe_in_daylight(monitor):
    monitor.roof_classifier_app = FakeApp("OPEN", 5, sun_safe=False)
    monitor.connected = True

    monitor.refresh_safety_status()

    assert monitor.is_safe is False


def test_refresh_leaves_the_flag_alone_when_no_client_is_connected(monitor):
    monitor.roof_classifier_app = FakeApp("OPEN", 5, True)
    monitor.connected = False
    monitor.is_safe = False

    monitor.refresh_safety_status()

    assert monitor.is_safe is False


def test_refresh_reports_unsafe_when_the_app_raises(monitor):
    class Exploding(FakeApp):
        def get_cached_status(self, max_age_seconds=MAX_STATUS_AGE_SECONDS):
            raise RuntimeError("boom")

    monitor.roof_classifier_app = Exploding()
    monitor.connected = True
    monitor.is_safe = True

    monitor.refresh_safety_status()

    assert monitor.is_safe is False
    assert "boom" in monitor.last_error


def test_refresh_reports_unsafe_without_a_classifier(monitor):
    """A standalone or misconfigured server has no basis to report safe."""
    monitor.roof_classifier_app = None
    monitor.connected = True
    monitor.is_safe = True

    monitor.refresh_safety_status()

    assert monitor.is_safe is False
    assert monitor.last_error


# ── transaction IDs ───────────────────────────────────────────────────────────

def test_server_transaction_ids_are_unique_across_threads(monitor):
    seen = []
    seen_lock = threading.Lock()

    def bump():
        for _ in range(200):
            with monitor._transaction_lock:
                monitor._server_transaction_id += 1
                value = monitor._server_transaction_id
            with seen_lock:
                seen.append(value)

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(seen) == len(set(seen)) == 800


# ── discovery responder ───────────────────────────────────────────────────────

def test_discovery_responder_survives_a_bad_packet(monitor):
    """One failed request must not take discovery down for the whole session.

    Before this, any exception in the loop hit a bare ``break`` and NINA could
    never find the device again without restarting the app.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.settimeout(5)
    port = server_sock.getsockname()[1]

    failures = {"count": 0}

    class FlakySocket:
        """Delegates to a real socket but fails the first send."""

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def recvfrom(self, size):
            return self._wrapped.recvfrom(size)

        def sendto(self, data, addr):
            failures["count"] += 1
            if failures["count"] == 1:
                raise ValueError("serialisation blew up")
            return self._wrapped.sendto(data, addr)

        def close(self):
            self._wrapped.close()

    monitor.discovery_socket = FlakySocket(server_sock)
    monitor._stop_event.clear()

    thread = threading.Thread(target=monitor.discovery_responder, daemon=True)
    thread.start()

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(5)
    try:
        client.sendto(b"alpacadiscovery1", ("127.0.0.1", port))
        client.sendto(b"alpacadiscovery1", ("127.0.0.1", port))
        data, _addr = client.recvfrom(2048)
    finally:
        monitor._stop_event.set()
        monitor.stop_discovery_responder()
        client.close()
        thread.join(timeout=5)

    assert b"AlpacaPort" in data
    assert failures["count"] == 2, "responder gave up after the first failure"


def test_stop_signals_the_background_threads(monitor):
    assert not monitor._stop_event.is_set()
    monitor.stop()
    assert monitor._stop_event.is_set()


def test_update_loop_exits_when_stopped(monitor):
    monitor.roof_classifier_app = FakeApp("OPEN", 5, True)
    monitor.connected = True

    thread = threading.Thread(target=monitor.update_safety_status, daemon=True)
    thread.start()
    monitor.stop()
    thread.join(timeout=5)

    assert not thread.is_alive(), "update thread ignored the stop event"


# ── the flag a client reads before anything has been classified ───────────────

def test_is_safe_starts_false(monitor):
    """A client polling before the first classification must not read "safe"."""
    assert monitor.is_safe is False
    assert monitor.last_error


def test_connecting_refreshes_the_flag_immediately(monitor):
    monitor.roof_classifier_app = FakeApp("OPEN", 5, True)
    client = monitor.app.test_client()

    response = client.put(
        "/api/v1/safetymonitor/0/connected", data={"Connected": "True"})

    assert response.status_code == 200
    assert monitor.connected is True
    assert monitor.is_safe is True


def test_issafe_endpoint_reports_the_current_flag(monitor):
    monitor.roof_classifier_app = FakeApp("CLOSED", 5, True)
    client = monitor.app.test_client()

    client.put("/api/v1/safetymonitor/0/connected", data={"Connected": "True"})
    payload = client.get("/api/v1/safetymonitor/0/issafe").get_json()

    assert payload["Value"] is False
    assert payload["ErrorNumber"] == 0


def test_status_endpoint_does_not_classify(monitor):
    """A GET on /status must not rewrite the roof status file."""
    app_stub = FakeApp("OPEN", 5, True)
    monitor.roof_classifier_app = app_stub
    client = monitor.app.test_client()

    payload = client.get("/api/v1/safetymonitor/0/status").get_json()

    assert payload["Value"]["RoofStatus"] == "OPEN"
    assert app_stub.classify_calls == 0


# ── IsSafe while no client is connected ───────────────────────────────────────

def test_issafe_is_false_when_not_connected(monitor):
    """ASCOM SafetyMonitor: IsSafe must be False while the device is not connected.

    The stored flag is only refreshed while connected, so a client polling
    without connecting used to read a value that could be hours old.
    """
    monitor.is_safe = True
    monitor.last_error = ""
    monitor.connected = False
    client = monitor.app.test_client()

    payload = client.get("/api/v1/safetymonitor/0/issafe").get_json()

    assert payload["Value"] is False
    assert payload["ErrorMessage"]


def test_disconnect_drops_the_flag(monitor):
    app_stub = FakeApp("OPEN", 5, True)
    monitor.roof_classifier_app = app_stub
    client = monitor.app.test_client()

    client.put("/api/v1/safetymonitor/0/connected", data={"Connected": "True"})
    assert client.get("/api/v1/safetymonitor/0/issafe").get_json()["Value"] is True

    client.put("/api/v1/safetymonitor/0/connected", data={"Connected": "False"})

    assert monitor.is_safe is False
    assert client.get("/api/v1/safetymonitor/0/issafe").get_json()["Value"] is False


def test_reconnecting_after_the_roof_closed_reports_unsafe(monitor):
    """The value from before a disconnect must not survive into the next session."""
    app_stub = FakeApp("OPEN", 5, True)
    monitor.roof_classifier_app = app_stub
    client = monitor.app.test_client()

    client.put("/api/v1/safetymonitor/0/connected", data={"Connected": "True"})
    client.put("/api/v1/safetymonitor/0/connected", data={"Connected": "False"})
    app_stub.status = "CLOSED"
    client.put("/api/v1/safetymonitor/0/connected", data={"Connected": "True"})

    assert client.get("/api/v1/safetymonitor/0/issafe").get_json()["Value"] is False


def test_status_endpoint_reports_unsafe_when_not_connected(monitor):
    monitor.roof_classifier_app = FakeApp("OPEN", 5, True)
    monitor.is_safe = True
    monitor.connected = False
    client = monitor.app.test_client()

    payload = client.get("/api/v1/safetymonitor/0/status").get_json()

    assert payload["Value"]["IsSafe"] is False
