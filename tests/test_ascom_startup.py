"""An ASCOM server that fails to bind must not be shown as running.

The port is bound on the server thread after start_ascom_server returns, and
Werkzeug reports a port in use with sys.exit(1). That escaped unnoticed, so the
UI kept reporting a running server - port fields locked, "waiting for client" -
with nothing listening.
"""

import socket

import pytest

import synthetic_roofstatus as srs
from ascom_alpaca_safety import AscomAlpacaSafetyMonitor


class Var:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


@pytest.fixture
def occupied_port():
    sock = socket.socket()
    sock.bind(("0.0.0.0", 0))
    sock.listen()
    yield sock.getsockname()[1]
    sock.close()


@pytest.fixture
def ascom_app(app, monkeypatch):
    app.ascom_enabled = Var(True)
    app._defer_to_ui = lambda func: func()
    app.errors = []
    monkeypatch.setattr(srs.messagebox, "showerror",
                        lambda title, msg, **k: app.errors.append(msg))
    return app


def failing_server(port):
    return AscomAlpacaSafetyMonitor(port=port, device_number=0, start_background=False)


def test_a_port_in_use_clears_the_running_state(ascom_app, occupied_port):
    server = failing_server(occupied_port)
    ascom_app.ascom_server = server

    ascom_app._serve_ascom(server, silent=False)

    assert ascom_app.ascom_server is None
    assert ascom_app.ascom_enabled.get() is False
    assert len(ascom_app.errors) == 1
    assert str(occupied_port) in ascom_app.errors[0]


def test_auto_start_failure_is_not_shown_in_a_dialog(ascom_app, occupied_port):
    server = failing_server(occupied_port)
    ascom_app.ascom_server = server

    ascom_app._serve_ascom(server, silent=True)

    assert ascom_app.ascom_server is None
    assert ascom_app.errors == []


def test_a_late_failure_does_not_clear_a_newer_server(ascom_app, occupied_port):
    old = failing_server(occupied_port)
    newer = object()
    ascom_app.ascom_server = newer

    ascom_app._serve_ascom(old, silent=False)

    assert ascom_app.ascom_server is newer
    assert ascom_app.ascom_enabled.get() is True
    assert ascom_app.errors == []
