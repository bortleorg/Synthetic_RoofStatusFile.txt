"""Webhook delivery reports whether it actually worked.

_send_webhook swallows every error so a dead endpoint cannot disturb the
monitor loop. The Test button used to rely on it raising, so it announced
"Webhook Sent" for a typo'd host or an HTTP 500 alike.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import synthetic_roofstatus as srs


class Handler(BaseHTTPRequestHandler):
    received = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        Handler.received.append((self.path, json.loads(body)))
        code = int(self.path.strip("/") or 200)
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    Handler.received = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_2xx_is_success(app, server):
    ok, detail = app._send_webhook(f"{server}/200", {"event": "heartbeat"})

    assert ok is True
    assert detail == "HTTP 200"
    assert Handler.received == [("/200", {"event": "heartbeat"})]


def test_204_is_success(app, server):
    assert app._send_webhook(f"{server}/204", {})[0] is True


def test_server_error_is_failure(app, server):
    ok, detail = app._send_webhook(f"{server}/500", {"event": "roof_open"})

    assert ok is False
    assert "500" in detail


def test_client_error_is_failure(app, server):
    assert app._send_webhook(f"{server}/404", {})[0] is False


def test_unreachable_host_is_failure_not_exception(app):
    ok, detail = app._send_webhook(f"http://127.0.0.1:{unused_port()}/", {})

    assert ok is False
    assert detail


def test_invalid_url_is_failure_not_exception(app):
    assert app._send_webhook("not a url", {})[0] is False


# ── the Test button ──────────────────────────────────────────────────────────

class FakeRoot:
    def after(self, _ms, func=None, *args):
        func(*args)


class FakeMessagebox:
    """Records dialogs; the Test button shows one from its worker thread."""

    def __init__(self):
        self.shown = []
        self.done = threading.Event()

    def _show(self, kind, title, message):
        self.shown.append((kind, title, message))
        self.done.set()

    def showinfo(self, title, message):
        self._show("info", title, message)

    def showerror(self, title, message):
        self._show("error", title, message)

    def showwarning(self, title, message):
        self._show("warning", title, message)

    def wait(self):
        assert self.done.wait(15), "no dialog shown"
        return [kind for kind, _title, _message in self.shown]


class Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


@pytest.fixture
def ui(app, monkeypatch):
    box = FakeMessagebox()
    monkeypatch.setattr(srs, "messagebox", box)
    app.root = FakeRoot()
    return box


def test_test_button_reports_success(app, server, ui):
    app._test_webhook(Var(f"{server}/200"))

    assert ui.wait() == ["info"]
    assert "HTTP 200" in ui.shown[0][2]
    assert Handler.received[0][1]["event"] == "test"


def test_test_button_reports_an_http_error(app, server, ui):
    app._test_webhook(Var(f"{server}/500"))

    assert ui.wait() == ["error"]
    assert "500" in ui.shown[0][2]


def test_test_button_reports_an_unreachable_host(app, ui):
    app._test_webhook(Var(f"http://127.0.0.1:{unused_port()}/hook"))

    assert ui.wait() == ["error"]
