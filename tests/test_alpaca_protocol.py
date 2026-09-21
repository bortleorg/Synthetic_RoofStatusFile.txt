"""The Alpaca HTTP surface follows the spec closely enough for real clients.

Parameter names are case-insensitive, a bad ClientTransactionID is answered
rather than crashing the request, a malformed Connected value is rejected
instead of silently disconnecting, and error numbers are real ASCOM codes.
"""

import pytest

import ascom_alpaca_safety as alpaca
from ascom_alpaca_safety import AscomAlpacaSafetyMonitor

BASE = "/api/v1/safetymonitor/0"


class FakeApp:
    def get_cached_status(self, max_age_seconds=None):
        return "OPEN", 5

    def is_sun_safe_for_open(self, config=None):
        return True

    def calculate_sun_angle(self, config=None):
        return -30.0


@pytest.fixture
def client():
    server = AscomAlpacaSafetyMonitor(port=0, device_number=0, roof_classifier_app=FakeApp(),
                                      start_background=False)
    client = server.app.test_client()
    client.server = server
    return client


# ── parse_bool ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("True", True), ("true", True), ("TRUE", True), (" True ", True),
    ("False", False), ("false", False), (True, True), (False, False),
])
def test_parse_bool_accepts_alpaca_booleans(text, expected):
    assert alpaca.parse_bool(text) is expected


@pytest.mark.parametrize("text", ["maybe", "", "2", "yes please"])
def test_parse_bool_rejects_anything_else(text):
    with pytest.raises(ValueError):
        alpaca.parse_bool(text)


# ── Connected ─────────────────────────────────────────────────────────────────

def test_connected_parameter_name_is_case_insensitive(client):
    response = client.put(f"{BASE}/connected", data={"connected": "true"})

    assert response.status_code == 200
    assert client.server.connected is True


def test_garbled_connected_value_is_rejected_and_does_not_disconnect(client):
    client.put(f"{BASE}/connected", data={"Connected": "True"})

    response = client.put(f"{BASE}/connected", data={"Connected": "maybe"})

    assert response.status_code == 400
    assert client.server.connected is True


def test_missing_connected_value_is_rejected(client):
    response = client.put(f"{BASE}/connected", data={})

    assert response.status_code == 400
    assert client.server.connected is False


# ── ClientTransactionID ───────────────────────────────────────────────────────

def test_client_transaction_id_is_echoed(client):
    payload = client.get(f"{BASE}/issafe?ClientTransactionID=42").get_json()
    assert payload["ClientTransactionID"] == 42


def test_client_transaction_id_name_is_case_insensitive(client):
    payload = client.get(f"{BASE}/issafe?clienttransactionid=7").get_json()
    assert payload["ClientTransactionID"] == 7


@pytest.mark.parametrize("value", ["abc", "-1", str(2 ** 32), "1.5"])
def test_invalid_client_transaction_id_is_answered_with_zero(client, value):
    """A bad ID used to raise inside the error handler too, giving a bare 500."""
    response = client.get(f"{BASE}/issafe?ClientTransactionID={value}")

    assert response.status_code == 200
    assert response.get_json()["ClientTransactionID"] == 0


def test_missing_client_transaction_id_is_zero(client):
    assert client.get(f"{BASE}/issafe").get_json()["ClientTransactionID"] == 0


def test_client_transaction_id_is_read_from_the_form_on_put(client):
    response = client.put(f"{BASE}/connected",
                          data={"Connected": "True", "ClientTransactionID": "99"})
    assert response.get_json()["ClientTransactionID"] == 99


# ── error numbers ─────────────────────────────────────────────────────────────

def test_unsupported_action_uses_action_not_implemented(client):
    payload = client.put(f"{BASE}/action", data={"Action": "foo", "Parameters": ""}).get_json()
    assert payload["ErrorNumber"] == alpaca.ERROR_ACTION_NOT_IMPLEMENTED == 0x40C


@pytest.mark.parametrize("endpoint", ["commandblind", "commandbool", "commandstring"])
def test_unsupported_commands_use_not_implemented(client, endpoint):
    payload = client.put(f"{BASE}/{endpoint}", data={"Command": "x", "Raw": "False"}).get_json()
    assert payload["ErrorNumber"] == alpaca.ERROR_NOT_IMPLEMENTED == 0x400


def test_unknown_endpoint_uses_not_implemented(client):
    payload = client.get(f"{BASE}/nosuchproperty").get_json()
    assert payload["ErrorNumber"] == alpaca.ERROR_NOT_IMPLEMENTED


@pytest.mark.parametrize("path", [
    f"{BASE}/action", f"{BASE}/commandblind", f"{BASE}/nosuchproperty", "/whatever",
])
def test_every_error_number_is_in_the_ascom_driver_range(client, path):
    payload = client.put(path, data={}).get_json()
    assert payload["ErrorNumber"] == 0 or 0x400 <= payload["ErrorNumber"] <= 0xFFF


def test_http_errors_keep_their_status_code(client):
    """The catch-all error handler used to turn every HTTP error into a 500."""
    from flask import abort

    client.server.app.add_url_rule("/teapot", "teapot", lambda: abort(418))

    assert client.get("/teapot").status_code == 418


def test_unexpected_exceptions_are_a_500_with_an_ascom_error(client):
    def explode():
        raise RuntimeError("kaboom")

    client.server.app.add_url_rule("/explode", "explode", explode)
    response = client.get("/explode")

    assert response.status_code == 500
    assert response.get_json()["ErrorNumber"] == alpaca.ERROR_UNSPECIFIED
