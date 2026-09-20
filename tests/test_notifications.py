"""Notification dispatch must run entirely off the configuration snapshot.

_check_and_send_notifications is called from the monitor thread. Tk variables may
only be read on the UI thread, so the ``app`` fixture deliberately has none: any
attempt to reach for ``self.notif_*`` raises AttributeError and fails the test.
"""

from datetime import datetime, timedelta

import pytest


BASE_CONFIG = {
    'notif_stale_enabled': False,
    'notif_stale_minutes': '10',
    'notif_stale_url': '',
    'notif_open_enabled': False,
    'notif_open_url': '',
    'notif_closed_enabled': False,
    'notif_closed_url': '',
    'notif_heartbeat_enabled': False,
    'notif_heartbeat_minutes': '5',
    'notif_heartbeat_url': '',
}


@pytest.fixture
def notifier(app):
    """App fixture wired up for notification checks, with webhooks captured."""
    app.previous_status = None
    app.last_image_hash = None
    app.last_new_hash_time = None
    app._last_stale_notification_time = None
    app._last_heartbeat_time = None
    app.sent = []
    app._send_webhook = lambda url, payload: app.sent.append((url, payload))
    return app


def config(**overrides):
    merged = dict(BASE_CONFIG)
    merged.update(overrides)
    return merged


def test_no_tk_access_when_everything_is_disabled(notifier):
    notifier._check_and_send_notifications("OPEN", config())
    assert notifier.sent == []
    assert notifier.previous_status == "OPEN"


def test_open_transition_fires_once(notifier):
    cfg = config(notif_open_enabled=True, notif_open_url="http://hook/open")

    notifier._check_and_send_notifications("OPEN", cfg)
    notifier._check_and_send_notifications("OPEN", cfg)

    assert len(notifier.sent) == 1
    url, payload = notifier.sent[0]
    assert url == "http://hook/open"
    assert payload["event"] == "roof_open"
    assert payload["status"] == "OPEN"


def test_closed_transition_fires_on_change_back(notifier):
    cfg = config(
        notif_open_enabled=True, notif_open_url="http://hook/open",
        notif_closed_enabled=True, notif_closed_url="http://hook/closed",
    )

    notifier._check_and_send_notifications("OPEN", cfg)
    notifier._check_and_send_notifications("CLOSED", cfg)
    notifier._check_and_send_notifications("OPEN", cfg)

    assert [p["event"] for _u, p in notifier.sent] == [
        "roof_open", "roof_closed", "roof_open",
    ]


def test_transition_webhook_is_skipped_without_a_url(notifier):
    notifier._check_and_send_notifications("OPEN", config(notif_open_enabled=True))
    assert notifier.sent == []


def test_stale_notification_fires_when_the_image_stops_changing(notifier):
    notifier.last_new_hash_time = datetime.utcnow() - timedelta(minutes=30)
    cfg = config(
        notif_stale_enabled=True,
        notif_stale_minutes="10",
        notif_stale_url="http://hook/stale",
    )

    notifier._check_and_send_notifications("CLOSED", cfg)

    assert len(notifier.sent) == 1
    _url, payload = notifier.sent[0]
    assert payload["event"] == "image_stale"
    assert payload["stale_minutes"] >= 10


def test_stale_notification_is_rate_limited(notifier):
    notifier.last_new_hash_time = datetime.utcnow() - timedelta(minutes=30)
    cfg = config(
        notif_stale_enabled=True,
        notif_stale_minutes="10",
        notif_stale_url="http://hook/stale",
    )

    notifier._check_and_send_notifications("CLOSED", cfg)
    notifier._check_and_send_notifications("CLOSED", cfg)

    assert len(notifier.sent) == 1


def test_fresh_image_is_not_stale(notifier):
    notifier.last_new_hash_time = datetime.utcnow()
    cfg = config(
        notif_stale_enabled=True,
        notif_stale_minutes="10",
        notif_stale_url="http://hook/stale",
    )

    notifier._check_and_send_notifications("CLOSED", cfg)

    assert notifier.sent == []


def test_unparseable_stale_minutes_falls_back_to_the_default(notifier):
    notifier.last_new_hash_time = datetime.utcnow() - timedelta(minutes=12)
    cfg = config(
        notif_stale_enabled=True,
        notif_stale_minutes="not a number",
        notif_stale_url="http://hook/stale",
    )

    notifier._check_and_send_notifications("CLOSED", cfg)

    assert [p["event"] for _u, p in notifier.sent] == ["image_stale"]


def test_heartbeat_fires_then_waits_for_its_interval(notifier):
    notifier.last_new_hash_time = datetime.utcnow()
    cfg = config(
        notif_heartbeat_enabled=True,
        notif_heartbeat_minutes="5",
        notif_heartbeat_url="http://hook/beat",
    )

    notifier._check_and_send_notifications("OPEN", cfg)
    notifier._check_and_send_notifications("OPEN", cfg)

    assert [p["event"] for _u, p in notifier.sent] == ["heartbeat"]

    notifier._last_heartbeat_time = datetime.utcnow() - timedelta(minutes=6)
    notifier._check_and_send_notifications("OPEN", cfg)

    assert [p["event"] for _u, p in notifier.sent] == ["heartbeat", "heartbeat"]


def test_heartbeat_is_suppressed_while_the_image_is_stale(notifier):
    notifier.last_new_hash_time = datetime.utcnow() - timedelta(minutes=30)
    cfg = config(
        notif_heartbeat_enabled=True,
        notif_heartbeat_minutes="5",
        notif_heartbeat_url="http://hook/beat",
    )

    notifier._check_and_send_notifications("OPEN", cfg)

    assert notifier.sent == []


def test_missing_config_is_reported_not_crashed(notifier):
    """With no snapshot available the call is skipped rather than reading Tk."""
    notifier._get_monitor_config = lambda: None

    notifier._check_and_send_notifications("OPEN", None)

    assert notifier.sent == []
