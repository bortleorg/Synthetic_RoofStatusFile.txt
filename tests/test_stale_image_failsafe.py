"""A frozen camera feed must not keep the roof reported OPEN.

A hung camera, a URL serving a cached frame, or capture software that stopped
writing all hand the classifier the same frame every pass. That classifies
cleanly, so none of the failure paths fired and an OPEN frame was reported all
night. With the fail-safe setting on, once the image has not changed for the
stale threshold the status is reported CLOSED. By default the model's view of
the frozen frame is still reported.
"""

import threading
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

import synthetic_roofstatus as srs


class OpenModel:
    def predict(self, _x):
        return [1]


@pytest.fixture
def pipeline(app, tmp_path):
    cv2.imwrite(str(tmp_path / "frame.png"), np.full((48, 64), 128, dtype=np.uint8))

    app._classify_lock = threading.RLock()
    app._last_classification = None
    app._last_good_pass_at = None
    app._failsafe_active = False
    app.model = OpenModel()
    app.last_image_hash = None
    app.last_new_hash_time = None
    app.previous_classified_status = None
    app._in_disagreement = False
    app.save_sample_if_needed = lambda path, config=None: None
    app._capture_preview = lambda *a, **k: None
    app.read_secondary_source = lambda config=None: (None, None)
    app.is_sun_safe_for_open = lambda config=None: True
    app.calculate_sun_angle = lambda config=None: -30.0
    app.writes = []
    app._write_status_file = lambda status, reason, *a, **k: app.writes.append((status, reason)) or True
    app.config = {
        'camera_url': '', 'monitor_path': str(tmp_path),
        'output_path': str(tmp_path / "status.txt"),
        'save_on_disagreement': False, 'save_on_toggle': False,
        'notif_stale_minutes': '10',
        # The fail-safe is opt-in; most tests here exercise it.
        'stale_image_action': srs.STALE_ACTION_CLOSED,
    }
    return app


def classify(app):
    return app.classify_latest_png(app.config, max_cache_age=0)


def freeze_for(app, minutes):
    """Pretend the current frame has been unchanged for *minutes*."""
    app.last_new_hash_time = datetime.utcnow() - timedelta(minutes=minutes)


def test_fresh_open_frame_is_reported_open(pipeline):
    assert classify(pipeline) == ("frame.png", "OPEN")
    assert pipeline.writes[-1] == ("OPEN", "")


def test_frozen_open_frame_is_reported_closed(pipeline):
    classify(pipeline)
    freeze_for(pipeline, 11)

    _filename, status = classify(pipeline)

    assert status == "CLOSED"
    written_status, reason = pipeline.writes[-1]
    assert written_status == "CLOSED"
    assert "Image unchanged" in reason and "failsafe" in reason


def test_frozen_frame_reaches_ascom_as_closed(pipeline):
    classify(pipeline)
    freeze_for(pipeline, 30)
    classify(pipeline)

    status, _age = pipeline.get_cached_status()
    assert status == "CLOSED"


def test_frame_unchanged_for_less_than_the_threshold_is_trusted(pipeline):
    classify(pipeline)
    freeze_for(pipeline, 9)

    assert classify(pipeline)[1] == "OPEN"


def test_threshold_comes_from_the_configuration(pipeline):
    pipeline.config['notif_stale_minutes'] = '3'
    classify(pipeline)
    freeze_for(pipeline, 4)

    assert classify(pipeline)[1] == "CLOSED"


def test_invalid_threshold_falls_back_to_the_default(pipeline):
    pipeline.config['notif_stale_minutes'] = 'soon'
    classify(pipeline)
    freeze_for(pipeline, srs.DEFAULT_STALE_MINUTES - 1)
    assert classify(pipeline)[1] == "OPEN"

    freeze_for(pipeline, srs.DEFAULT_STALE_MINUTES + 1)
    assert classify(pipeline)[1] == "CLOSED"


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf"])
def test_nonsense_thresholds_use_the_default(value):
    assert srs.RoofClassifierApp._stale_threshold_minutes(
        {'notif_stale_minutes': value}) == srs.DEFAULT_STALE_MINUTES


def test_new_frame_lifts_the_failsafe(pipeline, tmp_path):
    classify(pipeline)
    freeze_for(pipeline, 20)
    assert classify(pipeline)[1] == "CLOSED"

    cv2.imwrite(str(tmp_path / "frame.png"), np.full((48, 64), 90, dtype=np.uint8))

    assert classify(pipeline)[1] == "OPEN"


def test_manual_override_still_wins_over_a_frozen_frame(pipeline):
    classify(pipeline)
    freeze_for(pipeline, 20)
    pipeline.override_active = "OPEN"

    assert classify(pipeline)[1] == "OPEN"
    assert pipeline.writes[-1] == ("OPEN", " (Manual override: OPEN)")


def test_frozen_frame_does_not_count_as_a_model_toggle(pipeline):
    """Recovery from a frozen feed is not a model transition worth capturing."""
    pipeline.config['save_on_toggle'] = True
    saved = []
    pipeline._save_frame_for_review = lambda path, reason, config=None: saved.append(reason)

    classify(pipeline)
    freeze_for(pipeline, 20)
    classify(pipeline)

    assert saved == []
    assert pipeline.previous_classified_status == "OPEN"


# ── the "while stale, report ..." setting ─────────────────────────────────────

def test_keep_setting_reports_the_frozen_frame_as_classified(pipeline):
    pipeline.config['stale_image_action'] = srs.STALE_ACTION_KEEP
    classify(pipeline)
    freeze_for(pipeline, 60)

    assert classify(pipeline)[1] == "OPEN"
    assert pipeline.writes[-1] == ("OPEN", "")


def test_closed_setting_is_the_failsafe(pipeline):
    pipeline.config['stale_image_action'] = srs.STALE_ACTION_CLOSED
    classify(pipeline)
    freeze_for(pipeline, 60)

    assert classify(pipeline)[1] == "CLOSED"


def test_default_reports_the_frozen_frame_as_classified(pipeline):
    """The fail-safe is opt-in: with no setting the model's view is reported."""
    del pipeline.config['stale_image_action']
    classify(pipeline)
    freeze_for(pipeline, 60)

    assert classify(pipeline)[1] == "OPEN"


def test_default_action_is_keep():
    assert srs.DEFAULT_STALE_ACTION == srs.STALE_ACTION_KEEP


@pytest.mark.parametrize("config", [{}, {'stale_image_action': ''}, {'stale_image_action': 'bogus'}])
def test_missing_or_unknown_setting_uses_the_default(config):
    assert srs.RoofClassifierApp._stale_failsafe_enabled(config) is False


def test_keep_setting_does_not_silence_the_stale_notification(app):
    """The option only changes the reported status; the webhook still fires."""
    sent = []
    app._send_webhook = lambda url, payload: sent.append(payload["event"]) or (True, "HTTP 200")
    app.previous_status = "OPEN"
    app._last_stale_notification_time = None
    app._last_heartbeat_time = None
    app.last_new_hash_time = datetime.utcnow() - timedelta(minutes=30)

    app._check_and_send_notifications("OPEN", {
        'stale_image_action': srs.STALE_ACTION_KEEP,
        'notif_stale_enabled': True, 'notif_stale_url': 'http://hook',
        'notif_stale_minutes': '10',
    })

    assert sent == ["image_stale"]
