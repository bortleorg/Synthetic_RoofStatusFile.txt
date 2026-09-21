"""Settings must survive a crash mid-save.

A half-written settings file used to be swallowed by a bare ``except``, which
silently reset every setting - including the persisted ASCOM UniqueID that
clients key on to reconnect, and any manual override in force.
"""

import json
import os

import pytest

import roof_io
import synthetic_roofstatus as srs


class FakeVar:
    """Stand-in for a Tk variable: holds a value, no GUI, no thread affinity."""

    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


# Every Tk variable save_settings/load_settings touch, with its default.
VARS = {
    'model_path': '', 'monitor_path': '', 'output_path': 'RoofStatusFile.txt',
    'log_enabled': False, 'log_path': 'roof_classifier.log',
    'latitude': '40.0', 'longitude': '-74.0', 'sun_angle_threshold': '-17.0',
    'secondary_source_enabled': False, 'secondary_source_path': '',
    'twilight_preset_var': 'Custom',
    'ascom_enabled': False, 'ascom_port': '11111', 'ascom_device_number': '0',
    'auto_start_monitoring': False, 'auto_start_ascom': False,
    'save_on_toggle_enabled': False, 'save_on_disagreement_enabled': False,
    'training_data_folder': '', 'sample_mode_enabled': False, 'sample_rate': '0.1',
    'validation_set_path': '', 'camera_url': '', 'preview_enabled': True,
    'override_duration': '1 hour', 'override_mode': 'AUTO',
    'notif_stale_enabled': False, 'notif_stale_minutes': '10', 'notif_stale_url': '',
    'stale_image_action': 'closed',
    'notif_open_enabled': False, 'notif_open_url': '',
    'notif_closed_enabled': False, 'notif_closed_url': '',
    'notif_heartbeat_enabled': False, 'notif_heartbeat_minutes': '5',
    'notif_heartbeat_url': '',
}


@pytest.fixture
def settings_app(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(srs, "SETTINGS_FILE", str(tmp_path / "roof_classifier_settings.json"))
    for name, default in VARS.items():
        setattr(app, name, FakeVar(default))
    app.ascom_unique_id = ""
    app.override_active = None
    app.override_expiry = None
    app._preview_on = True
    return app


def test_settings_round_trip(settings_app):
    settings_app.ascom_unique_id = "stable-id-1234"
    settings_app.latitude.set("51.5")
    settings_app.camera_url.set("http://cam/latest.jpg")

    settings_app.save_settings()

    settings_app.ascom_unique_id = ""
    settings_app.latitude.set("0.0")
    settings_app.camera_url.set("")
    settings_app.load_settings()

    assert settings_app.ascom_unique_id == "stable-id-1234"
    assert settings_app.latitude.get() == "51.5"
    assert settings_app.camera_url.get() == "http://cam/latest.jpg"


def test_save_is_atomic_and_leaves_no_temp_files(settings_app, tmp_path):
    settings_app.save_settings()
    settings_app.save_settings()

    assert [p.name for p in tmp_path.iterdir()] == ["roof_classifier_settings.json"]


def test_a_failed_save_does_not_destroy_the_previous_settings(
        settings_app, monkeypatch):
    settings_app.ascom_unique_id = "stable-id-1234"
    settings_app.save_settings()

    monkeypatch.setattr(
        roof_io.os, "replace",
        lambda src, dst: (_ for _ in ()).throw(OSError("disk full")),
    )
    settings_app.latitude.set("99.9")
    settings_app.save_settings()  # must not raise

    with open(srs.SETTINGS_FILE, encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert on_disk["ascom_unique_id"] == "stable-id-1234"
    assert on_disk["latitude"] == "40.0"


def test_corrupt_settings_file_is_quarantined_not_silently_ignored(settings_app):
    with open(srs.SETTINGS_FILE, "w", encoding="utf-8") as handle:
        handle.write('{"latitude": "51.5", "ascom_unique')

    settings_app.load_settings()

    assert settings_app.latitude.get() == "40.0", "defaults should apply"
    assert not os.path.exists(srs.SETTINGS_FILE)
    assert os.path.exists(srs.SETTINGS_FILE + ".corrupt")


def test_settings_are_rewritable_after_a_corrupt_file(settings_app):
    with open(srs.SETTINGS_FILE, "w", encoding="utf-8") as handle:
        handle.write("not json at all")

    settings_app.load_settings()
    settings_app.ascom_unique_id = "new-id"
    settings_app.save_settings()
    settings_app.ascom_unique_id = ""
    settings_app.load_settings()

    assert settings_app.ascom_unique_id == "new-id"


def test_missing_settings_file_leaves_defaults_alone(settings_app):
    settings_app.load_settings()
    assert settings_app.latitude.get() == "40.0"
    assert settings_app.output_path.get() == "RoofStatusFile.txt"


def test_stale_image_action_round_trips(settings_app):
    settings_app.stale_image_action.set(srs.STALE_ACTION_KEEP)
    settings_app.save_settings()

    settings_app.stale_image_action.set(srs.STALE_ACTION_CLOSED)
    settings_app.load_settings()

    assert settings_app.stale_image_action.get() == srs.STALE_ACTION_KEEP


def test_stale_image_action_defaults_to_failsafe_for_old_settings_files(settings_app):
    """Settings saved before the option existed get the fail-safe behaviour."""
    with open(srs.SETTINGS_FILE, "w", encoding="utf-8") as handle:
        json.dump({"latitude": "51.5"}, handle)
    settings_app.stale_image_action.set(srs.STALE_ACTION_KEEP)

    settings_app.load_settings()

    assert settings_app.stale_image_action.get() == srs.STALE_ACTION_CLOSED


def test_unrecognised_stale_image_action_falls_back_to_failsafe(settings_app):
    with open(srs.SETTINGS_FILE, "w", encoding="utf-8") as handle:
        json.dump({"stale_image_action": "whatever"}, handle)

    settings_app.load_settings()

    assert settings_app.stale_image_action.get() == srs.STALE_ACTION_CLOSED
