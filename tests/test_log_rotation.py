"""Log files are size-capped and rotated, and handles are not leaked.

The app runs unattended for months. The ASCOM server logged every request with
its full headers at DEBUG into a file that was never rotated, and the classifier
log re-opened a new file handle on every Start Monitoring without closing the
old one.
"""

import logging
import logging.handlers

import pytest

import ascom_alpaca_safety as alpaca
import synthetic_roofstatus as srs
from ascom_alpaca_safety import AscomAlpacaSafetyMonitor


class Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


@pytest.fixture
def logging_app(app, tmp_path):
    app.log_enabled = Var(True)
    app.log_path = Var(str(tmp_path / "roof_classifier.log"))
    app.output_path = Var(str(tmp_path / "RoofStatusFile.txt"))
    yield app
    for handler in app.logger.handlers[:]:
        app.logger.removeHandler(handler)
        handler.close()


def file_handlers(logger):
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def test_classifier_log_is_rotated(logging_app):
    logging_app.setup_logging()

    handlers = file_handlers(logging_app.logger)
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.handlers.RotatingFileHandler)
    assert handlers[0].maxBytes == srs.LOG_MAX_BYTES > 0
    assert handlers[0].backupCount == srs.LOG_BACKUP_COUNT > 0


def test_repeated_setup_closes_the_previous_file_handle(logging_app):
    logging_app.setup_logging()
    first = file_handlers(logging_app.logger)[0]

    logging_app.setup_logging()

    assert first.stream is None, "old log file handle left open"
    assert len(file_handlers(logging_app.logger)) == 1


def test_classifier_log_actually_rotates(logging_app, monkeypatch):
    monkeypatch.setattr(srs, "LOG_MAX_BYTES", 200)
    logging_app.setup_logging()

    for i in range(50):
        logging_app.logger.info(f"pass {i} " + "x" * 40)

    log_path = logging_app.log_path.get()
    import os
    assert os.path.exists(log_path + ".1")
    assert os.path.getsize(log_path) <= 400


def test_ascom_log_is_rotated_and_not_at_debug():
    AscomAlpacaSafetyMonitor(port=0, device_number=0, start_background=False)
    logger = logging.getLogger("AscomAlpacaSafetyMonitor")

    assert logger.level == alpaca.LOG_LEVEL
    assert not logger.isEnabledFor(logging.DEBUG)
    for handler in file_handlers(logger):
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes > 0


def test_ascom_requests_do_not_log_headers_by_default(caplog):
    server = AscomAlpacaSafetyMonitor(port=0, device_number=0, start_background=False)
    client = server.app.test_client()

    with caplog.at_level(logging.INFO, logger="AscomAlpacaSafetyMonitor"):
        client.get("/api/v1/safetymonitor/0/issafe", headers={"X-Secret": "hunter2"})

    assert "hunter2" not in caplog.text
    assert "Headers" not in caplog.text
