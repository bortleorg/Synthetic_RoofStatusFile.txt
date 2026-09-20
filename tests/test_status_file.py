"""The roof status file is the app's contract with ASCOM clients and SkyRoof."""

import os

import roof_io


def test_write_status_file_uses_the_sro_format(app, tmp_path):
    target = tmp_path / "RoofStatusFile.txt"

    assert app._write_status_file("OPEN", output_path=str(target),
                                  timestamp="2026-09-20 09:30:00PM")

    assert target.read_text(encoding="utf-8") == (
        "???2026-09-20 09:30:00PM Roof Status: OPEN\n"
    )


def test_write_status_file_includes_the_override_reason(app, tmp_path):
    target = tmp_path / "RoofStatusFile.txt"

    app._write_status_file("CLOSED", " (Manual override: CLOSED)",
                           str(target), "2026-09-20 09:30:00PM")

    assert target.read_text(encoding="utf-8").endswith(
        "Roof Status: CLOSED (Manual override: CLOSED)\n"
    )


def test_write_status_file_leaves_no_temp_files(app, tmp_path):
    target = tmp_path / "RoofStatusFile.txt"

    app._write_status_file("OPEN", output_path=str(target), timestamp="t1")
    app._write_status_file("CLOSED", output_path=str(target), timestamp="t2")

    assert [p.name for p in tmp_path.iterdir()] == ["RoofStatusFile.txt"]


def test_write_status_file_reports_failure_instead_of_raising(app, tmp_path):
    """A bad output path must not take the monitor thread down."""
    target = tmp_path / "a-directory"
    target.mkdir()

    assert app._write_status_file("OPEN", output_path=str(target)) is False


def test_written_status_survives_a_failed_rewrite(app, tmp_path, monkeypatch):
    target = tmp_path / "RoofStatusFile.txt"
    app._write_status_file("OPEN", output_path=str(target), timestamp="good")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(roof_io.os, "replace", boom)
    assert app._write_status_file("CLOSED", output_path=str(target)) is False

    assert "Roof Status: OPEN" in target.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["RoofStatusFile.txt"]


def test_status_file_is_readable_as_a_secondary_source(app, tmp_path):
    """Two instances chained together must agree on what was written."""
    target = tmp_path / "RoofStatusFile.txt"
    app._write_status_file("CLOSED", " (Sun too high - safety override)",
                           str(target), "2026-09-20 02:00:00PM")

    status, mod_time = app._read_secondary_values(True, str(target))

    assert status == "CLOSED"
    assert mod_time is not None


def test_secondary_source_missing_file_is_not_an_error(app, tmp_path):
    assert app._read_secondary_values(True, str(tmp_path / "gone.txt")) == (None, None)


def test_secondary_source_disabled_returns_nothing(app, tmp_path):
    target = tmp_path / "RoofStatusFile.txt"
    target.write_text("Roof Status: OPEN\n", encoding="utf-8")

    assert app._read_secondary_values(False, str(target)) == (None, None)


def test_secondary_source_empty_file_returns_nothing(app, tmp_path):
    target = tmp_path / "RoofStatusFile.txt"
    target.write_text("   \n\n", encoding="utf-8")

    assert app._read_secondary_values(True, str(target)) == (None, None)


def test_secondary_source_ambiguous_line_is_not_read_as_open(app, tmp_path):
    target = tmp_path / "RoofStatusFile.txt"
    target.write_text("Roof is not open\n", encoding="utf-8")

    status, _mod_time = app._read_secondary_values(True, str(target))

    assert status is None
