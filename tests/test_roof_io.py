"""Tests for the atomic file writers and the roof status parser."""

import json
import os
import threading
import time

import pytest

import roof_io


# ── atomic_write_text ─────────────────────────────────────────────────────────

def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "RoofStatusFile.txt"
    roof_io.atomic_write_text(str(target), "hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_atomic_write_replaces_existing_content(tmp_path):
    target = tmp_path / "RoofStatusFile.txt"
    roof_io.atomic_write_text(str(target), "a much longer first line\n")
    roof_io.atomic_write_text(str(target), "short\n")
    assert target.read_text(encoding="utf-8") == "short\n"


def test_atomic_write_leaves_no_temp_files_behind(tmp_path):
    target = tmp_path / "status.txt"
    roof_io.atomic_write_text(str(target), "one\n")
    roof_io.atomic_write_text(str(target), "two\n")
    assert [p.name for p in tmp_path.iterdir()] == ["status.txt"]


def test_atomic_write_creates_missing_directory(tmp_path):
    target = tmp_path / "nested" / "dir" / "status.txt"
    roof_io.atomic_write_text(str(target), "ok\n")
    assert target.read_text(encoding="utf-8") == "ok\n"


def test_failed_write_leaves_previous_content_intact(tmp_path, monkeypatch):
    """The whole point of the temp-file dance: a failure must not truncate."""
    target = tmp_path / "status.txt"
    roof_io.atomic_write_text(str(target), "GOOD\n")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(roof_io.os, "replace", boom)
    with pytest.raises(OSError):
        roof_io.atomic_write_text(str(target), "BAD\n")

    assert target.read_text(encoding="utf-8") == "GOOD\n"
    assert [p.name for p in tmp_path.iterdir()] == ["status.txt"]


def test_destination_is_never_truncated_during_a_write(tmp_path, monkeypatch):
    """The destination holds the complete old line right up to the rename.

    This is what a plain truncate-then-write cannot promise, and what an ASCOM
    client polling the roof status file depends on.
    """
    target = tmp_path / "status.txt"
    old_line = roof_io.format_status_line("CLOSED", timestamp="2026-09-20 01:00:00AM")
    new_line = roof_io.format_status_line("OPEN", timestamp="2026-09-20 09:00:00PM")
    roof_io.atomic_write_text(str(target), old_line)

    real_replace = os.replace
    seen = {}

    def spy(src, dst):
        seen["dest_before"] = target.read_text(encoding="utf-8")
        seen["temp_content"] = open(src, encoding="utf-8").read()
        return real_replace(src, dst)

    monkeypatch.setattr(roof_io.os, "replace", spy)
    roof_io.atomic_write_text(str(target), new_line)

    assert seen["dest_before"] == old_line, "destination was modified before the rename"
    assert seen["temp_content"] == new_line, "new content was not staged in the temp file"
    assert target.read_text(encoding="utf-8") == new_line


def test_concurrent_reader_never_sees_a_partial_line(tmp_path):
    """Every read that succeeds returns a whole status line.

    A reader can still collide with the rename itself and get a transient open
    error, which it retries; what it must never get is a truncated or empty file
    that parses as a different roof state.
    """
    target = tmp_path / "status.txt"
    roof_io.atomic_write_text(str(target), roof_io.format_status_line("CLOSED"))

    observed = []
    failures = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                observed.append(target.read_text(encoding="utf-8"))
            except OSError as exc:
                failures.append(exc)
            time.sleep(0.001)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for i in range(100):
            status = "OPEN" if i % 2 else "CLOSED"
            roof_io.atomic_write_text(
                str(target), roof_io.format_status_line(status, timestamp="t" * i)
            )
    finally:
        stop.set()
        thread.join(timeout=5)

    assert observed, "reader thread never managed a read"
    for content in observed:
        assert content.startswith("???"), content
        assert content.endswith("\n"), content
        assert roof_io.parse_roof_status(content)[0] in ("OPEN", "CLOSED"), content


def test_write_falls_back_in_place_when_the_rename_cannot_win(tmp_path, monkeypatch):
    """A destination locked against rename still gets the new status."""
    target = tmp_path / "status.txt"
    roof_io.atomic_write_text(str(target), "OLD\n")

    monkeypatch.setattr(roof_io, "_REPLACE_RETRIES", 2)
    monkeypatch.setattr(roof_io, "_REPLACE_RETRY_DELAY", 0)
    monkeypatch.setattr(
        roof_io.os, "replace",
        lambda src, dst: (_ for _ in ()).throw(PermissionError("locked")),
    )

    roof_io.atomic_write_text(str(target), "NEW\n", allow_in_place_fallback=True)

    assert target.read_text(encoding="utf-8") == "NEW\n"
    assert [p.name for p in tmp_path.iterdir()] == ["status.txt"]


def test_write_without_fallback_leaves_the_destination_intact(tmp_path, monkeypatch):
    """Settings saves must fail rather than risk truncating the file in place."""
    target = tmp_path / "settings.json"
    roof_io.write_json_atomic(str(target), {"ascom_unique_id": "abc"})

    monkeypatch.setattr(roof_io, "_REPLACE_RETRIES", 2)
    monkeypatch.setattr(roof_io, "_REPLACE_RETRY_DELAY", 0)
    monkeypatch.setattr(
        roof_io.os, "replace",
        lambda src, dst: (_ for _ in ()).throw(PermissionError("locked")),
    )

    with pytest.raises(PermissionError):
        roof_io.write_json_atomic(str(target), {"ascom_unique_id": "xyz"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ascom_unique_id": "abc"}
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]


# ── JSON helpers ──────────────────────────────────────────────────────────────

def test_write_json_atomic_roundtrip(tmp_path):
    target = tmp_path / "settings.json"
    roof_io.write_json_atomic(str(target), {"ascom_unique_id": "abc", "n": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "ascom_unique_id": "abc",
        "n": 1,
    }


def test_load_json_missing_file_returns_none(tmp_path):
    assert roof_io.load_json_with_recovery(str(tmp_path / "nope.json")) is None


def test_load_json_corrupt_file_is_quarantined(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text('{"model_path": "C:/m.joblib", "trunc', encoding="utf-8")

    seen = []
    result = roof_io.load_json_with_recovery(
        str(target), on_corrupt=lambda backup, exc: seen.append((backup, exc))
    )

    assert result is None
    assert not target.exists(), "corrupt file should be moved aside"
    assert os.path.exists(str(target) + ".corrupt")
    assert len(seen) == 1 and seen[0][0] == str(target) + ".corrupt"


def test_load_json_good_file_is_not_touched(tmp_path):
    target = tmp_path / "settings.json"
    roof_io.write_json_atomic(str(target), {"a": 1})
    assert roof_io.load_json_with_recovery(str(target)) == {"a": 1}
    assert target.exists()
    assert not os.path.exists(str(target) + ".corrupt")


# ── format_status_line ────────────────────────────────────────────────────────

def test_status_line_matches_sro_format():
    line = roof_io.format_status_line("OPEN", timestamp="2026-09-20 09:30:00PM")
    assert line == "???2026-09-20 09:30:00PM Roof Status: OPEN\n"


def test_status_line_includes_reason():
    line = roof_io.format_status_line(
        "CLOSED", " (Manual override: CLOSED)", "2026-09-20 09:30:00PM"
    )
    assert line.endswith("Roof Status: CLOSED (Manual override: CLOSED)\n")


def test_status_line_round_trips_through_the_parser():
    """Our own output must be readable as a secondary source by another instance."""
    for status in ("OPEN", "CLOSED"):
        line = roof_io.format_status_line(status, timestamp="2026-09-20 09:30:00PM")
        assert roof_io.parse_roof_status(line)[0] == status


# ── parse_roof_status ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("OPEN", "OPEN"),
    ("CLOSED", "CLOSED"),
    ("open", "OPEN"),
    ("Roof Status: OPEN", "OPEN"),
    ("???2026-09-20 09:30:00PM Roof Status: CLOSED", "CLOSED"),
    ("Roof is shut", "CLOSED"),
    ("", None),
    ("   \n  \n", None),
    ("no status here", None),
])
def test_parse_basic_cases(text, expected):
    assert roof_io.parse_roof_status(text)[0] == expected


def test_parse_uses_the_last_non_empty_line():
    text = "Roof Status: OPEN\nRoof Status: CLOSED\n\n\n"
    status, last_line = roof_io.parse_roof_status(text)
    assert status == "CLOSED"
    assert last_line == "Roof Status: CLOSED"


@pytest.mark.parametrize("line", [
    "Roof is not open",
    "roof NOT OPEN",
    "The roof is not closed",
])
def test_negated_status_is_not_guessed(line):
    """"not open" must not be read as OPEN - guessing here is the unsafe direction."""
    assert roof_io.parse_roof_status(line)[0] is None


def test_ambiguous_line_resolves_to_closed():
    """When a line mentions both states, never report the roof as open."""
    assert roof_io.parse_roof_status("OPEN CLOSED")[0] == "CLOSED"
    assert roof_io.parse_roof_status("closed open")[0] == "CLOSED"


@pytest.mark.parametrize("line", [
    "Roof is opening",
    "Roof is reopened",
    "OPENED_AT_DAWN",
])
def test_substrings_of_other_words_do_not_count(line):
    """Whole-word matching only: 'opening' is a roof in motion, not an open roof."""
    assert roof_io.parse_roof_status(line)[0] is None


def test_parse_returns_the_line_it_examined():
    _status, last_line = roof_io.parse_roof_status("junk\nnothing useful here")
    assert last_line == "nothing useful here"
