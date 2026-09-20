"""Filesystem and parsing helpers for the roof status pipeline.

This module deliberately depends on the standard library only: the roof status
file and the settings file are the two artefacts other programs (ASCOM clients,
SkyRoof, NINA) read, so the code that produces and consumes them is kept small,
importable without a GUI, and directly testable.
"""

import json
import os
import re
import shutil
import tempfile
import time

# os.replace needs DELETE access on the destination. On Windows a reader that
# opened the roof status file without FILE_SHARE_DELETE blocks that until it
# closes the handle, which is a matter of milliseconds for a polling client.
_REPLACE_RETRIES = 10
_REPLACE_RETRY_DELAY = 0.05

# Tokens recognised in a roof status line, and the value each one reports.
_STATUS_TOKENS = {
    "OPEN": "OPEN",
    "CLOSED": "CLOSED",
    "CLOSE": "CLOSED",
    "SHUT": "CLOSED",
}

# Words that invert the meaning of a following status token ("roof is not open").
_NEGATIONS = {"NOT", "NO", "ISN'T", "AREN'T", "NEVER"}

_WORD_RE = re.compile(r"[A-Z']+")


def _replace_with_retry(tmp_path, path):
    """os.replace, retrying while a reader still holds *path* open (Windows)."""
    last_error = None
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < _REPLACE_RETRIES - 1:
                time.sleep(_REPLACE_RETRY_DELAY)
    raise last_error


def atomic_write_text(path, text, encoding="utf-8"):
    """Write *text* to *path* so readers never observe a partial file.

    The content goes to a temporary file in the same directory, is flushed and
    fsynced, and is then moved into place with os.replace, which is atomic on
    both POSIX and Windows. A reader polling the roof status file therefore sees
    either the previous line or the new one, never a truncated or empty file.
    (On Windows a reader can still collide with the rename itself and get a
    transient open error; that is a retry, not a wrong answer.)

    If the rename keeps failing because a reader is holding the destination open,
    the content is written in place instead. That reintroduces the small window a
    plain write has always had, but it is strictly better than skipping the
    update and leaving a stale roof status on disk.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
    )
    wrote_in_place = False
    try:
        with os.fdopen(tmp_fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _replace_with_retry(tmp_path, path)
        except PermissionError:
            with open(path, "w", encoding=encoding, newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            wrote_in_place = True
    except BaseException:
        wrote_in_place = True  # nothing was moved into place; drop the temp file
        raise
    finally:
        if wrote_in_place:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def write_json_atomic(path, obj):
    """Serialise *obj* as indented JSON and write it atomically."""
    atomic_write_text(path, json.dumps(obj, indent=2))


def load_json_with_recovery(path, on_corrupt=None):
    """Load JSON from *path*, quarantining the file if it cannot be parsed.

    Returns the decoded object, or None when the file is missing or unreadable.
    A file that exists but does not parse is moved aside to ``<path>.corrupt`` so
    the next save starts from a clean slate instead of failing forever, and
    *on_corrupt* (if given) is called with the backup path and the exception.
    A truncated settings file otherwise silently resets every setting, including
    the persisted ASCOM UniqueID that clients rely on to reconnect.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (ValueError, UnicodeDecodeError) as exc:
        backup = path + ".corrupt"
        try:
            shutil.move(path, backup)
        except OSError:
            backup = None
        if on_corrupt:
            on_corrupt(backup, exc)
        return None


def format_status_line(status, reason="", timestamp=""):
    """Build the single roof status line written to the output file.

    Format follows the SRO Roof File spec:
    https://interactiveastronomy.com/skyroof_help/SROrooffile.html
    """
    return f"???{timestamp} Roof Status: {status}{reason}\n"


def parse_roof_status(text):
    """Parse OPEN/CLOSED out of the last non-empty line of *text*.

    Returns ``(status, last_line)``; *status* is None when nothing could be
    parsed. Matching is done on whole words, and the result is deliberately
    conservative:

    * A negated token ("roof is not open") yields None rather than its opposite
      — the writer's intent is unclear, and guessing OPEN would be unsafe.
    * When both OPEN and CLOSED appear on the line, CLOSED wins, so an ambiguous
      line can never be reported as an open roof.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, ""

    last_line = lines[-1]
    words = _WORD_RE.findall(last_line.upper())

    found = set()
    for index, word in enumerate(words):
        status = _STATUS_TOKENS.get(word)
        if status is None:
            continue
        if index > 0 and words[index - 1] in _NEGATIONS:
            # "not open" / "no open" - ambiguous, refuse to guess.
            return None, last_line
        found.add(status)

    if "CLOSED" in found:
        return "CLOSED", last_line
    if "OPEN" in found:
        return "OPEN", last_line
    return None, last_line
