"""Shared pytest fixtures.

The application is not packaged, so tests import the modules straight out of
``src/`` the same way the entry point does.
"""

import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture
def app():
    """A RoofClassifierApp with no Tk root and no GUI.

    Built with ``__new__`` so nothing is initialised. Any method that reaches for
    a Tk variable therefore raises AttributeError, which is exactly the failure
    these tests want to catch: the monitoring path must work from a plain
    configuration snapshot and never touch Tk off the UI thread.
    """
    import synthetic_roofstatus

    instance = synthetic_roofstatus.RoofClassifierApp.__new__(
        synthetic_roofstatus.RoofClassifierApp
    )
    instance.logger = None
    instance.ascom_server = None
    instance.override_active = None
    instance.override_expiry = None
    return instance
