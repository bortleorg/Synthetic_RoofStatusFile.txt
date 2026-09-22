"""The latest-image preview must fill its panel and refit as the panel resizes.

It used to be scaled to a fixed 460x380 cap, so a maximized window showed a small
image in a large black box. These tests run headless: Tk is replaced by fakes that
record scheduled callbacks and decode the rendered PNG to report its size.
"""

import base64

import cv2
import numpy as np
import pytest

import synthetic_roofstatus as srs


class FakeRoot:
    """Records after() jobs so the test can see what is queued and run it."""

    def __init__(self):
        self.jobs = {}
        self._next = 0

    def after(self, _ms, func, *args):
        self._next += 1
        job = f"after#{self._next}"
        self.jobs[job] = (func, args)
        return job

    def after_cancel(self, job):
        self.jobs.pop(job, None)

    def run_pending(self):
        for job in list(self.jobs):
            func, args = self.jobs.pop(job)
            func(*args)


class FakeHolder:
    def __init__(self, width, height):
        self.width, self.height = width, height

    def winfo_width(self):
        return self.width

    def winfo_height(self):
        return self.height


class FakeLabel:
    def __init__(self):
        self.options = {}

    def config(self, **kwargs):
        self.options.update(kwargs)


class FakePhotoImage:
    """Decodes the base64 PNG the app hands Tk, so the rendered size is observable."""

    def __init__(self, data):
        png = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        self._img = cv2.imdecode(png, cv2.IMREAD_COLOR)

    def width(self):
        return self._img.shape[1]

    def height(self):
        return self._img.shape[0]


@pytest.fixture
def preview(app, monkeypatch):
    monkeypatch.setattr(srs.tk, "PhotoImage", FakePhotoImage)
    app.root = FakeRoot()
    app.preview_holder = FakeHolder(1406, 806)
    app.preview_label = FakeLabel()
    app._preview_src = None
    app._preview_resize_job = None
    app._preview_tk_img = None
    return app


def frame(width, height):
    return np.full((height, width, 3), 90, dtype=np.uint8)


def rendered_size(app):
    return app._preview_tk_img.width(), app._preview_tk_img.height()


def test_image_fills_the_panel_height_keeping_its_aspect_ratio(preview):
    preview.preview_holder.width, preview.preview_holder.height = 1760, 806  # maximized
    preview._preview_src = frame(640, 360)  # 16:9, as in a wide camera frame
    preview._render_preview()

    width, height = rendered_size(preview)
    avail_w, avail_h = 1760 - 6, 806 - 6
    assert height == avail_h                       # upscaled to fill, not capped
    assert width <= avail_w
    assert abs(width / height - 640 / 360) < 0.01  # not stretched


def test_image_follows_the_panel_when_it_shrinks(preview):
    preview._preview_src = frame(640, 360)
    preview._render_preview()
    preview.preview_holder.width, preview.preview_holder.height = 406, 306

    preview._on_preview_resized()
    preview.root.run_pending()

    width, height = rendered_size(preview)
    assert width == 406 - 6
    assert height <= 306 - 6


def test_repeated_resize_events_coalesce_into_one_refit(preview):
    preview._preview_src = frame(640, 360)
    renders = []
    original = preview._render_preview
    preview._render_preview = lambda: (renders.append(1), original())

    for _ in range(10):
        preview._on_preview_resized()

    assert len(preview.root.jobs) == 1
    preview.root.run_pending()
    assert len(renders) == 1


def test_a_new_frame_cancels_a_queued_resize_refit(preview):
    preview._preview_src = frame(640, 360)
    preview._on_preview_resized()
    assert len(preview.root.jobs) == 1

    preview._render_preview()  # what _apply_preview does when a frame arrives

    assert preview.root.jobs == {}
    assert preview._preview_resize_job is None


def test_nothing_is_rendered_before_the_panel_is_laid_out(preview):
    preview._preview_src = frame(640, 360)
    preview.preview_holder.width, preview.preview_holder.height = 1, 1

    preview._render_preview()

    assert preview._preview_tk_img is None
