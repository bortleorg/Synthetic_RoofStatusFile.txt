"""Training and loading a model fail with a message, not a traceback.

Training with only one class raised out of the button handler with no message.
Load Model was unguarded. A model trained at a different image size loaded
without complaint and then failed every monitoring pass for the whole night.
"""

import os

import cv2
import numpy as np
import pytest
from joblib import dump
from sklearn.linear_model import LogisticRegression

import synthetic_roofstatus as srs


class Var:
    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


@pytest.fixture
def dialogs(monkeypatch):
    shown = {"error": [], "info": []}
    monkeypatch.setattr(srs.messagebox, "showerror", lambda title, msg: shown["error"].append(msg))
    monkeypatch.setattr(srs.messagebox, "showinfo", lambda title, msg: shown["info"].append(msg))
    return shown


@pytest.fixture
def trainer(app, tmp_path, dialogs):
    app.training_data_folder = Var(str(tmp_path / "train"))
    app.model_path = Var("")
    app.model = None
    app.save_settings = lambda: None
    return app


def add_images(base, label, count, value):
    folder = base / label
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        cv2.imwrite(str(folder / f"{label}{i}.png"),
                    np.full((40, 40), value + i, dtype=np.uint8))


def fit_model(n_features):
    rng = np.random.default_rng(0)
    X = rng.random((6, n_features))
    return LogisticRegression().fit(X, [0, 1, 0, 1, 0, 1])


# ── training ──────────────────────────────────────────────────────────────────

def test_training_with_one_class_shows_an_error(trainer, tmp_path, dialogs):
    add_images(tmp_path / "train", "open", 3, 200)

    trainer.train_model()

    assert trainer.model is None
    assert len(dialogs["error"]) == 1
    assert "closed" in dialogs["error"][0]


def test_training_with_both_classes_succeeds(trainer, tmp_path, dialogs):
    add_images(tmp_path / "train", "open", 3, 200)
    add_images(tmp_path / "train", "closed", 3, 20)

    trainer.train_model()

    assert trainer.model is not None
    assert dialogs["error"] == []


def test_failed_save_keeps_the_trained_model(trainer, tmp_path, dialogs):
    add_images(tmp_path / "train", "open", 3, 200)
    add_images(tmp_path / "train", "closed", 3, 20)
    trainer.model_path = Var(str(tmp_path / "no" / "such" / "dir" / "m.joblib"))

    trainer.train_model()

    assert trainer.model is not None
    assert "could not save" in dialogs["info"][0]


# ── loading ───────────────────────────────────────────────────────────────────

def test_loading_a_matching_model(tmp_path):
    path = tmp_path / "good.joblib"
    dump(fit_model(srs.IMG_SIZE * srs.IMG_SIZE), path)

    model = srs.RoofClassifierApp._load_model_file(str(path))

    assert model.n_features_in_ == srs.IMG_SIZE * srs.IMG_SIZE


def test_loading_a_model_of_the_wrong_size_is_rejected(tmp_path):
    path = tmp_path / "old.joblib"
    dump(fit_model(64 * 64), path)

    with pytest.raises(ValueError, match="Retrain"):
        srs.RoofClassifierApp._load_model_file(str(path))


def test_loading_something_that_is_not_a_model_is_rejected(tmp_path):
    path = tmp_path / "junk.joblib"
    dump({"not": "a model"}, path)

    with pytest.raises(ValueError, match="not a classifier"):
        srs.RoofClassifierApp._load_model_file(str(path))


def test_bundled_example_model_loads():
    example = os.path.join(os.path.dirname(__file__), "..", "examples", "roofgpt_nano.joblib")
    srs.RoofClassifierApp._load_model_file(example)


def test_load_model_button_reports_a_bad_file(app, tmp_path, dialogs, monkeypatch):
    path = tmp_path / "old.joblib"
    dump(fit_model(64 * 64), path)
    monkeypatch.setattr(srs.filedialog, "askopenfilename", lambda **k: str(path))
    app.model = "previous"
    app.model_path = Var("previous.joblib")
    app.save_settings = lambda: None

    app.load_model()

    assert app.model == "previous", "a rejected file replaced the working model"
    assert app.model_path.get() == "previous.joblib"
    assert len(dialogs["error"]) == 1


# ── no duplicate definitions shadowing each other ─────────────────────────────

def test_no_method_is_defined_twice():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(srs))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoofClassifierApp")
    names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"methods defined more than once: {duplicates}"
