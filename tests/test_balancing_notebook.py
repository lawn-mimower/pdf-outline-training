import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

NOTEBOOK = Path(__file__).resolve().parents[1] / "AIH_model" / "LGBM_balancer.ipynb"
CLASS_COUNTS = {0: 160, 1: 14, 2: 40, 3: 30, 4: 20, 5: 15}  # BODY_TEXT dominates, TITLE/H4 are rare


def write_training_csv(path):
    """stage3-style CSV: 12 overlapping feature columns with a few NaNs, plus the label."""
    rng = np.random.default_rng(0)
    features, labels = [], []
    for label, n in CLASS_COUNTS.items():
        features.append(rng.normal(loc=label * 0.7, scale=1.0, size=(n, 12)))
        labels += [label] * n
    df = pd.DataFrame(np.vstack(features), columns=[f"feature_{i}" for i in range(12)])
    df.loc[::13, "feature_11"] = np.nan
    df["label"] = labels
    df.sample(frac=1, random_state=0).to_csv(path, index=False)


def run_notebook(balancing, tmp_path, monkeypatch):
    csv = tmp_path / "final_training_dataset.csv"
    write_training_csv(csv)
    monkeypatch.setenv("DATASET_PATH", str(csv))
    monkeypatch.setenv("BALANCING", balancing)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    namespace = {}
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    return namespace


@pytest.mark.parametrize("balancing", ["class_weight", "smote", "adasyn"])
def test_each_balancing_option_trains_and_evaluates(balancing, tmp_path, monkeypatch):
    ns = run_notebook(balancing, tmp_path, monkeypatch)

    model = ns["lgbm"]
    assert model.get_params().get("is_unbalance", False) is (balancing == "class_weight")
    assert len(ns["y_pred"]) == len(ns["y_test"])
    assert 0.0 <= ns["accuracy"] <= 1.0

    y_resampled = pd.Series(ns["y_train_resampled"])
    if balancing == "class_weight":
        # the model reweights classes itself; the training split is used as is
        assert len(y_resampled) == len(ns["y_train"])
    else:
        # oversampling fills the minority classes of the training split only
        counts = y_resampled.value_counts()
        assert len(y_resampled) > len(ns["y_train"])
        assert set(counts.index) == set(CLASS_COUNTS)
        assert counts.min() >= 0.8 * counts.max()
        assert not ns["X_train_resampled"].isnull().any().any()
        if balancing == "smote":
            assert counts.nunique() == 1


def test_unknown_balancing_option_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="BALANCING must be one of"):
        run_notebook("undersample", tmp_path, monkeypatch)
