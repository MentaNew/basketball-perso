import numpy as np

from ml.experiments import log_run, read_runs


def test_runs_are_appended_and_read_back(tmp_path):
    path = tmp_path / "metrics.jsonl"
    log_run("demo", metrics={"mae": np.float64(7.5)}, params={"k": 20}, path=path)
    log_run("demo", metrics={"mae": 7.1}, path=path)
    runs = read_runs(path)
    assert list(runs["metrics.mae"]) == [7.5, 7.1]         # numpy numbers stored as plain floats
    assert runs.loc[0, "params.k"] == 20
    assert {"time", "commit", "uncommitted_changes", "model"} <= set(runs.columns)
