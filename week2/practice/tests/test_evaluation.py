import socket
import sqlite3
from unittest.mock import Mock

import pytest

import check
import sdk_tasks


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network_forbidden_in_tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def make_truth_db(tmp_path, monkeypatch):
    path = tmp_path / "truth.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE poi(id INTEGER, name TEXT, kind TEXT, street TEXT, brand TEXT, lat REAL, lon REAL)")
    con.executemany("INSERT INTO poi VALUES (?, ?, ?, ?, ?, ?, ?)", [
        (1, "Coffee Man", "cafe", "Пушкинская", None, 47.2, 39.7),
        (2, "Test cafe", "cafe", "Other street", None, 47.2, 39.7),
        (3, "Test pharmacy", "pharmacy", None, None, 47.2940, 39.8880),
        (4, "Test school", "school", None, None, 47.2690, 39.7170),
        (5, "Test pickup", "outpost", None, "Ozon", 47.2, 39.7),
    ])
    con.commit()
    con.close()
    monkeypatch.setattr(sdk_tasks, "DB", path)
    return path


@pytest.fixture
def truth(tmp_path, monkeypatch):
    return make_truth_db(tmp_path, monkeypatch)


def test_count_is_not_a_substring_or_coerced_value():
    assert check.is_solved({"type": "count", "count": 19}, 0)
    assert not check.is_solved({"type": "count", "count": 119}, 0)
    assert not check.is_solved("не 19, а 119", 0)
    assert not check.is_solved({"type": "count", "count": "19"}, 0)
    assert not check.is_solved({"type": "count", "count": 19, "answer": "119"}, 0)


def test_all_listed_places_must_be_real_and_match_filter(truth):
    good = {"type": "places", "places": [{"id": 1, "name": "Coffee Man"}]}
    assert check.is_solved(good, 6)
    bad = {"type": "places", "places": good["places"] + [{"id": 2, "name": "Test cafe"}]}
    assert not check.is_solved(bad, 6)
    assert not check.is_solved({"type": "places", "places": [{"id": 99, "name": "Coffee Man"}]}, 6)


def test_nearest_requires_id_name_and_distance(truth):
    answer = {"type": "nearest", "place": {"id": 3, "name": "Test pharmacy"}, "distance_m": 0}
    assert check.is_solved(answer, 7)
    assert not check.is_solved({**answer, "distance_m": 1}, 7)
    assert not check.is_solved({**answer, "place": {"id": 4, "name": "Test pharmacy"}}, 7)
    assert not check.is_solved({**answer, "place": {"id": 3, "name": "Invented"}}, 7)


def test_brands_reject_invented_extras(truth):
    assert check.is_solved({"type": "brands", "brands": ["Ozon"]}, 9)
    assert not check.is_solved({"type": "brands", "brands": ["Ozon", "Invented"]}, 9)


def test_missing_truth_database_is_not_created(monkeypatch, tmp_path):
    missing = tmp_path / "missing.sqlite"
    monkeypatch.setattr(sdk_tasks, "DB", missing)
    with pytest.raises(sqlite3.Error):
        check.is_solved({"type": "brands", "brands": ["Ozon"]}, 9)
    assert not missing.exists()


def test_any_and_all_metrics_are_distinct():
    matrix = [[True, True, True], [True, False, True], [False, False, False]]
    assert check.pass_at_k(matrix, 1) == pytest.approx(2 / 3)
    assert check.pass_at_k(matrix, 3) == pytest.approx(2 / 3)
    assert check.pass_power_k(matrix, 3) == pytest.approx(1 / 3)
    assert check.pass_power_k(matrix, 3) != check.pass_at_k(matrix, 1) ** 3


@pytest.mark.parametrize("matrix,k", [([], 3), ([[True]], 3), ([[True], [True, False]], 1),
                                      ([[1]], 1), ([[True]], 0)])
def test_incomplete_metrics_are_rejected(matrix, k):
    with pytest.raises(ValueError):
        check.pass_power_k(matrix, k)
    with pytest.raises(ValueError):
        check.pass_at_k(matrix, k)


def test_wilson_at_boundaries():
    assert check.wilson_interval(1, 30) == (0.8865, 1.0)
    assert check.wilson_interval(0, 30) == (0.0, 0.1135)
    assert check.wilson_interval(0, 0) == (0.0, 1.0)


def test_token_cost_per_attempt_and_per_task():
    traces = [dict(input_tokens=10, output_tokens=2, usage_complete=True)] * 6
    summary = check.token_summary(traces, 2)
    assert summary["avg_total"] == 12 and summary["tokens_per_task"] == 36
    traces = traces + [dict(input_tokens=None, output_tokens=None, usage_complete=False)]
    summary = check.token_summary(traces, 2)
    assert summary["avg_total"] is None and summary["unknown_usage"] == 1


def test_api_failure_is_saved_before_suite_stops(monkeypatch, truth):
    trace = dict(error="api_error:Error", final_answer=None)
    monkeypatch.setattr(check, "run_agent", Mock(return_value=trace))
    saved = Mock()
    monkeypatch.setattr(check, "save_run", saved)
    with pytest.raises(RuntimeError):
        check.run_suite(object(), "test-model", 3, "full", 6, "unit-partial")
    saved.assert_called_once()
    assert saved.call_args.args[0]["solved"] is False
