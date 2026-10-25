import json
import socket
import sqlite3
from unittest.mock import Mock

import pytest

import agent_loop as agent
import failures
from test_agent_loop import call, client_with, response


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network_forbidden_in_tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def truth_db(tmp_path, monkeypatch):
    path = tmp_path / "diagnostic.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE poi (id INTEGER, kind TEXT, name TEXT, lat REAL, lon REAL, "
                "street TEXT, housenumber TEXT, opening_hours TEXT, brand TEXT)")
    for row in [(1, "pharmacy", "Near", 47.294, 39.888, "Street", "2", "24/7", None),
                (2, "pharmacy", "Far", 47.1, 39.5, None, None, None, None),
                (3, "school", "School", 47.269, 39.717, None, None, None, None),
                (4, "bank", "Bank", 47.2357, 39.7015, "Street", "4", None, None),
                (5, "outpost", "Point", 47.2, 39.7, None, None, None, "A"),
                (6, "outpost", "Point 2", 47.2, 39.7, None, None, None, None)]:
        con.execute("INSERT INTO poi VALUES (?,?,?,?,?,?,?,?,?)", row)
    con.commit()
    con.close()
    monkeypatch.setattr(failures.sdk_tasks, "DB", path)
    return path


def test_oracle_is_independent_of_injected_executors(truth_db):
    with failures.unavailable_storage(True):
        expected = failures.expected_answer(failures.SEARCH_TASKS[0])
        with pytest.raises(OSError):
            agent.EXECUTORS["nearest_place"]()
    assert expected == dict(type="details", place=dict(id=1, name="Near"),
                            address="Street 2", opening_hours="24/7")


def test_oracle_checks_all_brands_and_missing_fields(truth_db):
    expected = failures.expected_answer(failures.SEARCH_TASKS[-1])
    assert not failures.correct_answer(dict(type="brands", brands=["A"]), expected)
    assert failures.correct_answer(dict(type="brands", brands=["без бренда", "A"]), expected)
    card = failures.expected_answer(failures.SEARCH_TASKS[1])
    assert card["address"] is None and card["opening_hours"] is None
    assert failures.correct_answer(card, card)
    assert not failures.correct_answer({**card, "address": "Invented"}, card)


def test_storage_injection_restores_identity_and_protocol_on_exception():
    original = agent.EXECUTORS
    system, schemas, tools = agent.SYSTEM, agent.SCHEMAS, agent.make_tools("full")
    with pytest.raises(RuntimeError):
        with failures.unavailable_storage(True):
            assert agent.EXECUTORS is not original
            assert agent.make_tools("full") == tools
            raise RuntimeError("test interruption")
    assert agent.EXECUTORS is original
    assert agent.SYSTEM == system and agent.SCHEMAS is schemas


def test_task_outcome_is_distinct_from_safe_error_handling(truth_db):
    expected = dict(type="count", count=1)
    with failures.unavailable_storage(True):
        trace = agent.run_agent(client_with(response(calls=[call("count_places", {"kind": "bank"})]),
            response(answer='{"type":"unavailable","reason":"no_data"}')),
            "test-model", "verify banks", "full", 6)
    outcome = failures.assess(trace, expected, True)
    assert trace["error"] is None and not trace["stopped_by_limit"]
    assert outcome["outcome"] == "failed" and outcome["failure_class"] == "environment_error"
    assert not outcome["facts_correct"]


def test_recovery_is_success_and_api_failure_is_unassessed():
    expected = dict(type="count", count=1)
    trace = dict(final_answer=expected, error=None, stopped_by_limit=False,
                 tool_calls=[dict(error="tool_error:OSError"), dict(error=None)])
    assert failures.assess(trace, expected, False)["solved"] is True
    trace["error"] = "api_error:TimeoutError"
    result = failures.assess(trace, expected, False)
    assert result["solved"] is None and result["failure_class"] is None


def test_refusal_is_not_mislabeled_as_false_done():
    trace = dict(final_answer=dict(type="unavailable", reason="unsupported"),
                 error=None, tool_calls=[])
    result = failures.assess(trace, dict(type="count", count=1), False)
    assert result["failure_reason"] == "answerable_task_refused"
    assert result["failure_class"] == "incomplete_task"


def test_search_stops_at_three_and_persists_all_attempts(truth_db, tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    original = agent.EXECUTORS
    def fake_run(client, model, question, variant, max_steps, **metadata):
        return dict(**metadata, steps=2, error=None, tool_calls=[dict(error=None)],
                    final_answer=dict(type="count", count=0), usage_complete=True,
                    input_tokens=10, output_tokens=2)
    run = Mock(side_effect=fake_run)
    monkeypatch.setattr(agent, "run_agent", run)
    traces = failures.search(None, "test-model", "unit-failure-search")
    assert len(traces) == 3 and run.call_count == 3
    assert all(t["outcome"] == "failed" and not t["fault_injected"] for t in traces)
    saved = [json.loads(line) for line in (agent.RUNS_DIR / "experiment_unit-failure-search.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 3 and all(t["oracle_fingerprint"] for t in saved)
    assert agent.EXECUTORS is original
    with pytest.raises(ValueError, match="already_exists"):
        failures.search(None, "test-model", "unit-failure-search")
    assert run.call_count == 3


def test_search_keeps_successes_and_uses_bounded_fallback(truth_db, tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    original = agent.EXECUTORS
    def fake_run(client, model, question, variant, max_steps, **metadata):
        task = next((t for t in failures.SEARCH_TASKS if t["id"] == metadata["task_id"]), None)
        injected = agent.EXECUTORS is not original
        final = (dict(type="unavailable", reason="no_data") if injected
                 else failures.expected_answer(task))
        return dict(**metadata, steps=2, error=None, final_answer=final,
                    tool_calls=[dict(error="tool_error:OSError" if injected else None)],
                    usage_complete=True, input_tokens=10, output_tokens=2)
    monkeypatch.setattr(agent, "run_agent", fake_run)
    traces = failures.search(None, "test-model", "unit-fallback")
    assert len(traces) == 21
    assert sum(t["solved"] is True for t in traces) == 18
    assert sum(t["fault_injected"] for t in traces) == 3
    assert all(t["failure_class"] == "environment_error" for t in traces[-3:])
    assert agent.EXECUTORS is original


def test_unassessed_api_failure_stops_after_saving(truth_db, tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    run = Mock(side_effect=lambda *args, **metadata: dict(**metadata,
        final_answer=None, error="api_error:TimeoutError", tool_calls=[], usage_complete=False))
    monkeypatch.setattr(agent, "run_agent", run)
    with pytest.raises(RuntimeError, match="partial_trace_saved"):
        failures.search(None, "test-model", "unit-partial")
    assert run.call_count == 1
    saved = json.loads((agent.RUNS_DIR / "experiment_unit-partial.jsonl").read_text(encoding="utf-8"))
    assert saved["outcome"] == "unassessed" and saved["solved"] is None


def test_unknown_tool_loop_is_observed_without_changing_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    original = agent.EXECUTORS
    original_tools = agent.make_tools("full")
    client = client_with(*(response(calls=[call("count_places_verified", {"kind": "bank"})])
                           for _ in range(6)))
    traces = failures.search_unknown_tools(client, "test-model", "unit-unknown-loop")
    assert len(traces) == 1 and traces[0]["target_reproduced"]
    assert traces[0]["steps"] == 6 and traces[0]["error"] == "step_limit"
    assert traces[0]["repeated_unknown_tools"] == [dict(name="count_places_verified", consecutive_final_steps=6)]
    assert agent.EXECUTORS is original and agent.make_tools("full") == original_tools
    assert all(c.kwargs["tools"] == original_tools for c in client.chat.completions.create.call_args_list)
    assert all(c["error"] == "invalid_arguments" for c in traces[0]["tool_calls"])
    stored = json.loads((agent.RUNS_DIR / "experiment_unit-unknown-loop.jsonl").read_text(encoding="utf-8"))
    assert stored["failure_class"] == "wrong_tool" and stored["secondary_failure_class"] == "looping"
    assert not stored["fault_injected"] and len(stored["diagnostic_steps"]) == 6


def test_unknown_tool_once_then_refusal_does_not_count_as_loop():
    recorder = failures.ToolStepRecorder(client_with(
        response(calls=[call("count_places_verified", {"kind": "bank"})]),
        response(answer='{"type":"unavailable","reason":"unsupported"}')))
    trace = agent.run_agent(recorder, "test-model", "test", "full", 6)
    evidence = failures.unknown_tool_evidence(trace, recorder.steps)
    assert not evidence["target_reproduced"]
    assert evidence["unknown_tool_steps"] == {"count_places_verified": [1]}


def test_parallel_unknown_calls_on_one_step_do_not_count_as_iteration_loop():
    steps = [dict(step=i, advertised_tools=["count_places"],
                  tool_names=["count_places"] if i < 6 else ["missing"] * 6) for i in range(1, 7)]
    trace = dict(error="step_limit", stopped_by_limit=True, final_answer=None, max_steps=6,
                 tool_calls=[dict(name="missing", error="invalid_arguments") for _ in range(6)])
    assert not failures.unknown_tool_evidence(trace, steps)["target_reproduced"]


def test_unknown_tool_api_failure_is_saved_but_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    client = client_with(RuntimeError("api_key=FAKE_SECRET_MARKER"))
    with pytest.raises(RuntimeError, match="partial_trace_saved"):
        failures.search_unknown_tools(client, "test-model", "unit-unknown-api")
    stored_text = (agent.RUNS_DIR / "experiment_unit-unknown-api.jsonl").read_text(encoding="utf-8")
    stored = json.loads(stored_text)
    assert stored["outcome"] == "unassessed" and not stored["target_reproduced"]
    assert stored["diagnostic_steps"][0]["response_received"] is False
    assert "FAKE_SECRET_MARKER" not in stored_text
