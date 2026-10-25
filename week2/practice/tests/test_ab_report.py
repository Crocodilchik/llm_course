import copy
import json
import socket

import pytest

import ab_report as ab
import agent_loop as agent
import check
from test_agent_loop import call, client_with, response
from test_evaluation import make_truth_db


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network_forbidden_in_tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def sample_traces():
    traces = []
    for variant in ("short", "full"):
        for index, (question, _) in enumerate(check.TASKS):
            expected = ab.EXPECTED[question]
            for attempt in range(1, 4):
                traces.append(dict(question=question, variant=variant, format_version=2,
                    trace_id=f"{variant}-{index}-{attempt}", experiment_id="unit-complete",
                    task_id=f"task-{index+1:02d}", attempt=attempt, planned_attempts=3, max_steps=6,
                    model_fingerprint="model-hash", protocol_fingerprint="protocol-hash",
                    implementation_fingerprint="code-hash", suite_fingerprint="suite-hash",
                    descriptions_fingerprint=f"{variant}-descriptions", input_tokens=10,
                    output_tokens=2, usage_complete=True,
                    tool_calls=[dict(name=expected["tool"], args=copy.deepcopy(expected["args"]))]))
    return traces


def test_wrong_coordinates_do_not_receive_credit():
    trace = sample_traces()[7 * 3]
    assert ab.correct_call(trace)
    trace["tool_calls"][0]["args"]["lat"] = 47.0
    assert not ab.correct_call(trace)


def test_filter_default_is_checked_and_missing_false_is_valid():
    trace = sample_traces()[3]
    del trace["tool_calls"][0]["args"]["only_24_7"]
    assert ab.correct_call(trace)
    trace["tool_calls"][0]["args"]["only_24_7"] = True
    assert not ab.correct_call(trace)


def test_unrequested_limits_are_valid_if_bounded():
    trace = sample_traces()[6 * 3]
    trace["tool_calls"][0]["args"].update(limit=5, street="Пушкинская")
    assert ab.correct_call(trace)
    trace["tool_calls"][0]["args"]["limit"] = -1
    assert not ab.correct_call(trace)


def test_incomplete_duplicate_or_mixed_experiments_are_rejected():
    for mutate in (
        lambda ts: ts.pop(),
        lambda ts: ts.append(copy.deepcopy(ts[0])),
        lambda ts: ts[0].update(model_fingerprint="different-model"),
        lambda ts: ts[0].update(protocol_fingerprint="different-protocol"),
        lambda ts: ts[0].update(suite_fingerprint="different-dataset"),
        lambda ts: ts[0].update(attempt=2),
        lambda ts: ts[0].update(task_id="wrong-task"),
        lambda ts: ts[0].update(descriptions_fingerprint="different-descriptions"),
    ):
        traces = sample_traces()
        mutate(traces)
        with pytest.raises(ValueError):
            ab.summarize(traces)


def test_summary_reports_numeric_efficiency_and_unknown_usage():
    traces = sample_traces()
    result = ab.summarize(traces)
    assert result["short"]["correct"] == 30
    assert result["short"]["tokens_per_correct_choice"] == 12
    assert result["comparison"]["delta_accuracy"] == 0
    assert result["comparison"]["extra_tokens_per_extra_correct_choice"] is None
    traces[0].update(input_tokens=None, output_tokens=None, usage_complete=False)
    result = ab.summarize(traces)
    assert result["short"]["avg_total"] is None
    assert result["comparison"]["extra_tokens_per_attempt"] is None


def test_no_correct_calls_has_no_division_by_zero():
    traces = sample_traces()
    for trace in traces:
        trace["tool_calls"] = []
    assert ab.summarize(traces)["short"]["tokens_per_correct_choice"] is None


def test_legacy_loader_does_not_pool_other_files(monkeypatch, tmp_path):
    traces = sample_traces()
    for trace in traces:
        trace.pop("format_version")
    (tmp_path / "traces.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in traces), encoding="utf-8")
    (tmp_path / "failures.jsonl").write_text("not json", encoding="utf-8")
    monkeypatch.setattr(ab, "RUNS_DIR", tmp_path)
    assert len(ab.load_runs(legacy=True)) == 60
    assert ab.summarize(ab.load_runs(legacy=True), legacy=True)["full"]["n"] == 30
    with pytest.raises(ValueError):
        ab.load_runs()


def test_full_offline_suite_writes_and_reloads_sixty_typed_traces(monkeypatch, tmp_path):
    make_truth_db(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    monkeypatch.setattr(agent, "RUNS_DIR", runs)
    monkeypatch.setattr(ab, "RUNS_DIR", runs)
    for name in agent.EXECUTORS:
        monkeypatch.setitem(agent.EXECUTORS, name, lambda **kw: "synthetic observation")
    final_answers = ([dict(type="count", count=int(expected)) for _, expected in check.TASKS[:6]]
        + [dict(type="places", places=[dict(id=1, name="Coffee Man")]),
           dict(type="nearest", place=dict(id=3, name="Test pharmacy"), distance_m=0),
           dict(type="nearest", place=dict(id=4, name="Test school"), distance_m=0),
           dict(type="brands", brands=["Ozon"])])

    def create(**kwargs):
        messages = kwargs["messages"]
        question = messages[1]["content"]
        index = next(i for i, (q, _) in enumerate(check.TASKS) if q == question)
        if any(m["role"] == "tool" for m in messages):
            return response(answer=json.dumps(final_answers[index], ensure_ascii=False))
        expected = ab.EXPECTED[question]
        return response(calls=[call(expected["tool"], expected["args"])])

    client = client_with()
    client.chat.completions.create.side_effect = create
    for variant in ("short", "full"):
        traces = check.run_suite(client, "synthetic-model", 3, variant, 6, "unit-integration-20261021-01")
        assert len(traces) == 30 and all(t["solved"] for t in traces)
    loaded = ab.load_runs("unit-integration-20261021-01")
    result = ab.summarize(loaded)
    assert result["short"]["n"] == 30 and result["full"]["correct"] == 30
    assert result["full"]["avg_total"] == 24
    assert result["full"]["tokens_per_task"] == 72
