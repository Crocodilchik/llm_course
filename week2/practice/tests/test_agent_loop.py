import json
import socket
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

import agent_loop as agent
import check_provider
from answer_schema import parse_answer


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network_forbidden_in_tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def response(*, calls=None, answer=None, usage=True):
    return NS(choices=[NS(message=NS(tool_calls=calls, content=answer))],
              usage=NS(prompt_tokens=10, completion_tokens=2) if usage else None)


def call(name="count_places", args=None):
    return NS(id="call_test", function=NS(name=name, arguments=json.dumps(
        {"kind": "pharmacy"} if args is None else args)))


def client_with(*responses):
    return NS(chat=NS(completions=NS(create=Mock(side_effect=list(responses)))))


def test_api_failure_preserves_progress_and_saves_safe_trace(monkeypatch, tmp_path):
    monkeypatch.setitem(agent.EXECUTORS, "count_places", lambda **kw: "pharmacy: 19")
    client = client_with(response(calls=[call()]), RuntimeError("api_key=FAKE_SECRET_MARKER"))
    trace = agent.run_agent(client, "test-model", "question", "full", experiment_id="unit-api")
    assert trace["steps"] == 2 and trace["known_input_tokens"] == 10
    assert trace["known_output_tokens"] == 2 and len(trace["tool_calls"]) == 1
    assert trace["input_tokens"] is None and not trace["usage_complete"]
    assert trace["error"].startswith("api_error")
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    path = agent.save_run(trace)
    saved = path.read_text(encoding="utf-8")
    assert "FAKE_SECRET_MARKER" not in saved and "test-model" not in saved
    assert json.loads(saved)["trace_id"] == trace["trace_id"]


@pytest.mark.parametrize("name,args", [
    ("find_places", {"kind": "cafe", "limit": -1}),
    ("find_places", {"kind": "cafe", "limit": 51}),
    ("count_places", {"kind": "pharmacy", "only_24_7": "false"}),
    ("count_places", {"kind": "pharmacy", "extra": 1}),
    ("count_places", []),
    ("count_places", {"kind": None}),
    ("place_details", {"place_id": True}),
    ("nearest_place", {"kind": "cafe", "lat": 91, "lon": 39}),
    ("nearest_place", {"kind": "cafe", "lat": True, "lon": 39}),
    ("nearest_place", {"kind": "cafe", "lat": float("nan"), "lon": 39}),
    ("unknown_tool", {}),
])
def test_bad_arguments_become_observations(monkeypatch, name, args):
    executor = Mock(return_value="must not execute")
    if name in agent.EXECUTORS:
        monkeypatch.setitem(agent.EXECUTORS, name, executor)
    client = client_with(response(calls=[call(name, args)]),
                         response(answer='{"type":"unavailable","reason":"no_data"}'))
    trace = agent.run_agent(client, "test-model", "question", "short")
    executor.assert_not_called()
    assert trace["tool_calls"][0]["error"] == "invalid_arguments"
    assert trace["final_answer"]["type"] == "unavailable"
    assert trace["error"] is None
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert any(m["role"] == "tool" and "invalid_arguments" in m["content"] for m in messages)


def test_malformed_json_is_not_executed(monkeypatch):
    bad = call()
    bad.function.arguments = "{invalid"
    executor = Mock()
    monkeypatch.setitem(agent.EXECUTORS, "count_places", executor)
    trace = agent.run_agent(client_with(response(calls=[bad])), "test-model", "q", "full", 1)
    assert trace["tool_calls"][0]["error"] == "invalid_arguments"
    executor.assert_not_called()


def test_tool_error_can_recover_without_exception_text(monkeypatch):
    def broken(**kwargs):
        raise ZeroDivisionError("password=FAKE_SECRET_MARKER")
    monkeypatch.setitem(agent.EXECUTORS, "count_places", broken)
    client = client_with(response(calls=[call()]), response(answer='{"type":"count","count":19}'))
    trace = agent.run_agent(client, "test-model", "q", "full")
    assert trace["error"] is None and trace["final_answer"]["count"] == 19
    assert "ZeroDivisionError" in trace["tool_calls"][0]["result"]
    assert "FAKE_SECRET_MARKER" not in json.dumps(trace)
    assert "FAKE_SECRET_MARKER" not in str(client.chat.completions.create.call_args_list)


def test_step_limit_is_finite(monkeypatch):
    monkeypatch.setitem(agent.EXECUTORS, "count_places", lambda **kw: "try again")
    client = client_with(*(response(calls=[call()]) for _ in range(3)))
    trace = agent.run_agent(client, "test-model", "q", "short", 3)
    assert trace["stopped_by_limit"] and trace["error"] == "step_limit"
    assert client.chat.completions.create.call_count == 3


def test_missing_usage_remains_unknown(monkeypatch):
    monkeypatch.setitem(agent.EXECUTORS, "count_places", lambda **kw: "19")
    trace = agent.run_agent(client_with(response(calls=[call()], usage=False),
        response(answer='{"type":"count","count":19}')), "test-model", "q", "full")
    assert trace["input_tokens"] is None and trace["output_tokens"] is None
    assert trace["known_input_tokens"] == 10 and not trace["usage_complete"]


def test_large_result_is_not_sent_to_model(monkeypatch):
    monkeypatch.setitem(agent.EXECUTORS, "count_places", lambda **kw: "x" * 10_000)
    trace = agent.run_agent(client_with(response(calls=[call()])), "test-model", "q", "full", 1)
    assert trace["tool_calls"][0]["error"] == "result_too_large"
    assert len(trace["tool_calls"][0]["result"]) < 200


def test_invalid_final_can_be_corrected_without_retaining_raw_text():
    client = client_with(response(answer="password=FAKE_SECRET_MARKER"),
                         response(answer='{"type":"count","count":19}'))
    trace = agent.run_agent(client, "test-model", "q", "full", 2)
    assert trace["answer_errors"] == 1 and trace["final_answer"]["count"] == 19
    assert "FAKE_SECRET_MARKER" not in str(client.chat.completions.create.call_args_list)


@pytest.mark.parametrize("answer", [
    {"type": "count", "count": True}, {"type": "count", "count": "19"},
    {"type": "count", "count": 19, "text": "119"},
    {"type": "nearest", "place": {"id": 1}, "distance_m": -1},
    {"type": "places", "places": []}, {"type": "brands", "brands": [""]},
    {"type": "unavailable", "reason": "arbitrary_claim"}, "[19]",
])
def test_answer_contract_rejects_ambiguous_data(answer):
    with pytest.raises(ValueError):
        parse_answer(answer)


def test_redaction_does_not_need_real_configuration():
    marker = "sk-" + "x" * 30
    source = {"nested": ["Bearer FAKE_SECRET_MARKER", marker, "api_key=FAKE_SECRET_MARKER"],
              "OPENAI_API_KEY": "FAKE_SECRET_MARKER", "phone": "+7 (000) 000-00-00",
              "contact": "nobody@example.invalid", "url": "https://user:pass@example.invalid",
              "forged_fingerprint": marker}
    cleaned = json.dumps(agent.safe_value(source))
    assert "FAKE_SECRET_MARKER" not in cleaned and marker not in cleaned
    assert "nobody@" not in cleaned and "user:pass" not in cleaned
    assert "+7 (000)" not in cleaned


def test_card_includes_address_without_contact(monkeypatch):
    monkeypatch.setattr(agent.sdk_tasks, "details", lambda _: dict(
        id=42, name="Test place", street="Test street", housenumber="1",
        opening_hours=None, phone="+7 (000) 000-00-00"))
    result = agent._place_details(42)
    assert "Test street 1" in result and "id=42" in result
    assert "+7" not in result


def test_write_paths_cannot_escape_practice(monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        agent.inside_practice(agent.HERE.parent / "forbidden.jsonl")
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    with pytest.raises(ValueError):
        agent.save_run({"experiment_id": "../forbidden"})


def test_dated_metadata_ids_survive_while_contact_text_is_redacted(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    trace = {"experiment_id": "live-ab-20261021-01", "trace_id": "trace-20261021-01",
             "task_id": "task-01", "question": "+7 (000) 000-00-00",
             "nested": {"experiment_id": "+7 (000) 000-00-00"}}
    stored = json.loads(agent.save_run(trace).read_text(encoding="utf-8"))
    for field in ("experiment_id", "trace_id", "task_id"):
        assert stored[field] == trace[field]
    assert "+7 (000)" not in stored["question"]
    assert "+7 (000)" not in stored["nested"]["experiment_id"]


def test_credential_prefix_cannot_be_preserved_as_metadata_id(monkeypatch, tmp_path):
    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path / "runs")
    fake_token = "sk-" + "x" * 30
    with pytest.raises(ValueError):
        agent.save_run({"experiment_id": fake_token})
    with pytest.raises(ValueError):
        agent.save_run({"experiment_id": "unit-safe", "trace_id": fake_token})


def test_provider_diagnostics_hide_exception(monkeypatch, capsys):
    def broken():
        raise RuntimeError("api_key=FAKE_SECRET_MARKER")
    monkeypatch.setattr(check_provider, "load_client", broken)
    assert check_provider.main() == 1
    assert "FAKE_SECRET_MARKER" not in capsys.readouterr().out


def test_provider_probe_rejects_wrong_arguments():
    assert not check_provider.probe(client_with(response(calls=[call(args={"kind": "bank"})])), "test")
    assert check_provider.probe(client_with(response(calls=[call()])), "test")
