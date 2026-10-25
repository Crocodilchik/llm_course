import json
import socket
import sqlite3
import ssl
from unittest.mock import MagicMock

import pytest

import make_poi_dataset as dataset


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network_forbidden_in_tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)


def poi_row():
    return (1, "node", "amenity", "cafe", "Test", None, None, None,
            None, None, None, None, None, None, None, None, None, 47.2, 39.7)


def test_failed_build_keeps_existing_database(monkeypatch, tmp_path):
    target = tmp_path / "poi.sqlite"
    target.write_bytes(b"ORIGINAL_TEST_DATABASE")
    monkeypatch.setattr(dataset, "DB", target)
    with pytest.raises(sqlite3.Error):
        dataset.build([("wrong row shape",)])
    assert target.read_bytes() == b"ORIGINAL_TEST_DATABASE"


def test_successful_build_replaces_only_after_verification(monkeypatch, tmp_path):
    target = tmp_path / "poi.sqlite"
    target.write_bytes(b"ORIGINAL_TEST_DATABASE")
    monkeypatch.setattr(dataset, "DB", target)
    dataset.build([poi_row()])
    con = sqlite3.connect(target)
    assert con.execute("SELECT COUNT(*) FROM poi").fetchone()[0] == 1
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()


def test_empty_build_preserves_target(monkeypatch, tmp_path):
    target = tmp_path / "poi.sqlite"
    target.write_bytes(b"ORIGINAL_TEST_DATABASE")
    monkeypatch.setattr(dataset, "DB", target)
    with pytest.raises(ValueError):
        dataset.build([])
    assert target.read_bytes() == b"ORIGINAL_TEST_DATABASE"


def test_outside_target_is_rejected(monkeypatch):
    monkeypatch.setattr(dataset, "DB", dataset.HERE.parent / "forbidden.sqlite")
    with pytest.raises(ValueError):
        dataset.build([poi_row()])


def fake_response(payload):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
    return response


def test_download_verifies_tls_and_atomically_saves_snapshot(monkeypatch, tmp_path):
    target = tmp_path / "snapshot.json"
    target.write_bytes(b"OLD_TEST_SNAPSHOT")
    monkeypatch.setattr(dataset, "RAW", target)
    monkeypatch.setattr(dataset, "KEYS", ["amenity"])
    monkeypatch.setattr(dataset, "MIRRORS", ["https://example.invalid/overpass"])

    def urlopen(request, *, timeout, context):
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        return fake_response(dict(elements=[dict(id=1, type="node", lat=47.2, lon=39.7)]))

    monkeypatch.setattr(dataset.urllib.request, "urlopen", urlopen)
    raw = dataset.fetch_raw()
    assert len(raw["elements"]) == 1
    assert json.loads(target.read_text(encoding="utf-8")) == raw


def test_partial_download_never_replaces_snapshot(monkeypatch, tmp_path):
    target = tmp_path / "snapshot.json"
    target.write_bytes(b"OLD_TEST_SNAPSHOT")
    monkeypatch.setattr(dataset, "RAW", target)
    monkeypatch.setattr(dataset, "KEYS", ["amenity", "shop"])
    monkeypatch.setattr(dataset, "MIRRORS", ["https://example.invalid/overpass"])
    attempts = iter([fake_response(dict(elements=[dict(id=1, type="node")])), OSError("offline")])

    def urlopen(*args, **kwargs):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(dataset.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError):
        dataset.fetch_raw()
    assert target.read_bytes() == b"OLD_TEST_SNAPSHOT"


def test_overpass_timeout_remark_is_not_success(monkeypatch, tmp_path):
    target = tmp_path / "snapshot.json"
    target.write_bytes(b"OLD_TEST_SNAPSHOT")
    monkeypatch.setattr(dataset, "RAW", target)
    monkeypatch.setattr(dataset, "KEYS", ["amenity"])
    monkeypatch.setattr(dataset, "MIRRORS", ["https://example.invalid/overpass"])
    monkeypatch.setattr(dataset.urllib.request, "urlopen", lambda *a, **k: fake_response(
        dict(elements=[dict(id=1, type="node")], remark="runtime timeout")))
    with pytest.raises(RuntimeError):
        dataset.fetch_raw()
    assert target.read_bytes() == b"OLD_TEST_SNAPSHOT"
