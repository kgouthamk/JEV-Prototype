"""
Mock test block for the JEV P&C prototype.

    pytest -q                                  # offline: payloads, parsing, fallback, mock engine
    JEV_LIVE=1 BEATS_API_KEY=... pytest -q -m live -s   # dataset vs. real endpoint; paced 61s/call for the free tier
"""

from __future__ import annotations

import os
import time

import pytest
import requests

import jev_client

from jev_client import (
    CONFIDENCE_THRESHOLD,
    QUESTION_KEY,
    SCENARIOS,
    build_payload,
    evaluate,
    get_api_key,
    parse_answer,
    run_scenario_case,
)

ALL_CASES = [(s, c) for s, sc in SCENARIOS.items() for c in sc["cases"]]


# --------------------------------------------------------------------------- #
# Fake HTTP session returning canned JEV-shaped responses
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self._body, self.status_code, self.text = body, status, str(body)
        self.headers = headers or {}

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, body, status=200, exc=None, headers=None):
        self.body, self.status, self.exc, self.calls, self.headers = body, status, exc, [], headers

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self.exc:
            raise self.exc
        return FakeResponse(self.body, self.status, self.headers)


def wrap(answer):
    return {"model": "jev-1.13.0", "answers": {QUESTION_KEY: answer}, "usage": {"input_tokens": 200}}


CANNED = {
    ("fnol_triage", "A · High emergency"): wrap({"type": "noul", "noul": 0.97}),
    ("fnol_triage", "B · Low emergency"): wrap({"type": "noul", "noul": 0.03}),
    ("commercial_routing", "A · Workers comp"): wrap({
        "type": "choice", "choice": "workers_comp", "confidence": 0.93,
        "probabilities": {"commercial_auto": 0.0, "general_liability": 0.04, "workers_comp": 0.95, "property": 0.01}}),
    ("commercial_routing", "B · Commercial auto"): wrap({
        "type": "choice", "choice": "commercial_auto", "confidence": 0.96,
        "probabilities": {"commercial_auto": 0.98, "general_liability": 0.01, "workers_comp": 0.0, "property": 0.01}}),
    ("fraud_risk", "A · Low risk"): wrap({
        "type": "score", "score": 0.08, "confidence": 0.91,
        "probabilities": {"0": 0.93, "1": 0.06, "2": 0.01, "3": 0.0}}),
    ("fraud_risk", "B · High risk"): wrap({
        "type": "score", "score": 2.9, "confidence": 0.88,
        "probabilities": {"0": 0.0, "1": 0.01, "2": 0.08, "3": 0.91}}),
}


def assert_expectation(scenario, case, result):
    exp = SCENARIOS[scenario]["cases"][case]["expect"]
    for k, v in exp.items():
        assert result.detail[k] == v, f"{scenario}/{case}: {k}={result.detail[k]!r}, expected {v!r}"


# --------------------------------------------------------------------------- #
# Payload contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario,case", ALL_CASES)
def test_payload_matches_contract(scenario, case):
    sc = SCENARIOS[scenario]
    p = build_payload(sc["type"], sc["cases"][case]["text"], sc["instructions"], sc["criteria"])
    assert p["model"] == "jev-1.13-free"
    assert p["state"] == sc["cases"][case]["text"]
    q = p["questions"][QUESTION_KEY]
    assert q["type"] == sc["type"] and q["instructions"] == sc["instructions"]
    if sc["type"] == "score":
        assert isinstance(q["criteria"], list)
    elif sc["type"] == "choice":
        assert isinstance(q["criteria"], dict)
    else:
        assert set(q["criteria"]) == {"true", "false"}


def test_default_endpoint_is_beatapi_systemone():
    session = FakeSession(CANNED[("fnol_triage", "A · High emergency")])
    run_scenario_case("fnol_triage", "A · High emergency", api_key="k", session=session)
    assert session.calls[0]["url"] == "https://api.beatapi.io/v1/systemone"


def test_rate_limit_is_explained():
    r = evaluate("noul", "t", "i", api_key="k",
                 session=FakeSession({"error": "rate_limited"}, status=429, headers={"Retry-After": "42"}))
    assert "429" in r.error and "retry after 42s" in r.error and r.fallback


def test_beatapi_env_var_alias(monkeypatch):
    monkeypatch.delenv("BEATS_API_KEY", raising=False)
    monkeypatch.setenv("BEATAPI_API_KEY", "alias-key")
    session = FakeSession(wrap({"type": "noul", "noul": 0.97}))
    evaluate("noul", "t", "i", session=session)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer alias-key"


def test_payload_rejects_bad_criteria():
    with pytest.raises(ValueError):
        build_payload("score", "x", "y", {"a": "b"})
    with pytest.raises(ValueError):
        build_payload("choice", "x", "y", ["a", "b"])
    with pytest.raises(ValueError):
        build_payload("noul", "x", "y", ["a", "b"])


# --------------------------------------------------------------------------- #
# End-to-end against canned API responses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario,case", ALL_CASES)
def test_dataset_with_canned_api(scenario, case):
    session = FakeSession(CANNED[(scenario, case)])
    r = run_scenario_case(scenario, case, api_key="test-key", session=session)
    assert r.error is None
    assert r.latency_ms >= 0
    assert session.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert_expectation(scenario, case, r)
    assert not r.fallback, f"{r.confidence} should clear {CONFIDENCE_THRESHOLD}"


# --------------------------------------------------------------------------- #
# Safe Fallback Rule
# --------------------------------------------------------------------------- #
def test_confident_no_is_not_low_confidence():
    """P(emergency)=0.03 is a 97%-confident 'no' — must NOT trip fallback."""
    parsed = parse_answer("noul", {"noul": 0.03})
    assert parsed["confidence"] == pytest.approx(0.97)


@pytest.mark.parametrize("answer,primitive,criteria", [
    ({"noul": 0.55}, "noul", None),
    ({"choice": "property", "confidence": 0.61, "probabilities": {"property": 0.61, "general_liability": 0.39}}, "choice", {"property": "", "general_liability": ""}),
    ({"score": 1.6, "confidence": 0.52, "probabilities": {"0": 0.1, "1": 0.52, "2": 0.38, "3": 0.0}}, "score", ["a", "b", "c", "d"]),
])
def test_low_confidence_triggers_fallback(answer, primitive, criteria):
    r = evaluate(primitive, "text", "instr", criteria, api_key="k", session=FakeSession(wrap(answer)))
    assert r.confidence < CONFIDENCE_THRESHOLD
    assert r.fallback


def test_http_error_triggers_fallback():
    r = evaluate("noul", "t", "i", api_key="k", session=FakeSession({"error": "unauthorized"}, status=401))
    assert r.error == "HTTP 401" and r.fallback


def test_timeout_triggers_fallback():
    r = evaluate("noul", "t", "i", api_key="k", session=FakeSession(None, exc=requests.Timeout()))
    assert "Timed out" in r.error and r.fallback


def test_schema_drift_triggers_fallback():
    r = evaluate("noul", "t", "i", api_key="k", session=FakeSession({"unexpected": True}))
    assert r.error and r.fallback


def test_missing_key_triggers_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("BEATS_API_KEY", raising=False)
    monkeypatch.delenv("BEATAPI_API_KEY", raising=False)
    monkeypatch.setattr(jev_client, "SECRETS_FILE", tmp_path / "missing.toml")
    r = evaluate("noul", "t", "i")
    assert "BEATS_API_KEY" in r.error and r.fallback


def test_key_read_from_secrets_file(monkeypatch, tmp_path):
    monkeypatch.delenv("BEATS_API_KEY", raising=False)
    monkeypatch.delenv("BEATAPI_API_KEY", raising=False)
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('BEATS_API_KEY = "from-secrets"\n')
    monkeypatch.setattr(jev_client, "SECRETS_FILE", secrets)
    session = FakeSession(wrap({"type": "noul", "noul": 0.97}))
    evaluate("noul", "t", "i", session=session)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer from-secrets"


def test_env_var_overrides_secrets_file(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('BEATS_API_KEY = "from-secrets"\n')
    monkeypatch.setattr(jev_client, "SECRETS_FILE", secrets)
    monkeypatch.setenv("BEATS_API_KEY", "from-env")
    assert get_api_key() == "from-env"


def test_score_uses_argmax_when_confidence_missing():
    parsed = parse_answer("score", {"score": 2.4, "probabilities": {"0": 0, "1": 0.1, "2": 0.3, "3": 0.6}}, ["a", "b", "c", "d"])
    assert parsed["detail"]["index"] == 3 and parsed["confidence"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# Mock engine sanity (so the offline demo shows the expected story)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario,case", ALL_CASES)
def test_mock_engine_matches_dataset(scenario, case):
    r = run_scenario_case(scenario, case, use_mock=True)
    assert r.error is None and r.mock
    assert_expectation(scenario, case, r)


# --------------------------------------------------------------------------- #
# Live validation against the real endpoint (opt-in)
# --------------------------------------------------------------------------- #
live = pytest.mark.skipif(
    not (os.getenv("JEV_LIVE") and get_api_key()),
    reason="set JEV_LIVE=1 and BEATS_API_KEY to run live tests",
)

# BeatAPI free tier: an account that has never topped up gets 1 successful request/minute.
LIVE_DELAY_S = float(os.getenv("JEV_LIVE_DELAY", "61"))
_last_live_call = [0.0]


@pytest.mark.live
@live
@pytest.mark.parametrize("scenario,case", ALL_CASES)
def test_live_dataset(scenario, case):
    wait = _last_live_call[0] + LIVE_DELAY_S - time.monotonic()
    if _last_live_call[0] and wait > 0:
        time.sleep(wait)
    r = run_scenario_case(scenario, case)
    _last_live_call[0] = time.monotonic()
    print(f"\n{scenario}/{case}: {r.decision} | conf={r.confidence_pct} | {r.latency_ms:.0f}ms | fallback={r.fallback}")
    assert r.error is None, r.error
    assert_expectation(scenario, case, r)
