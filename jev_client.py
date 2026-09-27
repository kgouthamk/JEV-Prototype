"""
JEV client + P&C scenario definitions for the TypeSafe JEV prototype.

Kept free of any UI code so it can be unit-tested and reused (Streamlit app,
batch evals, notebooks).

Talks to TypeSafe's JEV via BeatAPI (free tier model `jev-1.13-free`):
    POST https://api.beatapi.io/v1/systemone   (not OpenAI-compatible)
    Free tier: an account that has never topped up gets 1 successful request/minute.

Response shape handled (per BeatAPI's JEV guide, https://beatapi.io/blog/free-jev-api):
    {
      "id": "task_...",
      "model": "jev-1.13-free",
      "answers": {
        "triage_task": {"type": "noul",   "noul": 0.95}
        "triage_task": {"type": "choice", "choice": "billing", "confidence": 0.8,
                        "probabilities": {"billing": 0.87, ...}}
        "triage_task": {"type": "score",  "score": 1.04, "confidence": 0.94,
                        "legend": {"0": "...", ...}, "probabilities": {"0": 0.0, "1": 0.96, ...}}
      },
      "usage": {...}
    }
The parser is defensive: gateways sometimes re-wrap payloads, so it falls back
gracefully instead of crashing the UI.
"""

from __future__ import annotations

import os
import re
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

MODEL = "jev-1.13-free"
DEFAULT_ENDPOINT = os.getenv("BEATS_API_URL", "https://api.beatapi.io/v1/systemone")
API_KEY_ENV_VARS = ("BEATS_API_KEY", "BEATAPI_API_KEY")  # BeatAPI's docs use the second name
QUESTION_KEY = "triage_task"
CONFIDENCE_THRESHOLD = 0.85
FALLBACK_MESSAGE = "⚠️ Low Confidence Fallback: Routing to Manual Review Queue"


# --------------------------------------------------------------------------- #
# P&C scenario dataset
# --------------------------------------------------------------------------- #
SCENARIOS: dict[str, dict[str, Any]] = {
    "fnol_triage": {
        "label": "1 · FNOL Emergency Triage (Noul)",
        "type": "noul",
        "instructions": (
            "Does this text indicate an active emergency, severe bodily injury, "
            "or a non-driveable/severely damaged vehicle?"
        ),
        "criteria": {  # BeatAPI's noul example describes each side under "true"/"false"
            "true": "Active emergency, injury, or a vehicle that cannot be driven safely",
            "false": "Routine or cosmetic damage with no injury, safe for standard handling",
        },
        "cases": {
            "A · High emergency": {
                "text": (
                    "My delivery truck swerved to avoid a deer and hit a guardrail on I-95. "
                    "The front end is completely smashed, oil is leaking everywhere, and it "
                    "can't be driven. The driver is dizzy and being looked at by paramedics "
                    "right now."
                ),
                "expect": {"decision_is_yes": True},
            },
            "B · Low emergency": {
                "text": (
                    "I noticed a minor crack on my windshield this morning while parked in my "
                    "driveway. It looks like a pebble hit it on my way home yesterday. I just "
                    "want to schedule a glass repair when a technician is in the area."
                ),
                "expect": {"decision_is_yes": False},
            },
        },
    },
    "commercial_routing": {
        "label": "2 · Commercial Line Routing (Choice)",
        "type": "choice",
        "instructions": (
            "Which commercial insurance line of business queue should handle this "
            "broker submission?"
        ),
        "criteria": {
            "commercial_auto": "Fleet tracking, heavy trucks, transport liability, and company vehicles.",
            "general_liability": "Third-party bodily injury, property damage occurring on premises, or advertising injury.",
            "workers_comp": "Employee injuries, medical payouts for staff, workplace safety incidents.",
            "property": "Buildings, warehouses, business personal property, fire, and storm damage.",
        },
        "cases": {
            "A · Workers comp": {
                "text": (
                    "Submission for Apex Manufacturing. A warehouse technician slipped on a wet "
                    "surface near the assembly line yesterday, resulting in a fractured wrist. "
                    "Seeking a quote for mandatory state employee injury coverage for 45 "
                    "full-time staff members."
                ),
                "expect": {"choice": "workers_comp"},
            },
            "B · Commercial auto": {
                "text": (
                    "New business application for Flash Delivery Logistics. Requesting coverage "
                    "for a newly acquired fleet of 12 Ford Transit cargo vans used daily for "
                    "localized final-mile shipping operations."
                ),
                "expect": {"choice": "commercial_auto"},
            },
        },
    },
    "fraud_risk": {
        "label": "3 · Fraud Risk Level (Score)",
        "type": "score",
        "instructions": (
            "Rate the level of risk, inconsistency, or potential fraud flags present in "
            "this claims statement."
        ),
        "criteria": [
            "Completely consistent and low risk",
            "Minor variance but likely standard",
            "Ambiguous timeline requires verification",
            "High risk indicators present",
        ],
        "cases": {
            "A · Low risk": {
                "text": (
                    "Policyholder states storm winds knocked a tree branch onto their roof at "
                    "9:15 PM during yesterday's highly documented storm. Police reports and "
                    "local news confirm widespread power outages and falling trees at that "
                    "exact hour."
                ),
                "expect": {"index": 0},
            },
            "B · High risk": {
                "text": (
                    "Claimant is filing for a high-end luxury watch allegedly stolen from a gym "
                    "locker. The policy was purchased exactly 36 hours prior to the reported "
                    "theft. Claimant has failed to provide an original purchase receipt and has "
                    "a record of two similar off-premises theft claims with another carrier "
                    "over the last 18 months."
                ),
                "expect": {"index": 3},
            },
        },
    },
}


# --------------------------------------------------------------------------- #
# Result model
# --------------------------------------------------------------------------- #
@dataclass
class JevResult:
    primitive: str
    decision: str = ""
    confidence: float | None = None          # 0..1, the value the fallback rule uses
    latency_ms: float = 0.0                   # client-observed round trip
    probabilities: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    error: str | None = None
    mock: bool = False
    threshold: float = CONFIDENCE_THRESHOLD

    @property
    def fallback(self) -> bool:
        """Safe Fallback Rule: missing/errored or < threshold -> manual review."""
        return self.error is not None or self.confidence is None or self.confidence < self.threshold

    @property
    def confidence_pct(self) -> str:
        return "—" if self.confidence is None else f"{self.confidence * 100:.0f}%"


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
def build_payload(
    primitive: str,
    state: str,
    instructions: str,
    criteria: Any = None,
    model: str = MODEL,
) -> dict[str, Any]:
    question: dict[str, Any] = {"type": primitive, "instructions": instructions}
    if primitive in ("choice", "score"):
        if not criteria:
            raise ValueError(f"'{primitive}' requires criteria")
        if primitive == "score" and not isinstance(criteria, list):
            raise ValueError("'score' criteria must be an ordered list of strings")
        if primitive == "choice" and not isinstance(criteria, dict):
            raise ValueError("'choice' criteria must be an object of {option: description}")
        question["criteria"] = criteria
    elif criteria:  # noul: optional {"true": ..., "false": ...} descriptions
        if not isinstance(criteria, dict):
            raise ValueError("'noul' criteria must be an object of {\"true\": ..., \"false\": ...}")
        question["criteria"] = criteria
    return {"model": model, "state": state, "questions": {QUESTION_KEY: question}}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _extract_answer(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("Response is not a JSON object")
    for container in ("answers", "results", "data"):
        block = response.get(container)
        if isinstance(block, dict) and QUESTION_KEY in block:
            return block[QUESTION_KEY]
    if QUESTION_KEY in response:
        return response[QUESTION_KEY]
    raise ValueError(f"No '{QUESTION_KEY}' answer found in response")


def _floatify(d: Any) -> dict[str, float]:
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def parse_answer(primitive: str, answer: dict[str, Any], criteria: Any = None) -> dict[str, Any]:
    """Normalise one JEV answer into decision / confidence / probabilities / detail."""
    if primitive == "noul":
        p = answer.get("noul", answer.get("probability"))
        if p is None:
            raise ValueError("noul answer missing probability")
        p = float(p)
        yes = p >= 0.5
        # Noul has no separate confidence field; the probability IS the distribution.
        # Decision confidence = certainty in whichever side won. A confident "no"
        # (p=0.03) is 97% confident, not 3%.
        return {
            "decision": ("YES — Emergency: escalate FNOL" if yes else "NO — Standard FNOL handling"),
            "confidence": max(p, 1 - p),
            "probabilities": {"yes": p, "no": 1 - p},
            "detail": {"p_yes": p, "decision_is_yes": yes},
        }

    if primitive == "choice":
        probs = _floatify(answer.get("probabilities"))
        choice = answer.get("choice") or (max(probs, key=probs.get) if probs else None)
        if choice is None:
            raise ValueError("choice answer missing 'choice'")
        conf = answer.get("confidence")
        conf = float(conf) if conf is not None else probs.get(choice)
        return {
            "decision": f"Route to: {choice}",
            "confidence": conf,
            "probabilities": probs,
            "detail": {"choice": choice, "p_choice": probs.get(choice)},
        }

    if primitive == "score":
        probs = _floatify(answer.get("probabilities"))
        score = answer.get("score")
        if probs:
            idx = int(max(probs, key=probs.get))
        elif score is not None:
            idx = int(round(float(score)))
        else:
            raise ValueError("score answer missing 'score' and 'probabilities'")
        legend = answer.get("legend") or {}
        labels = list(criteria) if isinstance(criteria, list) else []
        if labels:
            idx = max(0, min(idx, len(labels) - 1))
        label = legend.get(str(idx)) or (labels[idx] if idx < len(labels) else f"Level {idx}")
        conf = answer.get("confidence")
        conf = float(conf) if conf is not None else (probs.get(str(idx)) if probs else None)
        return {
            "decision": f"Level {idx}: {label}",
            "confidence": conf,
            "probabilities": probs,
            "detail": {
                "index": idx,
                "continuous_score": float(score) if score is not None else None,
                "label": label,
            },
        }

    raise ValueError(f"Unknown primitive '{primitive}'")


# --------------------------------------------------------------------------- #
# Mock engine (demo without a key; clearly labelled in the UI)
# --------------------------------------------------------------------------- #
_NOUL_SIGNALS = r"paramedic|ambulance|injur|bleed|dizzy|unconscious|can'?t be driven|not driveable|non-driveable|smashed|totaled|fire|leaking|trapped|hospital"
_CHOICE_SIGNALS = {
    "commercial_auto": r"fleet|van|truck|vehicle|driver|shipping|delivery|transit|logistics",
    "general_liability": r"third[- ]party|customer|visitor|premises|advertis|slip.*customer|lawsuit",
    "workers_comp": r"employee|staff|technician|worker|workplace|fractur|injur|state .*coverage",
    "property": r"building|warehouse fire|roof|storm|fire|flood|business personal property",
}
_RISK_SIGNALS = r"allegedly|no receipt|failed to provide|prior to the reported|days? prior|hours? prior|similar .*claims|another carrier|luxury|cash|inconsistent"
_VERIFY_SIGNALS = r"unsure|approximately|can'?t recall|sometime|unclear|later remembered"


def _hits(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _normalise(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values()) or 1.0
    return {k: round(v / total, 4) for k, v in scores.items()}


def mock_response(primitive: str, state: str, criteria: Any = None) -> dict[str, Any]:
    """Keyword heuristic that mimics the JEV response shape. NOT a model."""
    if primitive == "noul":
        h = _hits(_NOUL_SIGNALS, state)
        p = min(0.98, 0.04 + 0.19 * h)
        answer = {"type": "noul", "noul": round(p, 4)}
    elif primitive == "choice":
        keys = list(criteria or _CHOICE_SIGNALS)
        raw = {k: 0.25 + 2.0 * _hits(_CHOICE_SIGNALS.get(k, r"$^"), state) ** 1.5 for k in keys}
        probs = _normalise(raw)
        top = max(probs, key=probs.get)
        answer = {"type": "choice", "choice": top, "confidence": probs[top], "probabilities": probs}
    else:
        levels = len(criteria or [0, 1, 2, 3])
        risk = _hits(_RISK_SIGNALS, state)
        verify = _hits(_VERIFY_SIGNALS, state)
        target = min(levels - 1, 3 if risk >= 3 else (2 if verify else (1 if risk else 0)))
        raw = {str(i): (6.0 if i == target else 0.12 / (1 + abs(i - target))) for i in range(levels)}
        probs = _normalise(raw)
        score = sum(int(k) * v for k, v in probs.items())
        answer = {
            "type": "score",
            "score": round(score, 3),
            "confidence": probs[str(target)],
            "legend": {str(i): c for i, c in enumerate(criteria or [])},
            "probabilities": probs,
        }
    return {"model": f"{MODEL} (MOCK)", "answers": {QUESTION_KEY: answer}, "usage": {}}


SECRETS_FILE = Path(__file__).parent / ".streamlit" / "secrets.toml"


def get_api_key() -> str:
    """Environment variable first, then .streamlit/secrets.toml (same names)."""
    for name in API_KEY_ENV_VARS:
        if os.getenv(name):
            return os.environ[name]
    try:
        secrets = tomllib.loads(SECRETS_FILE.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    return next((str(secrets[n]) for n in API_KEY_ENV_VARS if secrets.get(n)), "")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def evaluate(
    primitive: str,
    state: str,
    instructions: str,
    criteria: Any = None,
    *,
    api_key: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = MODEL,
    timeout: float = 15.0,
    use_mock: bool = False,
    threshold: float = CONFIDENCE_THRESHOLD,
    session: requests.Session | None = None,
) -> JevResult:
    result = JevResult(primitive=primitive, mock=use_mock, threshold=threshold)
    try:
        payload = build_payload(primitive, state, instructions, criteria, model)
    except ValueError as exc:
        result.error = f"Invalid request: {exc}"
        return result
    result.request = payload

    t0 = time.perf_counter()
    try:
        if use_mock:
            body = mock_response(primitive, state, criteria)
        else:
            key = api_key or get_api_key()
            if not key:
                raise RuntimeError("BEATS_API_KEY is not set (env var or .streamlit/secrets.toml)")
            http = session or requests
            resp = http.post(
                endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                timeout=timeout,
            )
            result.latency_ms = (time.perf_counter() - t0) * 1000
            try:
                body = resp.json()
            except ValueError:
                body = {"_non_json_body": resp.text[:2000]}
            if resp.status_code >= 400:
                result.raw = body
                result.error = f"HTTP {resp.status_code}"
                if resp.status_code == 429:
                    retry = resp.headers.get("Retry-After")
                    result.error += " rate limited (free tier: 1 request/min)" + (f", retry after {retry}s" if retry else "")
                return result
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.raw = body
        parsed = parse_answer(primitive, _extract_answer(body), criteria)
    except requests.Timeout:
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.error = f"Timed out after {timeout:.0f}s"
        return result
    except Exception as exc:  # network, auth, schema drift
        if not result.latency_ms:
            result.latency_ms = (time.perf_counter() - t0) * 1000
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.decision = parsed["decision"]
    result.confidence = parsed["confidence"]
    result.probabilities = parsed["probabilities"]
    result.detail = parsed["detail"]
    return result


def run_scenario_case(scenario_key: str, case_key: str, **kwargs) -> JevResult:
    sc = SCENARIOS[scenario_key]
    return evaluate(sc["type"], sc["cases"][case_key]["text"], sc["instructions"], sc["criteria"], **kwargs)
