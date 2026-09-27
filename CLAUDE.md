# CLAUDE.md — JEV P&C Prototype

## Status (as of 2026-09-27)
- **Built:** a Streamlit prototype that validates TypeSafe AI's JEV model (`jev-1.13-free`) on three P&C insurance workflows. Stakeholders see a confidence badge and meter, API latency, and a Safe Fallback badge.
- **Tested offline:** 33 offline pytest pass, using canned responses and a mock engine. A headless Streamlit `AppTest` run of all 6 dataset cases gave the expected answers, and the fallback badge rendered on a low-confidence custom input.
- **Endpoint fixed on 2026-09-27** to BeatAPI's documented `https://api.beatapi.io/v1/systemone`. The old bare domain was wrong. An unauthenticated POST returns 401, so the route exists. **Not yet run with a real key.**

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# key: BEATS_API_KEY in .streamlit/secrets.toml (gitignored) or env var; none = Mock mode
streamlit run app.py
pytest -q                            # offline suite
JEV_LIVE=1 pytest -q -m live -s      # sends the 6 dataset cases to the real endpoint, prints decision/conf/latency
```

## Files
- `jev_client.py`: all logic, no UI code. It contains:
  - `SCENARIOS`, the dataset: 3 scenarios × 2 cases, each with an `expect` value
  - `build_payload`, `evaluate` (HTTP call + latency timer), `parse_answer`, `_extract_answer`
  - `mock_response`, a keyword heuristic that is NOT a model
  - `JevResult`, whose `.fallback` property implements the Safe Fallback Rule
- `app.py`: split-screen Streamlit UI.
  - Left: scenario radio, test-case select plus "Custom", editable state text, question config (instructions and criteria JSON) in a collapsed expander, and the Run button.
  - Right: confidence % with a colored HTML meter (a tick marks the threshold), latency in ms, fallback/auto-route badge, decision, probability bar chart, and Response/Request JSON tabs.
  - Sidebar: API key, endpoint, model, mock toggle, threshold slider, timeout.
- `test_jev.py`: `FakeSession` canned responses, fallback edge cases (low confidence, HTTP 401, timeout, schema drift, missing key), mock-engine checks, and the opt-in `live` marker.

## API contract (confirmed from BeatAPI docs, 2026-09-27)
Source: https://beatapi.io/blog/free-jev-api and the `model:jev-1.13-free` entry in https://beatapi.io/openapi.json.
POST `https://api.beatapi.io/v1/systemone` (override with `BEATS_API_URL`). It is not OpenAI-compatible. Headers: `Authorization: Bearer <key>` and `Content-Type: application/json`. The key comes from `get_api_key()`: env var `BEATS_API_KEY`/`BEATAPI_API_KEY` first, then the same names in `.streamlit/secrets.toml` (gitignored; template in `secrets.toml.example`).
```json
{"model": "jev-1.13-free", "state": "<text or object>",
 "questions": {"triage_task": {"type": "noul|choice|score", "instructions": "...", "criteria": {...} | [...]}}}
```
Noul takes optional `{"true": ..., "false": ...}` criteria, and the FNOL scenario sends them. Choice takes a `{option: description}` object. Score takes an ordered array of strings.

Response: `{"id", "model", "answers": {"triage_task": {...}}, "usage"}`, where the answer is `{"type":"noul","noul":p}` or `{"type":"choice","choice","probabilities","confidence"}`. The score answer shape is not documented by BeatAPI; the assumed `{score, confidence, legend, probabilities}` still needs a live check.

**Free tier:** an account that has never topped up gets 1 successful request per minute, with 32k tokens of context. A 429 shows up in the UI as a rate-limit error. The live tests pace themselves at 61s per call (`JEV_LIVE_DELAY` overrides this), so a full run takes about 5 minutes.

**Still to check live:** the score response shape, and whether a string `state` works on the free tier (BeatAPI's HF guide shows it working).

## Design decisions (keep unless the user says otherwise)
- **Noul confidence = max(p, 1−p).** Otherwise a confident "no" (the windshield case, P(yes)≈0.03) would show 3% and trip the fallback. The UI still shows the raw P(yes).
- **Choice/Score confidence** uses the API's `confidence` field. If that's missing, it falls back to the winning class probability. Score's index is the argmax of `probabilities`, with the rounded continuous `score` as a fallback.
- **The fallback is fail-safe:** `error OR confidence is None OR confidence < threshold` (default 0.85). The badge text must stay exactly: `⚠️ Low Confidence Fallback: Routing to Manual Review Queue`.
- **Latency** is the client-observed round trip measured with `time.perf_counter`, not server inference time. Keep the label honest.
- **Mock mode** must stay clearly labeled in the UI. Never present its numbers as JEV results.

## Expected results for the dataset
| Scenario | Case A | Case B |
|---|---|---|
| FNOL triage (noul) | yes, emergency | no |
| Commercial routing (choice) | `workers_comp` | `commercial_auto` |
| Fraud risk (score) | index 0 | index 3 |

## Owner context
Goutham is a Principal PM for AI Strategy at Guidewire. The prototype is for stakeholder validation of JEV's speed and calibration in regulated P&C workflows. He prefers BLUF, concise, factual communication.
