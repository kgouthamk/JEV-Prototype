# TypeSafe JEV · P&C Workflow Validator

Streamlit prototype for testing JEV's three primitives (Noul, Choice, Score) on P&C workflows. It shows confidence, latency, and a Safe Fallback rule side by side.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Put your BeatAPI key in .streamlit/secrets.toml (BEATS_API_KEY = "..."); unset = Mock mode
streamlit run app.py
```

## Test

```bash
pytest -q                                        # offline: 33 tests (payload contract, parsing, fallback, mock engine)
JEV_LIVE=1 BEATS_API_KEY=... pytest -q -m live -s   # sends the 6 dataset cases to BeatAPI (~5 min: free tier is 1 req/min)
```

## Files

| File | Purpose |
|---|---|
| `jev_client.py` | Payload builder, HTTP call and latency timer, response parser, mock engine, P&C scenario dataset. No UI code. |
| `app.py` | Split-screen Streamlit UI. |
| `test_jev.py` | Mock test block: canned-response E2E, fallback edge cases, opt-in live run. |

## Design decisions

- **Noul confidence = max(p, 1−p).** Noul returns only P(yes). A clear "no" (P=0.03) counts as a 97%-confident "no", so the low-emergency windshield case does not route to manual review. The raw P(yes) appears under the decision.
- **The fallback is fail-safe.** It routes to manual review when confidence is below the threshold (default 0.85, adjustable in the sidebar). It also routes there on HTTP errors, timeouts, a missing key, or an unexpected response schema.
- **Latency is the client-observed round trip** (network + inference), not server-side inference time.
- **Mock mode** is a keyword heuristic that returns JEV-shaped JSON so you can demo the UI without a key. The UI labels mock results clearly. Don't present them as model results.
- **Backend:** TypeSafe JEV via BeatAPI's free tier: `POST https://api.beatapi.io/v1/systemone`, model `jev-1.13-free`. An account that has never topped up gets 1 successful request per minute, and a 429 routes to the fallback with a rate-limit message.
- The endpoint (`BEATS_API_URL`), model, threshold, and timeout can all be set from the sidebar or environment.
