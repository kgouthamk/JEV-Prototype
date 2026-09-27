"""
TypeSafe JEV · P&C Workflow Validator (Streamlit)

Run:
    export BEATS_API_KEY=...        # optional; toggle Mock mode without it
    streamlit run app.py
"""

from __future__ import annotations

import html
import json

import pandas as pd

import streamlit as st

from jev_client import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_ENDPOINT,
    FALLBACK_MESSAGE,
    MODEL,
    get_api_key,
    SCENARIOS,
    JevResult,
    evaluate,
)

st.set_page_config(page_title="JEV P&C Validator", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
.conf-big {font-size: 3rem; font-weight: 800; line-height: 1; margin: 0;}
/* Theme-neutral: tinted backgrounds + inherited text so it reads in light and dark mode. */
.conf-sub {color: #8b949e; font-size: 0.85rem; margin-top: 2px;}
.meter {width: 100%; height: 18px; background: rgba(128,128,128,0.25); border-radius: 9px; position: relative; overflow: hidden; margin: 10px 0 4px;}
.meter-fill {height: 100%; border-radius: 9px; transition: width .4s ease;}
.meter-tick {position: absolute; top: 0; bottom: 0; width: 3px; background: #111827; box-shadow: 0 0 0 1px #ffffff;}
.badge {padding: 14px 18px; border-radius: 10px; font-weight: 700; font-size: 1.05rem; margin: 8px 0; color: inherit;}
.badge-warn {background: rgba(220,38,38,0.12); border: 2px solid #dc2626;}
.badge-ok {background: rgba(22,163,74,0.12); border: 2px solid #16a34a;}
.decision {font-size: 1.35rem; font-weight: 700; padding: 10px 14px; color: inherit; background: rgba(37,99,235,0.10); border-left: 5px solid #2563eb; border-radius: 6px;}
.check {font-size: 0.9rem; margin-top: 6px;}
.latency {font-size: 2rem; font-weight: 800; font-family: ui-monospace, monospace;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Sidebar — connection settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Connection")
    env_key = get_api_key()
    api_key = st.text_input("BEATS_API_KEY", value=env_key, type="password",
                            help="Read from BEATS_API_KEY (env var or .streamlit/secrets.toml) if set. Get a key at beatapi.io.")
    endpoint = st.text_input("Endpoint", value=DEFAULT_ENDPOINT)
    model = st.text_input("Model", value=MODEL)
    use_mock = st.toggle("Mock mode (no API call)", value=not bool(env_key),
                         help="Keyword heuristic that mimics JEV's response shape. For UI demos only — not model output.")
    threshold = st.slider("Fallback threshold", 0.50, 0.99, CONFIDENCE_THRESHOLD, 0.01)
    timeout = st.number_input("Timeout (s)", 1.0, 60.0, 15.0, 1.0)
    st.caption("Latency is client-observed round trip (network + inference).")

st.title("🛡️ TypeSafe JEV · P&C Workflow Validator")
st.caption(f"Model `{model}` · Safe Fallback below **{threshold:.0%}** confidence routes to manual review.")

left, right = st.columns([1, 1.15], gap="large")


# --------------------------------------------------------------------------- #
# Left — inputs
# --------------------------------------------------------------------------- #
with left:
    st.subheader("Input")
    scenario_key = st.radio(
        "Scenario", list(SCENARIOS), format_func=lambda k: SCENARIOS[k]["label"],
    )
    sc = SCENARIOS[scenario_key]

    case_key = st.selectbox("Test case", list(sc["cases"]) + ["Custom"])
    default_text = sc["cases"].get(case_key, {}).get("text", "")
    state = st.text_area("Claim / submission text (state)", value=default_text, height=190,
                         key=f"state::{scenario_key}::{case_key}")

    with st.expander(f"Question config · primitive `{sc['type']}`", expanded=False):
        instructions = st.text_area("Instructions", value=sc["instructions"], height=90,
                                    key=f"instr::{scenario_key}")
        criteria = sc["criteria"]
        if criteria is not None:
            crit_text = st.text_area("Criteria (JSON)", value=json.dumps(criteria, indent=2),
                                     height=200, key=f"crit::{scenario_key}")
            try:
                criteria = json.loads(crit_text)
            except json.JSONDecodeError as exc:
                st.error(f"Criteria JSON invalid: {exc}")
                criteria = None
        else:
            st.caption("Noul takes no criteria — returns P(yes) for the instruction.")

    expect = sc["cases"].get(case_key, {}).get("expect")
    if expect:
        st.caption(f"Expected: `{expect}`")

    run = st.button("🚀 Run JEV Engine", type="primary", use_container_width=True,
                    disabled=not state.strip())

    if run:
        with st.spinner("Calling JEV…"):
            st.session_state["result"] = evaluate(
                sc["type"], state, instructions, criteria,
                api_key=api_key or None, endpoint=endpoint, model=model,
                timeout=timeout, use_mock=use_mock, threshold=threshold,
            )
            st.session_state["result_run"] = (scenario_key, case_key, state)


# --------------------------------------------------------------------------- #
# Right — outputs
# --------------------------------------------------------------------------- #
def _color(conf: float | None, thr: float) -> str:
    if conf is None:
        return "#9ca3af"
    if conf >= max(thr, 0.95):
        return "#16a34a"
    if conf >= thr:
        return "#65a30d"
    if conf >= thr - 0.15:
        return "#f59e0b"
    return "#dc2626"


def render(result: JevResult, expect: dict | None = None) -> None:
    if result.mock:
        st.info("MOCK MODE — heuristic output, not JEV. Latency is not representative.")

    m1, m2 = st.columns([1.3, 1])
    with m1:
        conf = result.confidence
        color = _color(conf, result.threshold)
        width = 0 if conf is None else conf * 100
        st.markdown(
            f"""
            <p class="conf-big" style="color:{color}">{result.confidence_pct} Confidence</p>
            <div class="meter">
              <div class="meter-fill" style="width:{width:.1f}%; background:{color};"></div>
              <div class="meter-tick" style="left:{result.threshold*100:.1f}%;" title="fallback threshold"></div>
            </div>
            <p class="conf-sub">Tick = {result.threshold:.0%} fallback threshold</p>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown("**API Latency**")
        st.markdown(f'<div class="latency">{result.latency_ms:,.0f} ms</div>', unsafe_allow_html=True)
        st.caption(f"primitive: `{result.primitive}`")

    if result.fallback:
        err = result.error or ""
        short = err if len(err) <= 90 else err.split(":", 1)[0]  # e.g. "ConnectionError"
        reason = f" — {html.escape(short)}" if short else ""
        st.markdown(f'<div class="badge badge-warn">{FALLBACK_MESSAGE}{reason}</div>',
                    unsafe_allow_html=True)
        if short != err:
            with st.expander("Error details"):
                st.code(err, language=None)
    else:
        st.markdown('<div class="badge badge-ok">✅ Auto-route: confidence above threshold</div>',
                    unsafe_allow_html=True)

    st.markdown("**Answer / Decision**")
    st.markdown(f'<div class="decision">{html.escape(result.decision or "—")}</div>', unsafe_allow_html=True)
    if expect and not result.error:
        misses = [k for k, v in expect.items() if result.detail.get(k) != v]
        st.markdown(
            '<p class="check">❌ Does not match expected: ' + html.escape(", ".join(
                f"{k}={result.detail.get(k)!r} (expected {expect[k]!r})" for k in misses)) + "</p>"
            if misses else '<p class="check">✔️ Matches expected answer for this test case</p>',
            unsafe_allow_html=True,
        )

    if result.primitive == "noul" and "p_yes" in result.detail:
        st.caption(f"P(yes) = {result.detail['p_yes']:.3f} · confidence = certainty in the winning side")
    if result.primitive == "score" and result.detail.get("continuous_score") is not None:
        st.caption(f"Continuous score = {result.detail['continuous_score']:.2f} · top index = {result.detail['index']}")

    if result.probabilities:
        st.markdown("**Probability distribution**")
        probs = pd.DataFrame(
            {"probability": sorted(result.probabilities.values(), reverse=True)},
            index=sorted(result.probabilities, key=result.probabilities.get, reverse=True),
        )
        st.bar_chart(probs, horizontal=True, height=40 + 32 * len(probs))

    tab_resp, tab_req = st.tabs(["Response JSON", "Request JSON"])
    with tab_resp:
        st.code(json.dumps(result.raw, indent=2) if result.raw is not None else "null", language="json")
    with tab_req:
        st.code(json.dumps(result.request, indent=2), language="json")


with right:
    st.subheader("Output")
    res: JevResult | None = st.session_state.get("result")
    if res is None:
        st.caption("Pick a scenario and test case, then run the engine.")
    else:
        run_scenario, run_case, run_state = st.session_state.get("result_run", (None, None, None))
        current = (run_scenario, run_case) == (scenario_key, case_key)
        if not current or run_state != state:
            st.warning("Inputs changed since this run. Results below are from the previous run; press Run to refresh.")
        render(res, expect if current and run_state == state else None)
