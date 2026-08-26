"""Streamlit digital twin — choropleth, cohort, what-if, SHAP, write-back loop.

Run from the project root:

    streamlit run src/retrofittrust/dashboard/app.py

The FastAPI service should already be running (uvicorn) so rank/explain/ledger
calls go through the integration backend rather than in-process shortcuts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import requests
import streamlit as st

from retrofittrust.config import DEMO_COHORT_LSOA_COUNT, SQLITE_PATH
from retrofittrust.dashboard.cohort import select_demo_cohort
from retrofittrust.dashboard.data_loader import (
    apply_epc_uplift,
    load_geometries,
    load_lsoa_dataset,
    merge_twin_state,
)
from retrofittrust.dashboard.plots import choropleth_priority, shap_waterfall
from retrofittrust.dashboard.state import db_mtime_token, init_twin_db
from retrofittrust.ledger.synthetic import SYNTHETIC_LABEL

DEFAULT_API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RetrofitTrust Birmingham",
    page_icon="🏘️",
    layout="wide",
)

init_twin_db(SQLITE_PATH)


def _api_url() -> str:
    return st.session_state.get("api_base", DEFAULT_API).rstrip("/")


def api_get(path: str, timeout: int = 15) -> dict:
    response = requests.get(f"{_api_url()}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict, timeout: int = 60) -> dict:
    response = requests.post(f"{_api_url()}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner="Loading merged LSOA dataset…")
def cached_dataset() -> tuple[pd.DataFrame, str]:
    frame, source = load_lsoa_dataset()
    return frame, source


@st.cache_data(show_spinner="Loading LSOA geometries…")
def cached_geometries(codes: tuple[str, ...]) -> tuple[dict, str, str]:
    return load_geometries(list(codes))


def live_frame(base: pd.DataFrame, _mtime: float) -> pd.DataFrame:
    """Re-read SQLite on every interaction; ``_mtime`` busts cache after write-back."""
    _ = _mtime
    return merge_twin_state(base)


def filter_cohort(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "imd_decile" in work.columns:
        imd = pd.to_numeric(work["imd_decile"], errors="coerce")
        lo, hi = st.sidebar.slider("IMD decile (1 = most deprived)", 1, 10, (1, 10))
        work = work[imd.between(lo, hi) | imd.isna()]
    if "priority_display" in work.columns:
        scores = pd.to_numeric(work["priority_display"], errors="coerce")
        finite = scores.dropna()
        if not finite.empty:
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmin < vmax:
                selected = st.sidebar.slider(
                    "Priority score range",
                    min_value=round(vmin, 3),
                    max_value=round(vmax, 3),
                    value=(round(vmin, 3), round(vmax, 3)),
                )
                work = work[scores.between(selected[0], selected[1]) | scores.isna()]
    if "epc_current" in work.columns:
        bands = sorted(work["epc_current"].dropna().astype(str).unique().tolist())
        picked = st.sidebar.multiselect("Current EPC band", bands, default=bands)
        if picked:
            work = work[work["epc_current"].astype(str).isin(picked)]
    status_opts = sorted(work["verification_status"].dropna().astype(str).unique().tolist())
    if status_opts:
        picked_status = st.sidebar.multiselect(
            "Twin status (write-back)",
            status_opts,
            default=status_opts,
        )
        if picked_status:
            work = work[work["verification_status"].isin(picked_status)]
    return work


def main() -> None:
    st.title("RetrofitTrust Birmingham")
    st.caption(
        "Digital twin PoC — AI ranking, SHA-256 ledger, SQLite write-back. "
        "Not a production system."
    )

    st.sidebar.header("Service")
    st.session_state.setdefault("api_base", DEFAULT_API)
    st.sidebar.text_input("FastAPI base URL", key="api_base")

    api_ok = False
    health: dict = {}
    try:
        health = api_get("/health", timeout=3)
        api_ok = health.get("status") == "ok"
        st.sidebar.success("API connected")
        st.sidebar.caption(
            f"Model loaded: {health.get('model_loaded')} · "
            f"Ledger: {health.get('ledger_message')}"
        )
    except requests.RequestException:
        st.sidebar.error("API not reachable — start uvicorn (see sidebar help).")
        with st.sidebar.expander("How to start the API"):
            st.code(
                "uvicorn retrofittrust.api.main:app --reload --app-dir src --port 8000",
                language="bash",
            )

    base, data_source = cached_dataset()
    mtime = db_mtime_token()
    display = live_frame(base, mtime)

    if data_source == "synthetic_fallback":
        st.warning(
            f"**{SYNTHETIC_LABEL}** — Program 1 merged dataset not found. "
            "Showing a labelled demo frame so the integration loop can still run. "
            "Replace with `data/processed/merged_lsoa.parquet` when the pipeline is ready."
        )

    st.sidebar.header("Cohort filters")
    all_codes = display["lsoa21cd"].astype(str).tolist()
    if "cohort_select" not in st.session_state:
        demo = [c for c in select_demo_cohort(DEMO_COHORT_LSOA_COUNT) if c in set(all_codes)]
        st.session_state["cohort_select"] = demo or all_codes[:DEMO_COHORT_LSOA_COUNT]

    if st.sidebar.button(f"Load demo cohort ({DEMO_COHORT_LSOA_COUNT} LSOAs)", type="primary"):
        demo = [c for c in select_demo_cohort(DEMO_COHORT_LSOA_COUNT) if c in set(all_codes)]
        st.session_state["cohort_select"] = demo or all_codes[:DEMO_COHORT_LSOA_COUNT]

    filtered = filter_cohort(display)
    cohort = st.sidebar.multiselect(
        "Selected LSOAs",
        options=all_codes,
        key="cohort_select",
    )
    cohort_df = display[display["lsoa21cd"].isin(cohort)].copy() if cohort else filtered.head(0)

    geojson, featureidkey, geo_source = cached_geometries(tuple(display["lsoa21cd"].astype(str)))
    if geo_source == "synthetic_grid":
        st.info(
            f"**{SYNTHETIC_LABEL}** map geometries — ONS 2021 LSOA BGC GeoJSON was not found "
            "under `data/external/`. Choropleth uses a Birmingham-centred demo grid."
        )

    map_col, table_col = st.columns((1.35, 1), gap="large")
    with map_col:
        st.subheader("LSOA retrofit priority")
        colour_mode = st.radio(
            "Map colour",
            ("priority_display", "verified"),
            format_func=lambda x: "Priority score (live twin)" if x == "priority_display" else "Verified vs candidate",
            horizontal=True,
        )
        map_df = filtered.copy() if not filtered.empty else display.copy()
        if colour_mode == "verified":
            map_df["verified_num"] = map_df["verified"].astype(int)
            fig = choropleth_priority(
                map_df,
                geojson,
                featureidkey=featureidkey,
                color_col="verified_num",
                selected=cohort,
            )
        else:
            fig = choropleth_priority(
                map_df,
                geojson,
                featureidkey=featureidkey,
                color_col="priority_display",
                selected=cohort,
            )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Data: `{data_source}` · Geometries: `{geo_source}` · SQLite mtime: `{mtime}`")

    with table_col:
        st.subheader("Cohort table")
        show_cols = [
            c
            for c in (
                "lsoa21cd",
                "lsoa21nm",
                "priority_display",
                "epc_current",
                "imd_decile",
                "verification_status",
                "anomaly_flag",
            )
            if c in cohort_df.columns
        ]
        st.dataframe(cohort_df[show_cols] if show_cols else cohort_df, use_container_width=True, hide_index=True)
        if (
            not cohort_df.empty
            and "verification_status" in cohort_df.columns
            and cohort_df["verification_status"].astype(str).str.contains("SYNTHETIC", na=False).any()
        ):
            st.markdown(f"**{SYNTHETIC_LABEL}** badge — grant/verification rows are simulated.")

    st.divider()
    rank_col, whatif_col = st.columns(2)

    with rank_col:
        st.subheader("1–2. Rank via AI service")
        st.caption("Calls `POST /rank` then keeps scores in SQLite for the write-back loop.")
        if st.button("Rank selected cohort", disabled=not (api_ok and cohort)):
            try:
                ranked = api_post("/rank", {"lsoa_codes": cohort})
                st.session_state["last_rank"] = ranked
                st.success(f"Ranked {len(ranked.get('items', []))} LSOAs · source={ranked.get('source')}")
                for note in ranked.get("notes") or []:
                    st.caption(note)
                st.dataframe(pd.DataFrame(ranked.get("items", [])), use_container_width=True, hide_index=True)
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"/rank failed: {exc}")
        elif not cohort:
            st.info("Select a cohort (or load the 10-LSOA demo) first.")

    with whatif_col:
        st.subheader("What-if EPC uplift")
        st.caption(
            "Modelled before/after only — does not write the ledger. "
            "Fuel-poverty indicator is an assumed IMD+EPC proxy, not the official BEIS statistic."
        )
        bands = st.slider("Assumed EPC band uplift", 1, 3, 2)
        if not cohort_df.empty:
            scenario = apply_epc_uplift(cohort_df, bands)
            before_need = pd.to_numeric(scenario["epc_current_need"], errors="coerce").mean()
            after_need = pd.to_numeric(scenario["epc_need_after"], errors="coerce").mean()
            fp_before = int(scenario["fp_proxy_before"].sum())
            fp_after = int(scenario["fp_proxy_after"].sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean EPC need (1=A … 7=G)", f"{before_need:.2f}", f"{after_need - before_need:.2f}")
            m2.metric("After modelled uplift", f"{after_need:.2f}")
            m3.metric("Fuel-poverty proxy count", f"{fp_before}", f"{fp_after - fp_before}")
        else:
            st.info("Select LSOAs to compare before/after aggregates.")

    st.divider()
    shap_col, ledger_col = st.columns(2)

    with shap_col:
        st.subheader("SHAP / local explanation")
        pick = st.selectbox(
            "LSOA or property-level aggregate to explain",
            options=cohort or display["lsoa21cd"].astype(str).tolist()[:DEMO_COHORT_LSOA_COUNT],
        )
        if st.button("Explain selection", disabled=not (api_ok and pick)):
            try:
                explained = api_post("/explain", {"lsoa21cd": pick, "top_n": 10})
                st.session_state["last_explain"] = explained
            except requests.RequestException as exc:
                st.error(f"/explain failed: {exc}")
                explained = None
        else:
            explained = st.session_state.get("last_explain")

        if explained:
            st.caption(explained.get("caveat", ""))
            st.caption(
                f"method={explained.get('method')} · "
                f"prediction={explained.get('prediction'):.4f} · "
                f"base={explained.get('base_value'):.4f}"
            )
            fig = shap_waterfall(
                explained.get("features") or [],
                base_value=float(explained.get("base_value") or 0.0),
                prediction=float(explained.get("prediction") or 0.0),
                title=f"Local explanation — {explained.get('lsoa21cd')}",
            )
            st.plotly_chart(fig, use_container_width=True)

    with ledger_col:
        st.subheader("3–5. Ledger + write-back")
        st.markdown(f"**{SYNTHETIC_LABEL}** — eligibility, works claimed, and verification are generated.")
        target = st.selectbox(
            "LSOA for ledger events",
            options=cohort or display["lsoa21cd"].astype(str).tolist()[:1],
            key="ledger_target",
        )
        score_lookup = 0.0
        if target and target in set(display["lsoa21cd"].astype(str)):
            score_lookup = float(
                pd.to_numeric(
                    display.loc[display["lsoa21cd"].astype(str) == target, "priority_display"],
                    errors="coerce",
                ).iloc[0]
                or 0.0
            )

        c1, c2, c3 = st.columns(3)
        if c1.button("Append eligibility", disabled=not (api_ok and target)):
            try:
                api_post(
                    "/ledger/append",
                    {
                        "event_type": "eligibility",
                        "lsoa21cd": target,
                        "generate_synthetic": True,
                        "priority_score": score_lookup,
                    },
                )
                st.rerun()
            except requests.RequestException as exc:
                st.error(str(exc))
        if c2.button("Append works claimed", disabled=not (api_ok and target)):
            try:
                api_post(
                    "/ledger/append",
                    {
                        "event_type": "works_claimed",
                        "lsoa21cd": target,
                        "generate_synthetic": True,
                    },
                )
                st.rerun()
            except requests.RequestException as exc:
                st.error(str(exc))
        if c3.button("Append verification", disabled=not (api_ok and target), type="primary"):
            try:
                api_post(
                    "/ledger/append",
                    {
                        "event_type": "verification",
                        "lsoa21cd": target,
                        "generate_synthetic": True,
                        "epc_uplift_bands": 2,
                    },
                )
                st.rerun()
            except requests.RequestException as exc:
                st.error(str(exc))

        if st.button("Verify chain", disabled=not api_ok):
            try:
                st.session_state["ledger_verify"] = api_get("/ledger/verify")
            except requests.RequestException as exc:
                st.error(str(exc))

        verified = st.session_state.get("ledger_verify")
        if verified:
            if verified.get("valid"):
                st.success(verified.get("message"))
            else:
                st.error(verified.get("message"))
            st.json(verified.get("recent_blocks") or [])

        if st.button("Tampering demo (in-memory copy)", disabled=not api_ok):
            try:
                demo = api_get("/ledger/tamper-demo")
                st.json(demo)
            except requests.RequestException as exc:
                st.error(str(exc))

    with st.expander("Limitations (dissertation caveats)"):
        st.markdown(
            """
- **Ecological fallacy:** IMD is LSOA-level and must not be read as household deprivation.
- **EPC performance gap:** modelled vs metered energy differs (~16% gas, ~31% electric); ranking is relative.
- **SHAP:** TreeExplainer can misattribute importance among correlated features.
- **Ledger:** hashlib SHA-256 hash-chain simulation — not a real blockchain.
- **Grants/works/verification:** labelled **SYNTHETIC DATA**.
            """
        )


main()
