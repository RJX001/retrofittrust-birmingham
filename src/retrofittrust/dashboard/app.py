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
    build_lsoa_detail,
    load_geometries,
    load_lsoa_dataset,
    merge_twin_state,
    whatif_budget_projection,
)
from retrofittrust.dashboard.plots import bar_priority_fallback, choropleth_priority, shap_waterfall
from retrofittrust.dashboard.state import db_mtime_token, fetch_all_lsoa_state, init_twin_db
from retrofittrust.ledger.synthetic import SYNTHETIC_LABEL

DEFAULT_API = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="RetrofitTrust Birmingham",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_twin_db(SQLITE_PATH)


def table_column_config() -> dict:
    return {
        "lsoa21cd": st.column_config.TextColumn(
            "LSOA code",
            help="Official 2021 neighbourhood code (Lower Super Output Area).",
        ),
        "lsoa21nm": st.column_config.TextColumn("Neighbourhood name"),
        "priority_display": st.column_config.NumberColumn(
            "Priority score",
            help="Higher = more urgent for retrofit. Live twin score after any write-back.",
            format="%.3f",
        ),
        "epc_current": st.column_config.TextColumn(
            "Current EPC",
            help="Typical Energy Performance Certificate band for homes in this neighbourhood (A best, G worst).",
        ),
        "imd_decile": st.column_config.NumberColumn(
            "IMD decile",
            help="English Index of Multiple Deprivation. 1 = most deprived 10% of England; 10 = least deprived.",
        ),
        "verification_status": st.column_config.TextColumn(
            "Twin status",
            help="Candidate until a verification event is written back; then marked verified.",
        ),
        "anomaly_flag": st.column_config.TextColumn(
            "Data-quality flag",
            help="Flagged if the anomaly screen found implausible EPC-style values. Flagged rows are not deleted.",
        ),
    }


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root, .stApp, [data-testid="stAppViewContainer"] {
            --primary-color: #1565C0 !important;
            --text-color: #111111 !important;
            --background-color: #FFFFFF !important;
            --secondary-background-color: #F5F7FA !important;
        }
        html, body, [data-testid="stAppViewContainer"], .stApp, [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
            color: #111111 !important;
            color-scheme: light !important;
        }
        .block-container { padding-top: 1.35rem; padding-bottom: 2.4rem; color: #111111 !important; }
        .block-container p,
        .block-container label,
        .block-container span,
        .block-container small,
        .block-container [data-testid="stWidgetLabel"] p,
        .block-container [data-testid="stCaption"] p,
        .block-container [data-testid="stMarkdown"] p {
            color: #111111 !important;
        }
        .block-container input,
        .block-container textarea,
        .block-container [data-baseweb="input"],
        .block-container [data-baseweb="input"] input {
            background-color: #FFFFFF !important;
            color: #111111 !important;
            border: 1px solid #1565C0 !important;
            caret-color: #111111 !important;
        }
        .block-container [data-testid="stAlert"] p,
        .block-container [data-testid="stNotification"] p {
            color: #111111 !important;
        }
        h1 { font-weight: 650; letter-spacing: -0.025em; margin-bottom: 0.15rem; color: #111111; }
        h2, h3 { letter-spacing: -0.015em; color: #111111; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; color: #111111 !important; }
        [data-testid="stMetricLabel"] { font-size: 0.86rem; color: #111111 !important; }

        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background-color: #111111 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] [data-testid="baseButton-secondary"],
        [data-testid="stSidebar"] [data-testid="baseButton-primary"] {
            background-color: #1565C0 !important;
            background-image: none !important;
            color: #FFFFFF !important;
            border: 0 !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
            background-color: #1565C0 !important;
            background-image: none !important;
            border: 2px solid #FFFFFF !important;
        }
        [data-testid="stSidebar"] [data-testid="stThumbValue"],
        [data-testid="stSidebar"] [data-testid="stThumbValue"] * {
            color: #90CAF9 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"],
        [data-testid="stSidebar"] span[data-baseweb="tag"] {
            background-color: #1565C0 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            background-color: #1A1A1A !important;
            color: #FFFFFF !important;
            border: 1px solid #1565C0 !important;
        }

        div[data-testid="stTabs"] button { font-weight: 560; color: #111111; }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #1565C0 !important;
            border-bottom-color: #1565C0 !important;
        }
        .rt-lead { color: #333333; font-size: 1.02rem; line-height: 1.45; margin-bottom: 0.6rem; }
        .rt-muted { color: #333333; font-size: 0.92rem; line-height: 1.45; margin-bottom: 0.75rem; }
        [data-testid="stAlert"] {
            background: #F5F7FA !important;
            color: #111111 !important;
            border: 1px solid #1565C0 !important;
        }
        .stButton > button {
            border: 1px solid #1565C0 !important;
            color: #1565C0 !important;
            background: #FFFFFF !important;
        }
        [data-testid="baseButton-primary"],
        .stButton > button[kind="primary"] {
            background: #1565C0 !important;
            color: #FFFFFF !important;
            border: 1px solid #0D47A1 !important;
        }
        * { accent-color: #1565C0 !important; }
        div[data-testid="stRadio"] p { color: #111111 !important; }
        div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
            border-color: #1565C0 !important;
            background-color: #FFFFFF !important;
        }
        div[data-testid="stRadio"] [aria-checked="true"] [data-baseweb="radio"] > div:first-child {
            background-color: #1565C0 !important;
            border-color: #0D47A1 !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background-color: #1565C0 !important;
            border-color: #1565C0 !important;
        }
        [data-testid="stThumbValue"],
        [data-testid="stThumbValue"] p,
        [data-testid="stTickBarMin"],
        [data-testid="stTickBarMax"] {
            color: #1565C0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stThumbValue"],
        [data-testid="stSidebar"] [data-testid="stThumbValue"] p,
        [data-testid="stSidebar"] [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] [data-testid="stTickBarMax"] {
            color: #90CAF9 !important;
        }
        /* Streamlit default red (#FF4B4B) → blue */
        [style*="255, 75, 75"],
        [style*="255,75,75"] {
            color: #1565C0 !important;
        }
        [style*="background-color: rgb(255, 75, 75)"],
        [style*="background-color:rgb(255, 75, 75)"],
        [style*="background: rgb(255, 75, 75)"] {
            background: #1565C0 !important;
            background-color: #1565C0 !important;
        }
        [data-testid="stSidebar"] [style*="255, 75, 75"] {
            color: #90CAF9 !important;
        }
        span[data-baseweb="tag"],
        [data-baseweb="tag"] {
            background-color: #1565C0 !important;
            color: #FFFFFF !important;
        }
        div[data-testid="column"] { overflow: visible !important; }
        div[data-testid="stPlotlyChart"] {
            border: 2px solid #1565C0 !important;
            border-radius: 8px;
            padding: 0.15rem;
            margin-left: 4px !important;
            margin-right: 4px !important;
            background: #FFFFFF !important;
            box-sizing: border-box !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


@st.cache_data(show_spinner="Loading neighbourhood dataset…")
def cached_dataset() -> tuple[pd.DataFrame, str]:
    frame, source = load_lsoa_dataset()
    return frame, source


@st.cache_data(show_spinner="Loading map geometries…")
def cached_geometries(codes: tuple[str, ...], _version: str = "borders-v2") -> tuple[dict, str, str]:
    return load_geometries(list(codes))


def live_frame(base: pd.DataFrame, _mtime: float) -> pd.DataFrame:
    """Re-read SQLite on every interaction; ``_mtime`` busts cache after write-back."""
    _ = _mtime
    return merge_twin_state(base)


def local_explain(lsoa_code: str, base: pd.DataFrame) -> dict | None:
    """Fallback when FastAPI is unreachable — uses in-process composite/SHAP helper."""
    try:
        from retrofittrust.api.features import load_model_bundle, shap_for_row

        row = base[base["lsoa21cd"].astype(str) == str(lsoa_code)]
        if row.empty:
            return None
        bundle = load_model_bundle()
        result = shap_for_row(row.head(1), bundle, top_n=10)
        result["lsoa21cd"] = lsoa_code
        result["source"] = "local_fallback"
        return result
    except Exception as exc:  # noqa: BLE001 — PoC graceful degradation
        return {"error": str(exc), "lsoa21cd": lsoa_code}


def section_intro(title: str, description: str) -> None:
    st.subheader(title)
    st.markdown(f'<p class="rt-muted">{description}</p>', unsafe_allow_html=True)


def render_glossary() -> None:
    st.markdown(
        """
**Neighbourhood (LSOA)** — Lower Super Output Area: a small statistical area of roughly
1,000–3,000 people. All maps and scores in this PoC are at LSOA grain, not individual homes.

**EPC** — Energy Performance Certificate. Bands run **A** (most efficient) to **G** (least).
The *gap* is how many bands a typical home could improve (current vs potential).

**IMD decile** — English Index of Multiple Deprivation. **1 = most deprived 10%** of England;
**10 = least deprived**. This is an *area* statistic, not a household one.

**Priority score** — Higher means more urgent for retrofit. It blends poor energy efficiency
(EPC gap) with income deprivation. After a grant is verified, the twin **halves** this score
so the neighbourhood drops down the queue.

**Digital twin** — This dashboard: a live picture of the housing stock that updates when
verification is written back to SQLite.

**SHAP** — Explains *why* a neighbourhood scored highly (which features pushed the score
up or down). It is not a causal proof.

**Ledger** — A SHA-256 hash-chain of grant events. **Not a real blockchain.** Grant amounts
and inspections are labelled **SYNTHETIC DATA**.

**Data-quality flag** — The anomaly screen (AutoEncoder + Isolation Forest) flagged
implausible values. Flagged rows are kept and down-weighted, never silently deleted.
        """
    )


def render_lsoa_detail(row: pd.Series, twin_state: dict[str, dict]) -> None:
    """Per-LSOA energy profile, priority, twin/ledger status."""
    code = str(row.get("lsoa21cd", ""))
    detail = build_lsoa_detail(row, twin_state.get(code))
    st.markdown(f"**{detail['lsoa21nm'] or code}**  ·  `{code}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Priority (live twin)",
        f"{detail['priority_display'] or detail['priority_score'] or 0:.3f}",
        help="Higher = more urgent. Falls after verification write-back.",
    )
    m2.metric(
        "EPC now → potential",
        f"{detail['epc_current']} → {detail['epc_potential']}",
        help="Typical current certificate band versus modelled potential after improvement.",
    )
    m3.metric(
        "IMD decile",
        detail["imd_decile"] if detail["imd_decile"] is not None else "—",
        help="1 = most deprived 10% of England; 10 = least deprived.",
    )
    m4.metric(
        "Twin status",
        detail["verification_status"],
        help="Candidate until verification is recorded; then verified.",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Energy and deprivation profile**")
        st.caption("What the ranking model can see for this neighbourhood.")
        profile = {
            "EPC gap (bands of improvement available)": detail["epc_gap"],
            "Current need (1 = A … 7 = G)": detail["epc_current_need"],
            "Properties represented in this LSOA": detail["n_properties"],
            "Data-quality (anomaly) flag": detail["anomaly_flag"],
            "Verified in the twin?": detail["verified"],
            "Modelled EPC uplift after write-back (bands)": detail["epc_uplift_bands"],
        }
        st.dataframe(
            pd.DataFrame({"What this is": profile.keys(), "Value": profile.values()}),
            use_container_width=True,
            hide_index=True,
        )
    with c2:
        st.markdown("**Grant trail (ledger stub)**")
        st.caption("Latest synthetic eligibility / works / verification recorded for this LSOA.")
        if detail["is_synthetic"] or "SYNTHETIC" in str(detail["verification_status"]):
            st.caption(f"**{SYNTHETIC_LABEL}** — simulated grant and verification events.")
        ledger_rows = {
            "Latest ledger event": detail["ledger_event"] or "—",
            "Twin database last updated": detail["ledger_updated"] or "—",
            "EPC band before verification": detail["outcome_epc_before"] or "—",
            "EPC band after verification": detail["outcome_epc_after"] or "—",
            "Outcome recorded at": detail["outcome_recorded_at"] or "—",
        }
        st.dataframe(
            pd.DataFrame({"What this is": ledger_rows.keys(), "Value": ledger_rows.values()}),
            use_container_width=True,
            hide_index=True,
        )

    gap = detail["epc_gap"]
    imd = detail["imd_decile"]
    st.caption(
        "Simple composite weights behind the fallback score: "
        f"EPC gap = {gap if gap is not None else '—'}, "
        f"IMD decile = {imd if imd is not None else '—'}. "
        "Open **Rank & Explain** for a SHAP breakdown of the trained model (when loaded)."
    )


def filter_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Map filters in the left sidebar (not the ranking cohort)."""
    work = df.copy()
    st.sidebar.caption("These filters change the **map**. The neighbourhood list is what Rank, Explain, and Ledger use.")
    if "imd_decile" in work.columns:
        imd = pd.to_numeric(work["imd_decile"], errors="coerce")
        lo, hi = st.sidebar.slider(
            "IMD decile",
            1,
            10,
            (1, 10),
            help="1 = most deprived 10% of England. Narrow this to focus the map on poorer areas.",
        )
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
                    help="Higher scores are more urgent. Drag to hide low- or high-priority areas on the map.",
                )
                work = work[scores.between(selected[0], selected[1]) | scores.isna()]
    if "epc_current" in work.columns:
        bands = sorted(work["epc_current"].dropna().astype(str).unique().tolist())
        picked = st.sidebar.multiselect(
            "Current EPC band",
            bands,
            default=bands,
            help="Keep only neighbourhoods whose typical current certificate is in these bands.",
        )
        if picked:
            work = work[work["epc_current"].astype(str).isin(picked)]
    status_opts = sorted(work["verification_status"].dropna().astype(str).unique().tolist())
    if status_opts:
        picked_status = st.sidebar.multiselect(
            "Twin status",
            status_opts,
            default=status_opts,
            help="Candidate = not yet verified. Verified = grant loop has written back.",
        )
        if picked_status:
            work = work[work["verification_status"].isin(picked_status)]
    return work


def render_map_tab(
    display: pd.DataFrame,
    filtered: pd.DataFrame,
    cohort: list[str],
    cohort_df: pd.DataFrame,
    geojson: dict,
    featureidkey: str,
    geo_source: str,
    data_source: str,
    mtime: float,
) -> None:
    st.subheader("Where Retrofit Need Is Highest")
    st.caption(
        "Darker blue = higher priority. Black outline = selected cohort. "
        "Switch colour to grant status after verification."
    )
    colour_mode = st.radio(
        "What should the colours mean?",
        ("priority_display", "verified"),
        format_func=lambda x: (
            "Priority score — darker blue = more urgent"
            if x == "priority_display"
            else "Grant status — verified versus still a candidate"
        ),
        horizontal=True,
        help="Switch between ranking intensity and the write-back result.",
        label_visibility="collapsed",
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

    map_col, table_col = st.columns((1.2, 1), gap="medium")
    chart_config = {"displayModeBar": False}
    with map_col:
        st.plotly_chart(fig, use_container_width=True, config=chart_config)
    with table_col:
        st.markdown("**Selected Cohort**")
        show_cols = [
            c
            for c in (
                "lsoa21cd",
                "lsoa21nm",
                "priority_display",
                "epc_current",
                "imd_decile",
                "verification_status",
            )
            if c in cohort_df.columns
        ]
        st.dataframe(
            cohort_df[show_cols] if show_cols else cohort_df,
            use_container_width=True,
            hide_index=True,
            column_config=table_column_config(),
            height=300,
        )
        if cohort_df.empty:
            st.info("Load the demo cohort or pick neighbourhoods in the sidebar.")

    if geo_source in ("synthetic_grid",):
        bar_df = filtered.copy() if not filtered.empty else display.copy()
        bar_col = (
            "verified_num"
            if colour_mode == "verified" and "verified_num" in bar_df.columns
            else "priority_display"
        )
        st.plotly_chart(
            bar_priority_fallback(bar_df, color_col=bar_col, selected=cohort, top_n=10),
            use_container_width=True,
            config=chart_config,
        )
    st.caption(f"Dataset: `{data_source}` · Map geometry: `{geo_source}`")


def render_rank_tab(api_ok: bool, cohort: list[str], display: pd.DataFrame, base: pd.DataFrame) -> None:
    rank_col, shap_col = st.columns(2, gap="large")
    with rank_col:
        section_intro(
            "Rank this cohort",
            "Asks the FastAPI service to score the selected neighbourhoods. "
            "Scores are cached in the twin database so the map can update. "
            "If no LightGBM model is on disk, a labelled composite fallback is used.",
        )
        if st.button("Rank selected cohort", type="primary", disabled=not (api_ok and cohort)):
            try:
                ranked = api_post("/rank", {"lsoa_codes": cohort})
                st.session_state["last_rank"] = ranked
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Ranking failed: {exc}")
        elif not api_ok:
            st.info("Start FastAPI (sidebar) so ranking can run through the integration backend.")
        elif not cohort:
            st.info("Select a cohort (or load the 10-neighbourhood demo) first.")

        ranked = st.session_state.get("last_rank")
        if ranked:
            st.success(
                f"Ranked {len(ranked.get('items', []))} neighbourhoods. "
                f"Score source: `{ranked.get('source')}`."
            )
            for note in ranked.get("notes") or []:
                st.caption(note)
            items = pd.DataFrame(ranked.get("items", []))
            if not items.empty:
                st.dataframe(items, use_container_width=True, hide_index=True)

    with shap_col:
        section_intro(
            "Why did this neighbourhood rank here?",
            "SHAP attributes the score to input features. Dark blue raised priority; light blue lowered it. "
            "Correlated features (floor area, rooms, heating cost) can share credit — treat this as explanation, not cause.",
        )
        explain_pick = st.selectbox(
            "Neighbourhood to explain",
            options=cohort or display["lsoa21cd"].astype(str).tolist()[:DEMO_COHORT_LSOA_COUNT],
            key="explain_lsoa",
            help="Usually the same cohort as ranking. One LSOA at a time.",
        )
        if st.button("Explain selection", disabled=not explain_pick):
            explained = None
            if api_ok:
                try:
                    explained = api_post("/explain", {"lsoa21cd": explain_pick, "top_n": 10})
                    st.session_state["last_explain"] = explained
                except requests.RequestException as exc:
                    st.warning(f"API explain unavailable ({exc}); trying a local fallback.")
                    explained = local_explain(explain_pick, base)
                    if explained:
                        st.session_state["last_explain"] = explained
            else:
                explained = local_explain(explain_pick, base)
                if explained:
                    st.session_state["last_explain"] = explained
                    st.info("API offline — showing local composite or SHAP fallback.")

        explained = st.session_state.get("last_explain")
        if explained and explained.get("error"):
            st.error(explained["error"])
        elif explained:
            if explained.get("source") == "local_fallback":
                st.caption("Source: local helper (FastAPI was not used).")
            if explained.get("caveat"):
                st.caption(explained["caveat"])
            pred = explained.get("prediction")
            base_v = explained.get("base_value")
            st.caption(
                f"Method: `{explained.get('method')}` · "
                f"Predicted priority: {float(pred or 0):.4f} · "
                f"Baseline (average) score: {float(base_v or 0):.4f}"
            )
            fig = shap_waterfall(
                explained.get("features") or [],
                base_value=float(base_v or 0.0),
                prediction=float(pred or 0.0),
                title=f"What drove the score for {explained.get('lsoa21cd')}",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_ledger_tab(api_ok: bool, cohort: list[str], display: pd.DataFrame) -> None:
    section_intro(
        "Record a synthetic grant and watch the twin update",
        "This is the closed loop: twin → AI rank → ledger → SQLite → map. "
        "Run the three events in order. After **verification**, the neighbourhood is marked verified "
        "and its priority is halved so it falls down the queue. All grant payloads are synthetic.",
    )
    st.markdown(f"**{SYNTHETIC_LABEL}** — eligibility, works claimed, and verification are generated, not real awards.")
    target = st.selectbox(
        "Neighbourhood for this grant trail",
        options=cohort or display["lsoa21cd"].astype(str).tolist()[:1],
        key="ledger_target",
        help="Use one LSOA from the selected cohort.",
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

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**1. Eligibility**")
        st.caption("Records that this neighbourhood is eligible for a simulated grant.")
        if st.button("Append eligibility", disabled=not (api_ok and target)):
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
    with e2:
        st.markdown("**2. Works claimed**")
        st.caption("Records that retrofit works were claimed against that eligibility.")
        if st.button("Append works claimed", disabled=not (api_ok and target)):
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
    with e3:
        st.markdown("**3. Verification**")
        st.caption("Records inspection, writes back to the twin, and drops priority.")
        if st.button("Append verification", disabled=not (api_ok and target), type="primary"):
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

    st.divider()
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Check the hash-chain**")
        st.caption("Walks every block and confirms each SHA-256 link is intact.")
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
            with st.expander("Recent blocks (technical)", expanded=False):
                st.json(verified.get("recent_blocks") or [])
    with v2:
        st.markdown("**Tampering demo**")
        st.caption(
            "Mutates an **in-memory copy** of the chain so you can show that verification fails. "
            "The real ledger file is not changed."
        )
        if st.button("Run tampering demo", disabled=not api_ok):
            try:
                demo = api_get("/ledger/tamper-demo")
                st.session_state["tamper_demo"] = demo
            except requests.RequestException as exc:
                st.error(str(exc))
        demo = st.session_state.get("tamper_demo")
        if demo:
            st.json(demo)


def render_whatif_tab(cohort_df: pd.DataFrame) -> None:
    section_intro(
        "Explore a funding scenario (does not write the ledger)",
        "These sliders are a scratch-pad: they estimate how many neighbourhoods a budget could cover "
        "and how mean EPC need / a fuel-poverty *proxy* might move. "
        "They do **not** record grants. The fuel-poverty indicator is an IMD + EPC assumption, "
        "not the official BEIS statistic.",
    )
    c1, c2 = st.columns(2)
    with c1:
        bands = st.slider(
            "Assumed EPC band uplift per funded neighbourhood",
            1,
            3,
            2,
            help="How many certificate bands we pretend funded homes improve by (for example D → B is two bands).",
        )
        retrofit_rate = st.slider(
            "Share of the cohort you intend to fund (%)",
            0,
            100,
            50,
            help="Percentage of selected neighbourhoods considered for funding, highest priority first.",
        )
    with c2:
        budget_cap = st.number_input(
            "Budget cap (£) — SYNTHETIC DATA",
            min_value=0,
            value=100_000,
            step=5_000,
            help="Hard ceiling. Funding stops when this is exhausted.",
        )
        cost_per = st.number_input(
            "Assumed cost per neighbourhood (£) — SYNTHETIC DATA",
            min_value=1_000,
            value=8_500,
            step=500,
            help="Simplified unit cost so the PoC can show a budget intersection. Not a real works price.",
        )

    if cohort_df.empty:
        st.info("Select neighbourhoods in the sidebar to compare before and after aggregates.")
        return

    scenario = apply_epc_uplift(cohort_df, bands)
    before_need = pd.to_numeric(scenario["epc_current_need"], errors="coerce").mean()
    after_need = pd.to_numeric(scenario["epc_need_after"], errors="coerce").mean()
    fp_before = int(scenario["fp_proxy_before"].sum())
    fp_after = int(scenario["fp_proxy_after"].sum())
    projection = whatif_budget_projection(
        cohort_df,
        retrofit_rate_pct=float(retrofit_rate),
        budget_cap_gbp=float(budget_cap),
        cost_per_lsoa_gbp=float(cost_per),
        epc_uplift_bands=bands,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Mean EPC need (1 = A … 7 = G)",
        f"{before_need:.2f}",
        f"{after_need - before_need:.2f} after uplift",
        help="Lower need is better. Delta is the change if every selected LSOA received the assumed uplift.",
    )
    m2.metric(
        "Fuel-poverty proxy (full cohort)",
        f"{fp_before}",
        f"{fp_after - fp_before} after uplift",
        help="Count of LSOAs treated as fuel-poor under the IMD + EPC proxy. Not official statistics.",
    )
    m3.metric(
        "Neighbourhoods funded",
        projection["funded_count"],
        help="How many LSOAs fit inside both the funding rate and the budget cap, highest priority first.",
    )
    m4.metric(
        "Budget spent (£)",
        f"{projection['budget_spent_gbp']:,.0f}",
        help="Synthetic spend only.",
    )
    st.caption(
        f"**{SYNTHETIC_LABEL}** costs · remaining budget £{projection['budget_remaining_gbp']:,.0f} · "
        f"funded fuel-poverty proxy {projection['fp_proxy_before']} → {projection['fp_proxy_after']}"
    )
    if projection["funded_codes"]:
        st.caption(
            "Would be funded: "
            + ", ".join(projection["funded_codes"][:12])
            + (" …" if len(projection["funded_codes"]) > 12 else "")
        )


def main() -> None:
    apply_theme()
    st.title("RetrofitTrust Birmingham")
    st.markdown(
        '<p class="rt-lead">A neighbourhood-scale digital twin for domestic retrofit prioritisation. '
        "It ranks Lower Super Output Areas, explains the score, records a synthetic grant trail, "
        "and writes the result back so the map updates.</p>",
        unsafe_allow_html=True,
    )
    st.caption("MSc AI proof-of-concept — not a production allocations system. British English. Seed = 42.")

    st.session_state.setdefault("api_base", DEFAULT_API)

    conn_col, help_col = st.columns(2, gap="large")
    with conn_col:
        st.subheader("Connection")
        st.text_input(
            "FastAPI base URL",
            key="api_base",
            help="Must match the uvicorn port. Default is 8001.",
        )

        api_ok = False
        health: dict = {}
        try:
            health = api_get("/health", timeout=3)
            api_ok = health.get("status") == "ok"
            st.success("API connected — ranking and ledger buttons are live.")
            st.caption(
                f"Trained LightGBM loaded: {'yes' if health.get('model_loaded') else 'no (composite fallback)'} · "
                f"{health.get('ledger_message')}"
            )
        except requests.RequestException:
            api_ok = False
            health = {}
            st.error("API not reachable. Map and filters still work; ranking and ledger need FastAPI.")
            with st.expander("How to start the API"):
                st.code(
                    "uvicorn retrofittrust.api.main:app --reload --app-dir src --port 8001",
                    language="bash",
                )
    with help_col:
        with st.expander("How to use this dashboard", expanded=False):
            st.markdown(
                """
1. **Load demo cohort** (or pick LSOAs) — that set is what ranking and the ledger act on.
2. Open **Map & Cohort** — darker blue means higher retrofit urgency.
3. **Rank & Explain** — score the cohort, then ask why one neighbourhood ranked there.
4. **Grant Ledger** — eligibility → works claimed → verification. The map should then show verified and a lower priority.
5. **What-If** is a scratch-pad only; it does not write the ledger.
                """
            )

    with st.expander("Glossary — what the terms mean", expanded=False):
        render_glossary()

    base, data_source = cached_dataset()
    mtime = db_mtime_token()
    display = live_frame(base, mtime)
    twin_state = fetch_all_lsoa_state()
    all_codes = display["lsoa21cd"].astype(str).tolist()
    if "cohort_select" not in st.session_state:
        demo = [c for c in select_demo_cohort(DEMO_COHORT_LSOA_COUNT) if c in set(all_codes)]
        st.session_state["cohort_select"] = demo or all_codes[:DEMO_COHORT_LSOA_COUNT]

    verified_n = int(display["verified"].sum()) if "verified" in display.columns else 0
    wb1, wb2, wb3 = st.columns(3)
    wb1.metric(
        "Neighbourhoods in the twin",
        f"{len(display):,}",
        help="Count of LSOAs currently loaded (Birmingham extract, or a labelled synthetic demo if processed data is missing).",
    )
    wb2.metric(
        "Verified after write-back",
        verified_n,
        help="How many LSOAs have a verification event recorded. These should look different on the map.",
    )
    wb3.metric(
        "Selected for ranking",
        len(st.session_state.get("cohort_select") or []),
        help="Size of the cohort chosen in the sidebar.",
    )

    if data_source == "synthetic_fallback":
        st.warning(
            f"**{SYNTHETIC_LABEL}** — the merged Birmingham dataset is not in this clone, "
            "so a labelled demo frame is shown. Ranking, explanation, and the ledger loop still run."
        )

    st.sidebar.header("Filters")
    if st.sidebar.button(
        f"Load demo cohort ({DEMO_COHORT_LSOA_COUNT} neighbourhoods)",
        type="secondary",
        help="Picks the highest-priority LSOAs so you can demonstrate the loop without hunting for codes.",
    ):
        demo = [c for c in select_demo_cohort(DEMO_COHORT_LSOA_COUNT) if c in set(all_codes)]
        st.session_state["cohort_select"] = demo or all_codes[:DEMO_COHORT_LSOA_COUNT]

    filtered = filter_cohort(display)
    cohort = st.sidebar.multiselect(
        "Selected neighbourhoods (LSOA codes)",
        options=all_codes,
        key="cohort_select",
        help="Rank, explain, and ledger buttons use this list. Map filters above do not change it.",
    )
    cohort_df = display[display["lsoa21cd"].isin(cohort)].copy() if cohort else filtered.head(0)

    geojson, featureidkey, geo_source = cached_geometries(
        tuple(display["lsoa21cd"].astype(str)),
        "borders-v2",
    )
    has_real_geo = geo_source not in ("synthetic_grid",)
    tab_map, tab_detail, tab_rank, tab_ledger, tab_whatif = st.tabs(
        [
            "Map & Cohort",
            "Neighbourhood Profile",
            "Rank & Explain",
            "Grant Ledger",
            "What-If Scenario",
        ]
    )
    with tab_map:
        render_map_tab(
            display,
            filtered,
            cohort,
            cohort_df,
            geojson,
            featureidkey,
            geo_source,
            data_source,
            mtime,
        )
    with tab_detail:
        section_intro(
            "Inspect one neighbourhood",
            "Energy, deprivation, data-quality flag, and the latest synthetic grant event for a single LSOA.",
        )
        detail_pick = st.selectbox(
            "Neighbourhood to inspect",
            options=cohort or all_codes[: max(DEMO_COHORT_LSOA_COUNT, 1)],
            key="detail_lsoa",
            help="Defaults to the selected cohort.",
        )
        if detail_pick:
            detail_row = display[display["lsoa21cd"].astype(str) == str(detail_pick)]
            if not detail_row.empty:
                render_lsoa_detail(detail_row.iloc[0], twin_state)
            else:
                st.warning("That LSOA is not in the current dataset.")
    with tab_rank:
        render_rank_tab(api_ok, cohort, display, base)
    with tab_ledger:
        render_ledger_tab(api_ok, cohort, display)
    with tab_whatif:
        render_whatif_tab(cohort_df)

    with st.expander("Limitations (read this before treating numbers as policy)"):
        st.markdown(
            """
- **Ecological fallacy** — IMD is LSOA-level and must not be read as a household's deprivation.
- **EPC performance gap** — modelled versus metered energy differs (about 16% gas, 31% electric); ranking is relative, not a kWh forecast.
- **SHAP** — TreeExplainer can misattribute importance among correlated features.
- **Ledger** — hashlib SHA-256 hash-chain simulation, not a live blockchain.
- **Grants, works, and verification** — labelled **SYNTHETIC DATA**.
            """
        )

    apply_theme()


main()
