"""Plotly figures for the digital twin (choropleth + SHAP waterfall)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from retrofittrust.dashboard.data_loader import BIRMINGHAM_CENTRE


def choropleth_priority(
    df: pd.DataFrame,
    geojson: dict[str, Any],
    *,
    featureidkey: str,
    color_col: str = "priority_display",
    selected: list[str] | None = None,
) -> go.Figure:
    work = df.copy()
    if color_col not in work.columns:
        color_col = "priority_score" if "priority_score" in work.columns else work.columns[-1]
    hover = [c for c in ("lsoa21nm", "verification_status", "imd_decile", "epc_current") if c in work.columns]
    fig = px.choropleth_mapbox(
        work,
        geojson=geojson,
        locations="lsoa21cd",
        featureidkey=featureidkey,
        color=color_col,
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        center=BIRMINGHAM_CENTRE,
        zoom=10,
        opacity=0.7,
        hover_name="lsoa21cd",
        hover_data=hover or None,
        labels={color_col: "Priority"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        legend_title_text="",
        coloraxis_colorbar=dict(title="Priority"),
    )
    if selected:
        sel = work[work["lsoa21cd"].astype(str).isin([str(s) for s in selected])]
        if not sel.empty:
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson,
                    locations=sel["lsoa21cd"],
                    z=[1] * len(sel),
                    featureidkey=featureidkey,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    marker_line_width=2.5,
                    marker_line_color="#1a1a1a",
                    hoverinfo="skip",
                    name="Selected cohort",
                )
            )
    return fig


def bar_priority_fallback(
    df: pd.DataFrame,
    *,
    color_col: str = "priority_display",
    selected: list[str] | None = None,
    top_n: int = 25,
) -> go.Figure:
    """Sortable bar fallback when ONS GeoJSON is absent."""
    work = df.copy()
    if color_col not in work.columns:
        color_col = "priority_score" if "priority_score" in work.columns else work.columns[-1]
    work = work.sort_values(color_col, ascending=False).head(top_n)
    work["label"] = work.apply(
        lambda r: f"{r.get('lsoa21cd', '')} — {str(r.get('lsoa21nm', '') or '')[:28]}",
        axis=1,
    )
    colours = ["#E45756" if str(c) in {str(s) for s in (selected or [])} else "#4C78A8" for c in work["lsoa21cd"]]
    fig = go.Figure(
        go.Bar(
            x=pd.to_numeric(work[color_col], errors="coerce"),
            y=work["label"],
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}<br>Priority: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(320, 22 * len(work) + 80),
        margin=dict(l=20, r=20, t=30, b=20),
        title="Priority ranking (table fallback — no ONS GeoJSON)",
        xaxis_title="Retrofit priority score",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def shap_waterfall(
    features: list[dict[str, Any]],
    *,
    base_value: float,
    prediction: float,
    title: str,
) -> go.Figure:
    ordered = list(reversed(features))
    labels = [str(f.get("feature")) for f in ordered]
    deltas = [float(f.get("shap_value") or 0.0) for f in ordered]
    measures = ["relative"] * len(deltas)
    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=measures + ["total"],
            y=labels + ["prediction"],
            x=deltas + [prediction],
            base=base_value,
            connector={"line": {"color": "#888"}},
            decreasing={"marker": {"color": "#4C78A8"}},
            increasing={"marker": {"color": "#E45756"}},
            totals={"marker": {"color": "#72B7B2"}},
        )
    )
    fig.update_layout(
        title=title,
        height=max(280, 28 * (len(labels) + 2)),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        xaxis_title="Contribution to priority score",
    )
    return fig
