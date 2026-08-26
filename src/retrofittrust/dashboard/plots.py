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
        fig.update_traces(
            marker_line_width=1,
            selector=dict(type="choroplethmapbox"),
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
