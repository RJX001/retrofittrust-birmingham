"""Plotly figures for the digital twin (choropleth + SHAP waterfall)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# White / black / blue only — projector-friendly.
BLUE = "#1565C0"
BLUE_DARK = "#0D47A1"
BLUE_LIGHT = "#90CAF9"
BLACK = "#111111"
WHITE = "#FFFFFF"
GREY = "#616161"
BLUES = [[0.0, "#FFFFFF"], [0.35, "#BBDEFB"], [0.7, "#1565C0"], [1.0, "#0D47A1"]]


def _plain_layout(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(color=BLACK, size=13),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        **kwargs,
    )
    fig.update_xaxes(gridcolor="#E6E6E6", linecolor=BLACK, zerolinecolor="#E6E6E6")
    fig.update_yaxes(gridcolor="#E6E6E6", linecolor=BLACK)
    return fig


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
    colour_title = "Verified vs candidate" if color_col == "verified_num" else "Priority score"
    choropleth_kwargs: dict[str, Any] = dict(
        geojson=geojson,
        locations="lsoa21cd",
        featureidkey=featureidkey,
        color=color_col,
        color_continuous_scale=[[0.0, WHITE], [1.0, BLUE_DARK]] if color_col == "verified_num" else BLUES,
        hover_name="lsoa21cd",
        hover_data=hover or None,
        labels={
            color_col: colour_title,
            "lsoa21nm": "Neighbourhood",
            "verification_status": "Twin status",
            "imd_decile": "IMD decile",
            "epc_current": "Current EPC",
        },
    )
    if color_col == "verified_num":
        choropleth_kwargs["range_color"] = (0, 1)
    fig = px.choropleth(work, **choropleth_kwargs)
    fig.update_traces(
        marker_line_width=1.8,
        marker_line_color=BLACK,
        marker_opacity=1.0,
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor=WHITE,
        projection_type="mercator",
    )
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4),
        height=320,
        legend_title_text="",
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        geo=dict(
            bgcolor=WHITE,
            landcolor=WHITE,
            lakecolor=WHITE,
            showframe=False,
            showcoastlines=False,
            showland=False,
            showlakes=False,
            showcountries=False,
        ),
        font=dict(color=BLACK),
        coloraxis_colorbar=dict(
            title=colour_title,
            tickfont=dict(color=BLACK),
            titlefont=dict(color=BLACK),
            outlinecolor=BLACK,
            outlinewidth=0.6,
            bgcolor=WHITE,
        ),
    )
    if selected:
        sel = work[work["lsoa21cd"].astype(str).isin([str(s) for s in selected])]
        if not sel.empty:
            fig.add_trace(
                go.Choropleth(
                    geojson=geojson,
                    locations=sel["lsoa21cd"],
                    z=[1] * len(sel),
                    featureidkey=featureidkey,
                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale=False,
                    coloraxis="coloraxis2",
                    marker_line_width=3.2,
                    marker_line_color=BLACK,
                    hoverinfo="skip",
                    name="Selected cohort",
                )
            )
            fig.update_layout(coloraxis2=dict(showscale=False))
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
        lambda r: f"{r.get('lsoa21cd', '')}",
        axis=1,
    )
    selected_set = {str(s) for s in (selected or [])}
    colours = [BLUE_DARK if str(c) in selected_set else BLUE for c in work["lsoa21cd"]]
    fig = go.Figure(
        go.Bar(
            x=pd.to_numeric(work[color_col], errors="coerce"),
            y=work["label"],
            orientation="h",
            marker_color=colours,
            marker_line_color=BLACK,
            marker_line_width=0.4,
            hovertemplate="%{y}<br>Priority: %{x:.3f}<extra></extra>",
        )
    )
    _plain_layout(
        fig,
        height=220,
        margin=dict(l=8, r=8, t=36, b=8),
        title=dict(
            text="Highest-priority neighbourhoods",
            font=dict(color=BLACK, size=14),
        ),
        xaxis_title="",
        yaxis=dict(autorange="reversed", title="", tickfont=dict(color=BLACK, size=10)),
        xaxis=dict(tickfont=dict(color=BLACK)),
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
            connector={"line": {"color": GREY}},
            decreasing={"marker": {"color": BLUE_LIGHT}},
            increasing={"marker": {"color": BLUE_DARK}},
            totals={"marker": {"color": BLACK}},
        )
    )
    _plain_layout(
        fig,
        title=dict(text=title, font=dict(color=BLACK)),
        height=max(280, 28 * (len(labels) + 2)),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        xaxis_title="Dark blue raised the score; light blue lowered it",
    )
    return fig
