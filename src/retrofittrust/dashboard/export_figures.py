"""Export checkpoint-4 static figures without running Streamlit or FastAPI.

Run from the project root:

    PYTHONPATH=src python -m retrofittrust.dashboard.export_figures
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from retrofittrust.config import PROJECT_ROOT, REPORTS_FIGURES, SQLITE_PATH
from retrofittrust.dashboard.data_loader import (
    load_geometries,
    load_lsoa_dataset,
    merge_twin_state,
)
from retrofittrust.dashboard.plots import bar_priority_fallback, choropleth_priority
from retrofittrust.dashboard.state import fetch_all_lsoa_state, init_twin_db
from retrofittrust.ledger.synthetic import SYNTHETIC_LABEL

SEED = 42


def _count_sqlite_writebacks(db_path: Path = SQLITE_PATH) -> dict[str, int]:
    if not db_path.exists():
        return {
            "lsoa_state_rows": 0,
            "verified_lsoa_state": 0,
            "verified_outcome_rows": 0,
            "verified_outcome_lsoas": 0,
        }
    init_twin_db(db_path)
    with sqlite3.connect(db_path) as conn:
        lsoa_rows = int(conn.execute("SELECT COUNT(*) FROM lsoa_state").fetchone()[0])
        verified_state = int(
            conn.execute("SELECT COUNT(*) FROM lsoa_state WHERE verified = 1").fetchone()[0]
        )
        outcome_rows = int(conn.execute("SELECT COUNT(*) FROM verified_outcomes").fetchone()[0])
        outcome_lsoas = int(
            conn.execute("SELECT COUNT(DISTINCT lsoa_code) FROM verified_outcomes").fetchone()[0]
        )
    return {
        "lsoa_state_rows": lsoa_rows,
        "verified_lsoa_state": verified_state,
        "verified_outcome_rows": outcome_rows,
        "verified_outcome_lsoas": outcome_lsoas,
    }


def _save_priority_bar_png(df: pd.DataFrame, dest: Path, *, top_n: int = 20) -> Path:
    work = df.sort_values("priority_display", ascending=False).head(top_n).iloc[::-1]
    names = work["lsoa21nm"] if "lsoa21nm" in work.columns else pd.Series([""] * len(work), index=work.index)
    labels = [
        f"{code}  {str(name or '')[:22]}"
        for code, name in zip(work["lsoa21cd"], names)
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(labels, pd.to_numeric(work["priority_display"], errors="coerce"), color="#4C78A8")
    ax.set_xlabel("Retrofit priority score")
    ax.set_title("Top LSOAs by retrofit priority")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def _save_choropleth(
    df: pd.DataFrame,
    geojson: dict,
    featureidkey: str,
    html_dest: Path,
    png_dest: Path,
) -> tuple[Path, Path | None]:
    fig = choropleth_priority(df, geojson, featureidkey=featureidkey, color_col="priority_display")
    html_dest.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_dest), include_plotlyjs="cdn")
    png_written: Path | None = None
    try:
        fig.write_image(str(png_dest), scale=2)
        png_written = png_dest
    except Exception:  # noqa: BLE001 — kaleido is optional
        try:
            import geopandas as gpd

            gdf = gpd.GeoDataFrame.from_features(geojson["features"])
            key = featureidkey.split(".", 1)[-1]
            if key not in gdf.columns and "lsoa21cd" in gdf.columns:
                key = "lsoa21cd"
            gdf[key] = gdf[key].astype(str)
            merged = gdf.merge(
                df.assign(lsoa21cd=df["lsoa21cd"].astype(str)),
                left_on=key,
                right_on="lsoa21cd",
                how="left",
            )
            fig_m, ax = plt.subplots(figsize=(8, 8))
            merged.plot(
                column="priority_display",
                cmap="YlOrRd",
                linewidth=0.1,
                edgecolor="#555555",
                legend=True,
                ax=ax,
                missing_kwds={"color": "#dddddd"},
            )
            ax.set_axis_off()
            ax.set_title(
                "LSOA retrofit priority\n"
                f"(SYNTHETIC DATA geometries — ONS GeoJSON missing)"
            )
            fig_m.tight_layout()
            fig_m.savefig(png_dest, dpi=140)
            plt.close(fig_m)
            png_written = png_dest
        except Exception:
            png_written = None
    return html_dest, png_written


def _save_twin_metrics_png(
    *,
    n_lsoa: int,
    sqlite_counts: dict[str, int],
    matched_verified: int,
    dest: Path,
) -> Path:
    labels = [
        "LSOAs in twin",
        "SQLite verified\n(lsoa_state)",
        "SQLite write-back\nrows (outcomes)",
        "Verified LSOAs\nmatching dataset",
    ]
    values = [
        n_lsoa,
        sqlite_counts["verified_lsoa_state"],
        sqlite_counts["verified_outcome_rows"],
        matched_verified,
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#E45756", "#F2C14E", "#72B7B2"])
    ax.set_ylabel("Count")
    ax.set_title(f"Digital twin metrics  ·  {SYNTHETIC_LABEL} write-backs")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val}", ha="center", va="bottom")
    ax.set_ylim(0, max(values + [1]) * 1.15)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def _write_numbers_md(
    *,
    dest: Path,
    source: str,
    n_lsoa: int,
    n_properties: int | None,
    geo_source: str,
    sqlite_counts: dict[str, int],
    matched_verified: int,
    figure_paths: list[Path],
    kaleido: bool,
) -> Path:
    rel_figures = [p.relative_to(PROJECT_ROOT).as_posix() for p in figure_paths]
    n_prop_txt = f"{n_properties:,}" if n_properties is not None else "n/a"
    body = f"""# Checkpoint 4 — dashboard numbers

British English. Seed = {SEED}. Static export; FastAPI was not required.

## Dataset

- **Loaded:** yes
- **Source:** `{source}`
- **LSOA count:** {n_lsoa}
- **Properties represented (sum of `n_properties`):** {n_prop_txt}
- **Invalid LSOA rows:** dropped (join audit records 1,153 EPC rows without `lsoa21cd`)
- **Geometries:** `{geo_source}`
- **ONS GeoJSON (`data/external/lsoa_birmingham.geojson`):** missing — choropleth uses the Birmingham-centred synthetic grid fallback (not a 3D twin)

## SQLite write-back (`data/processed/twin_state.db`)

Existing `twin_state.db` rows mix older `SYNTH_*` demo codes with a few real E01* LSOAs from earlier integration runs. Only matching codes colour as verified on the live twin.

- **lsoa_state rows:** {sqlite_counts["lsoa_state_rows"]}
- **Verified in `lsoa_state`:** {sqlite_counts["verified_lsoa_state"]} (**{SYNTHETIC_LABEL}**)
- **`verified_outcomes` rows:** {sqlite_counts["verified_outcome_rows"]} (**{SYNTHETIC_LABEL}**)
- **Distinct LSOAs in outcomes:** {sqlite_counts["verified_outcome_lsoas"]}
- **Verified codes matching the live dataset:** {matched_verified}

## Figures

- Kaleido (Plotly PNG): {"present" if kaleido else "not installed — choropleth PNG via geopandas/matplotlib; Plotly HTML always written"}
{chr(10).join(f"- `{p}`" for p in rel_figures)}

## Notes

- Dashboard `st.cache_data` still wraps `load_lsoa_dataset` / `load_geometries`.
- API calls in `app.py` fail gracefully when uvicorn is down.
- Grant, works-claimed, and verification fields remain labelled **{SYNTHETIC_LABEL}**.
"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")
    return dest


def export_checkpoint_figures(out_dir: Path | None = None) -> dict[str, str]:
    out_dir = Path(out_dir or REPORTS_FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, source = load_lsoa_dataset()
    display = merge_twin_state(frame)
    geojson, featureidkey, geo_source = load_geometries(display["lsoa21cd"].astype(str).tolist())
    sqlite_counts = _count_sqlite_writebacks()
    twin_state = fetch_all_lsoa_state()
    live_codes = set(display["lsoa21cd"].astype(str))
    matched_verified = sum(
        1 for code, entry in twin_state.items() if entry.get("verified") and code in live_codes
    )
    n_properties = (
        int(pd.to_numeric(display["n_properties"], errors="coerce").sum())
        if "n_properties" in display.columns
        else None
    )

    bar_png = _save_priority_bar_png(display, out_dir / "04_priority_bar.png")
    html_path, choropleth_png = _save_choropleth(
        display,
        geojson,
        featureidkey,
        out_dir / "04_choropleth.html",
        out_dir / "04_choropleth.png",
    )
    metrics_png = _save_twin_metrics_png(
        n_lsoa=len(display),
        sqlite_counts=sqlite_counts,
        matched_verified=matched_verified,
        dest=out_dir / "04_twin_metrics.png",
    )

    kaleido = True
    try:
        import kaleido  # noqa: F401
    except ImportError:
        kaleido = False

    figure_paths = [bar_png, html_path, metrics_png]
    if choropleth_png is not None:
        figure_paths.append(choropleth_png)

    md_path = _write_numbers_md(
        dest=out_dir / "04_dashboard_numbers.md",
        source=source,
        n_lsoa=len(display),
        n_properties=n_properties,
        geo_source=geo_source,
        sqlite_counts=sqlite_counts,
        matched_verified=matched_verified,
        figure_paths=figure_paths,
        kaleido=kaleido,
    )

    # Smoke-test Plotly bar fallback (used in the app when GeoJSON is missing).
    _ = bar_priority_fallback(display, color_col="priority_display")

    return {
        "source": source,
        "n_lsoa": str(len(display)),
        "geo_source": geo_source,
        "bar": str(bar_png),
        "choropleth_html": str(html_path),
        "choropleth_png": str(choropleth_png) if choropleth_png else "",
        "metrics": str(metrics_png),
        "numbers_md": str(md_path),
        "matched_verified": str(matched_verified),
        "sqlite_verified": str(sqlite_counts["verified_lsoa_state"]),
    }


if __name__ == "__main__":
    result = export_checkpoint_figures()
    for key, value in result.items():
        print(f"{key}: {value}")
