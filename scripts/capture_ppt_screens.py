"""Capture 16:9 dashboard screenshots for a short lecturer PowerPoint.

Run while Streamlit is on http://127.0.0.1:8501:

    .venv\\Scripts\\python.exe scripts/capture_ppt_screens.py
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "ppt"
URL = "http://127.0.0.1:8501"
# 16:9, readable on a projector
VIEWPORT = {"width": 1920, "height": 1080}


def _click_tab(page, label: str) -> None:
    tabs = page.locator("[data-testid='stTabs']")
    target = tabs.get_by_text(label, exact=True)
    if target.count() == 0:
        target = page.get_by_text(label, exact=True)
    target.first.click(timeout=8000)
    page.wait_for_timeout(2500)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
        page.locator("text=RetrofitTrust Birmingham").first.wait_for(timeout=30000)
        page.wait_for_timeout(7000)
        page.add_style_tag(
            content="header, [data-testid='stToolbar'], [data-testid='stDecoration'] { display:none !important; }"
        )

        shots = [
            ("01_overview.png", None),
            ("02_map_cohort.png", "Map & cohort"),
            ("03_neighbourhood.png", "Neighbourhood profile"),
            ("04_rank_explain.png", "Rank & explain"),
            ("05_grant_ledger.png", "Grant ledger"),
        ]
        for filename, tab in shots:
            if tab:
                _click_tab(page, tab)
            dest = OUT / filename
            page.screenshot(path=str(dest), full_page=False)
            print(f"wrote {dest}")
        browser.close()


if __name__ == "__main__":
    main()
