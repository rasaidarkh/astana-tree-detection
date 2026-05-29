# -*- coding: utf-8 -*-
"""Capture two more Map-view screenshots that actually show detections:
one at city zoom (heat-map mode), one zoomed into a single block (polygon mode).
"""
from __future__ import annotations
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
FIGURES = HERE / "figures"
BASE_URL = "http://127.0.0.1:8000"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()

        page.goto(BASE_URL, wait_until="networkidle", timeout=20000)
        page.wait_for_selector(".topbar-brand", timeout=10000)
        page.wait_for_timeout(2500)

        # City-zoom + Heat-map mode
        page.evaluate("""() => {
            const divs = document.querySelectorAll('.leaflet-container');
            divs.forEach((d) => { if (d._leaflet_map) d._leaflet_map.setView([51.105, 71.415], 16); });
        }""")
        page.wait_for_timeout(1500)
        page.click("button.seg-btn:has-text('Heat')")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(FIGURES / "ui_canopy_map_heat.png"))
        print("  wrote ui_canopy_map_heat.png")

        # Zoom into a single block — Polygon display
        page.click("button.seg-btn:has-text('Polygon')")
        page.evaluate("""() => {
            const divs = document.querySelectorAll('.leaflet-container');
            divs.forEach((d) => { if (d._leaflet_map) d._leaflet_map.setView([51.105, 71.415], 19); });
        }""")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(FIGURES / "ui_canopy_map_polygon_zoom.png"))
        print("  wrote ui_canopy_map_polygon_zoom.png")

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
