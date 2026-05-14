# -*- coding: utf-8 -*-
"""Take screenshots of the web app for the thesis."""
import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
BASE_URL = "http://127.0.0.1:8000"

def shot(page, path, full=False):
    page.wait_for_timeout(2000)
    page.screenshot(path=path, full_page=full)
    print(f"Saved: {path}")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 900})

    # ── 1. Single image view (default) ───────────────────────────────────────
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle", timeout=15000)
    shot(page, os.path.join(FIGURES, "ui_single_image_view.png"))

    # ── 2. Try to switch to city-map view ────────────────────────────────────
    try:
        # Look for city map button/tab
        btns = page.locator("button, [role=tab]").all_text_contents()
        print("Buttons found:", btns[:10])
        for btn_text in ["City map", "City Map", "Городская карта", "city", "map view", "aggregate"]:
            locator = page.get_by_text(btn_text, exact=False)
            if locator.count() > 0:
                locator.first.click()
                page.wait_for_load_state("networkidle", timeout=8000)
                page.wait_for_timeout(3000)
                shot(page, os.path.join(FIGURES, "ui_city_map_view.png"))
                break
    except Exception as e:
        print(f"City map switch: {e}")
        # Take screenshot anyway
        shot(page, os.path.join(FIGURES, "ui_city_map_view.png"))

    # ── 3. Zoom into Astana on the map ───────────────────────────────────────
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle", timeout=15000)
    page.wait_for_timeout(2000)

    # Zoom map to Astana coordinates via JS
    page.evaluate("""
        () => {
            // Try to find Leaflet map and set view to Astana
            if (window.L) {
                const maps = Object.values(window.L._layers || {});
            }
            // Find map instances
            const mapDivs = document.querySelectorAll('.leaflet-container');
            mapDivs.forEach(div => {
                const map = div._leaflet_map;
                if (map) {
                    map.setView([51.166, 71.446], 14);
                }
            });
        }
    """)
    page.wait_for_timeout(3000)
    shot(page, os.path.join(FIGURES, "ui_map_astana.png"))

    # ── 4. Full page screenshot ───────────────────────────────────────────────
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle", timeout=15000)
    page.wait_for_timeout(2000)
    shot(page, os.path.join(FIGURES, "ui_full_interface.png"), full=True)

    browser.close()

print("All screenshots done.")
