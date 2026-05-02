"""
Drive a headless Chromium browser through every tab of the running Streamlit
app, take a full-page screenshot of each tab, and assemble them into a single
PDF that satisfies the Phase 4 submission requirement (`presentation_slides.pdf`).

Prerequisites:
  - Streamlit dev server running at http://localhost:8501
  - playwright + chromium installed:
        pip install playwright
        playwright install chromium

Usage:
    python presentation/capture_app_to_pdf.py [--url http://localhost:8501] [--out presentation/presentation_slides.pdf]
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


TAB_NAMES = [
    "Welcome",
    "Phase 1: Data",
    "Phase 2: Pipeline",
    "Phase 2: Models",
    "Phase 3: Spark + Macro",
    "Phase 3: MCP",
    "Predict a loan",
    "Insights + Next",
    "Q&A",
]

ROOT = Path(__file__).resolve().parent
SHOTS_DIR = ROOT / "screenshots"
DEFAULT_OUT = ROOT / "presentation_slides.pdf"


def set_predict_demo_state(page):
    """In the Predict tab, toggle compare mode AND set distinct presets
    on the two forms so the captured screenshot tells a story (Safe vs Risky
    side by side with two gauge readouts)."""
    try:
        # 1. Toggle compare mode on
        toggle = page.locator('label:has-text("Compare two loans")').first
        toggle.scroll_into_view_if_needed()
        toggle.click(timeout=4000)
        page.wait_for_timeout(900)

        # 2. Set Loan A preset to Safe
        preset_a = page.locator('div[data-testid="stSelectbox"]').filter(
            has_text="Quick preset").nth(0)
        preset_a.click(timeout=4000)
        page.wait_for_timeout(400)
        page.locator('li:has-text("Safe (A2, FICO 760)")').first.click(timeout=4000)
        page.wait_for_timeout(900)

        # 3. Set Loan B preset to Risky
        preset_b = page.locator('div[data-testid="stSelectbox"]').filter(
            has_text="Quick preset").nth(1)
        preset_b.click(timeout=4000)
        page.wait_for_timeout(400)
        page.locator('li:has-text("Risky (E4, FICO 620)")').first.click(timeout=4000)
        page.wait_for_timeout(900)

        # 4. Click Predict
        predict_btn = page.get_by_role("button", name="Predict both loans")
        predict_btn.scroll_into_view_if_needed()
        predict_btn.click(timeout=4000)

        # 5. Wait for the result panels to render (look for the result heading)
        page.wait_for_selector('text="Loan A result"', timeout=15000)
        # Extra time for both plotly gauges to fully animate in
        page.wait_for_timeout(2500)
    except Exception as exc:
        print(f"    note: could not stage compare-mode demo ({exc}); "
              f"capturing form as-is")


def expand_viewport_for_content(page):
    """Streamlit clips content to the viewport height. Resize the viewport so
    the full tab content fits, then we can take a normal screenshot."""
    # Compute the actual content height of the inner block container
    content_h = page.evaluate("""
        () => {
            const el = document.querySelector('.block-container');
            if (!el) return document.body.scrollHeight;
            return Math.max(el.scrollHeight, document.body.scrollHeight);
        }
    """)
    # Add slack for any animations / margins
    target_h = max(int(content_h) + 120, 1200)
    page.set_viewport_size({"width": page.viewport_size["width"],
                            "height": target_h})
    # let Streamlit re-layout under the new height
    page.wait_for_timeout(700)


def capture_tab(page, tab_index, tab_name, out_path, base_width):
    """Click the tab at position `tab_index` (0-based) in the tab list and
    screenshot it. Index-based is more reliable than text-match because
    Streamlit's [data-baseweb=tab] elements include both buttons and panels.
    """
    print(f"  - {tab_name}")
    # Reset viewport so the tab strip renders correctly first
    page.set_viewport_size({"width": base_width, "height": 900})
    page.wait_for_timeout(400)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)

    tab = page.get_by_role("tab").nth(tab_index)
    tab.click(timeout=15000)
    # Allow plotly + images to render
    page.wait_for_timeout(2000)

    if tab_name == "Predict a loan":
        set_predict_demo_state(page)

    # Resize viewport to fit the entire tab's content vertically
    expand_viewport_for_content(page)

    # Scroll to top inside that viewport
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(500)

    page.screenshot(path=str(out_path), full_page=True, animations="disabled")


def stitch_pdf(image_paths, out_pdf):
    """Combine PNG screenshots into a single multi-page PDF using Pillow.
    Each screenshot becomes one page sized to its native pixel dimensions."""
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        images.append(img)
    if not images:
        raise RuntimeError("no screenshots to stitch")
    images[0].save(
        str(out_pdf),
        save_all=True,
        append_images=images[1:],
        format="PDF",
        resolution=150.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--keep-screenshots", action="store_true",
                        help="don't delete the per-tab PNGs after stitching")
    args = parser.parse_args()

    SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = Path(args.out).resolve()

    print(f"Connecting to Streamlit at {args.url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": args.width, "height": 900},
            device_scale_factor=2,  # crisper screenshots
        )
        page = context.new_page()

        try:
            page.goto(args.url, wait_until="networkidle", timeout=60000)
        except Exception as exc:
            print(f"FAILED to load {args.url}: {exc}", file=sys.stderr)
            print(f"Is the streamlit server running? "
                  f"`streamlit run src/app/streamlit_app.py`", file=sys.stderr)
            sys.exit(1)

        # First-load app initialization (model load, cache warm-up)
        print("Waiting for initial load (model + pipeline pickling)...")
        page.wait_for_timeout(4000)

        shot_paths = []
        print("Capturing tabs:")
        for i, name in enumerate(TAB_NAMES, start=1):
            shot = SHOTS_DIR / f"slide_{i:02d}_{name.replace(' ', '_').replace('+', 'plus').replace(':', '').replace('?', '').replace('&', 'and')}.png"
            try:
                capture_tab(page, i - 1, name, shot, args.width)
                shot_paths.append(shot)
            except Exception as exc:
                print(f"    FAILED on '{name}': {exc}", file=sys.stderr)

        browser.close()

    if not shot_paths:
        print("No screenshots captured.", file=sys.stderr)
        sys.exit(1)

    print(f"\nStitching {len(shot_paths)} screenshots into {out_pdf}")
    stitch_pdf(shot_paths, out_pdf)
    print(f"  wrote {out_pdf}  ({out_pdf.stat().st_size / 1024:.1f} KB)")

    if not args.keep_screenshots:
        for p in shot_paths:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            SHOTS_DIR.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
