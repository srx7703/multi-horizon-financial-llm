"""
Drive the Streamlit RAG demo in a headed Playwright browser and capture
5 frames for the README GIF. Assembles `docs/demo.gif` with PIL.
"""
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

FRAMES_DIR = Path(__file__).parent / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
GIF_OUT = Path(__file__).parent / "demo.gif"
URL = "http://localhost:8501/"

QUERY = "What are NVIDIA's biggest risks from AI chip export controls?"
VIEWPORT = {"width": 1280, "height": 800}


def capture(page, path, pause=0.4):
    time.sleep(pause)
    page.screenshot(path=str(path), full_page=False)
    print(f"  saved {path.name}")


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()
        print(f"→ navigating to {URL}")
        page.goto(URL, wait_until="networkidle", timeout=60_000)

        page.wait_for_selector('textarea[aria-label="Ask a financial question:"]', timeout=30_000)
        time.sleep(1.0)

        print("frame_01: initial home")
        capture(page, FRAMES_DIR / "frame_01.png")

        print("frame_02: typing query")
        textarea = page.locator('textarea[aria-label="Ask a financial question:"]')
        textarea.click()
        textarea.fill(QUERY)
        capture(page, FRAMES_DIR / "frame_02.png", pause=0.6)

        print("frame_03: clicked Analyze, loading state")
        analyze = page.get_by_role("button", name="🔍 Analyze")
        analyze.click()
        capture(page, FRAMES_DIR / "frame_03.png", pause=1.2)

        print("frame_04: waiting for response (spinner to disappear)")
        for _ in range(60):
            time.sleep(1)
            spinner = page.locator('text=Retrieving from SEC filings').count()
            if spinner == 0:
                break
        time.sleep(2)
        page.evaluate("""() => {
          // Streamlit scrolls via section.main or the outer html — set both.
          const el = document.querySelector('section.main, .main');
          if (el) el.scrollTop = 260;
          window.scrollTo(0, 260);
        }""")
        capture(page, FRAMES_DIR / "frame_04.png", pause=0.6)

        print("frame_05: scrolled deeper to show sources")
        page.evaluate("""() => {
          const el = document.querySelector('section.main, .main');
          if (el) el.scrollTop = 900;
          window.scrollTo(0, 900);
        }""")
        capture(page, FRAMES_DIR / "frame_05.png", pause=0.8)

        browser.close()

    print("\n→ assembling GIF")
    frames = [Image.open(FRAMES_DIR / f"frame_0{i}.png").convert("RGB") for i in range(1, 6)]
    w = 900
    resized = [f.resize((w, int(f.height * w / f.width)), Image.LANCZOS) for f in frames]
    durations = [1500, 2000, 1800, 3500, 3000]
    resized[0].save(
        GIF_OUT,
        save_all=True,
        append_images=resized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = GIF_OUT.stat().st_size / 1024
    print(f"✓ {GIF_OUT} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
