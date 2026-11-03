"""The demo page in a real browser, against a real server with untrained models on the CPU."""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
START_SECONDS = 120.0
FRAME_MS = 60_000
SOLID = "(() => { const c = document.getElementById('%s'); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] === 126 && d[i+1] === 143 && d[i+2] === 134) n++; return n; })()"
SMOKE = "(() => { const c = document.getElementById('smoke'); return Array.from(c.getContext('2d').getImageData(0, 0, c.width, c.height).data).join(); })()"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def url():
    """A demo server. Few members and threads, so that it stays off the GPU and light on the CPU."""
    port = _free_port()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
    command = [sys.executable, "scripts/run_demo.py", "--random-weights", "--device", "cpu", "--members", "2", "--preset", "empty", "--port", str(port)]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    address = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + START_SECONDS
    try:
        while True:
            try:
                urllib.request.urlopen(address, timeout=2).close()
                break
            except OSError:
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"the demo server didn't start, exit code {process.poll()}")
                time.sleep(0.2)
        yield address
    finally:
        process.terminate()
        process.wait(timeout=30)


@pytest.fixture(scope="module")
def browser():
    """The installed Chrome, or Playwright's own. Software rendering keeps it off the GPU."""
    with sync_playwright() as playwright:
        kinds = [{"channel": "chrome"}] if shutil.which("google-chrome") else []
        try:
            launched = playwright.chromium.launch(args=["--disable-gpu"], **(kinds[0] if kinds else {}))
        except PlaywrightError:
            launched = playwright.chromium.launch(args=["--disable-gpu"])
        yield launched
        launched.close()


@pytest.fixture
def page(browser, url):
    context = browser.new_context(viewport={"width": 1280, "height": 720})
    opened = context.new_page()
    opened.goto(url)
    _wait_for_frame(opened)
    _send(opened, {"type": "preset", "name": "empty"})
    _send(opened, {"type": "mode", "mode": "ai"})
    yield opened
    context.close()


def _send(page, message: dict) -> None:
    page.evaluate("m => window.sandehaSend(m)", message)


def _wait_for_frame(page) -> None:
    page.wait_for_function("document.body.dataset.frames > 0", timeout=FRAME_MS)


def _wait_for_solid(page, canvas: str = "smoke", before: int = 0) -> int:
    page.wait_for_function(f"{SOLID % canvas} > {before}", timeout=FRAME_MS)
    return page.evaluate(SOLID % canvas)


def _wait_for_empty(page) -> None:
    page.wait_for_function(f"{SOLID % 'smoke'} === 0", timeout=FRAME_MS)


def _centre(page, canvas: str = "smoke") -> tuple[float, float]:
    box = page.locator(f"#{canvas}").bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def test_canvas_updates_over_time(page):
    first = page.evaluate(SMOKE)
    page.wait_for_function(f"{SMOKE} !== {first!r}", timeout=FRAME_MS)
    readout = page.locator("#fps")
    expect(readout).not_to_have_text("", timeout=FRAME_MS)
    assert page.locator("#share").inner_text().endswith("%")


def test_drawing_changes_obstacle_layer(page):
    _wait_for_empty(page)
    x, y = _centre(page)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 60, y, steps=4)
    page.mouse.up()
    drawn = _wait_for_solid(page)
    assert _wait_for_solid(page, "doubt") > 0

    # Erasing takes away what the brush drew, on the page, through the server.
    page.get_by_role("radio", name="Erase").check(force=True)
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 60, y, steps=4)
    page.mouse.up()
    page.wait_for_function(f"{SOLID % 'smoke'} < {drawn}", timeout=FRAME_MS)


def test_indicator_switches_in_solver_mode(page):
    physics, ai = page.locator("#lamp-physics"), page.locator("#lamp-ai")
    expect(ai).to_have_class("lamp lit", timeout=FRAME_MS)
    expect(physics).to_have_class("lamp")

    page.get_by_role("radio", name="Physics only").check(force=True)
    expect(physics).to_have_class("lamp lit", timeout=FRAME_MS)
    expect(ai).to_have_class("lamp")

    page.get_by_role("radio", name="AI only").check(force=True)
    expect(ai).to_have_class("lamp lit", timeout=FRAME_MS)
    expect(physics).to_have_class("lamp")


def test_no_external_requests(browser, url):
    context = browser.new_context(viewport={"width": 1280, "height": 720})
    opened = context.new_page()
    seen = []
    opened.on("request", lambda request: seen.append(request.url))
    opened.on("websocket", lambda socket_: seen.append(socket_.url))
    opened.goto(url)
    _wait_for_frame(opened)
    opened.evaluate("document.fonts.ready.then(() => null)")
    context.close()
    hosts = {urlparse(address).hostname for address in seen}
    assert hosts == {"127.0.0.1"}, seen
    assert any(address.startswith("ws://") for address in seen)
    assert any(address.endswith(".woff2") for address in seen)


def test_layout_fits_1280x720(page):
    size = page.evaluate("[document.documentElement.scrollWidth, document.documentElement.scrollHeight]")
    assert size[0] <= 1280 and size[1] <= 720, size

    smoke, doubt = page.locator("#smoke").bounding_box(), page.locator("#doubt").bounding_box()
    assert smoke["width"] == doubt["width"] == smoke["height"] == doubt["height"]
    assert smoke["y"] == doubt["y"]
    assert smoke["width"] >= 400

    controls = page.locator("button, input, canvas, #lamp-ai, #lamp-physics, #fps, #share")
    assert controls.count() > 15
    for box in (controls.nth(i).bounding_box() for i in range(controls.count())):
        assert box is not None
        assert box["x"] >= 0 and box["y"] >= 0
        assert box["x"] + box["width"] <= 1280 and box["y"] + box["height"] <= 720, box


def test_discloses_untrained_models_and_member_count(page):
    expect(page.locator("#weights")).to_be_visible()
    expect(page.locator("#weights")).to_contain_text("Untrained")
    expect(page.locator("#members")).to_contain_text("2 models")


def test_touch_draws(browser, url):
    context = browser.new_context(viewport={"width": 1280, "height": 720}, has_touch=True)
    opened = context.new_page()
    opened.goto(url)
    _wait_for_frame(opened)
    _send(opened, {"type": "preset", "name": "empty"})
    _send(opened, {"type": "mode", "mode": "ai"})
    _wait_for_empty(opened)
    x, y = _centre(opened)
    opened.touchscreen.tap(x, y)
    assert _wait_for_solid(opened) > 0
    context.close()


def test_keyboard_reaches_controls_and_draws(page):
    _wait_for_empty(page)
    page.locator("#smoke").focus()
    for _ in range(3):
        page.keyboard.press("ArrowRight")
    page.keyboard.press("Space")
    assert _wait_for_solid(page) > 0

    # Every control takes the focus, and shows that it has it.
    page.get_by_role("button", name="Reset").focus()
    outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
    assert outline != "none"
