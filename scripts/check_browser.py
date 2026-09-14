"""Check the real web UI with Chromium; optionally save documentation screenshots."""

import argparse
import io
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def check(base_url, screenshots):
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1200}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base_url)
        page.locator("#demoBtn").click()
        expect(page.locator("#resultMode")).to_have_text("ДЕМОНСТРАЦИЯ")
        expect(page.locator(".result-card")).to_have_count(2)
        expect(page.locator("#demoNotice")).to_be_visible()
        assert page.locator("#resultImage").evaluate(
            "(img) => img.complete && img.naturalWidth > 0"
        )
        if screenshots:
            screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshots / "desktop-demo.png"), full_page=True)
        with page.expect_download() as download_info:
            page.locator("#downloadBtn").click()
        download = download_info.value
        payload = json.loads(Path(download.path()).read_text())
        assert payload["mode"] == "demo" and "image_base64" not in payload
        for width in [390, 320]:
            page.set_viewport_size({"width": width, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                f"Overflow: {width}px"
            )
            if screenshots and width == 390:
                page.screenshot(path=str(screenshots / "mobile-demo.png"), full_page=True)
        page.set_viewport_size({"width": 1280, "height": 1000})
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), "#668877").save(buffer, format="PNG")
        page.locator("#imageFile").set_input_files(
            {"name": "synthetic.png", "mimeType": "image/png", "buffer": buffer.getvalue()}
        )
        expect(page.locator("#resultImage")).to_be_hidden()
        page.locator("#runBtn").click()
        expect(page.locator("#errorBox")).to_contain_text("Добавьте файлы моделей")
        expect(page.locator("#runBtn")).to_be_enabled()
        page.locator("#demoBtn").click()
        expect(page.locator(".result-card")).to_have_count(2)
        expect(page.locator("#errorBox")).to_be_hidden()
        assert not errors, errors
        browser.close()
    print(
        "PASS: desktop/mobile layout, demo, JSON download, upload error and recovery; no JS errors."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("Web server exited before startup")
            try:
                urllib.request.urlopen(url + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("Web server did not start")
        check(url, args.screenshots)
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
