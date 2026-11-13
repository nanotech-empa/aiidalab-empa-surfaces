"""Every maintained page must opt out of nested notebook output scrolling."""

import ast
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = subprocess.check_output(
    ["git", "-C", str(ROOT), "ls-files", "*.ipynb"], text=True
).splitlines()


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_every_canonical_notebook_applies_page_layout_outside_capture(notebook):
    cells = json.loads((ROOT / notebook).read_text())["cells"]
    code = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    guard = (
        "from surfaces_tools.page_layout import apply_page_layout\napply_page_layout()"
    )
    assert code and ast.dump(ast.parse(code[0])) == ast.dump(ast.parse(guard)), notebook
    assert sum("apply_page_layout()" in cell for cell in code) == 1
    assert not code[0].startswith("%%capture")


def test_long_output_scrolls_the_page_even_after_widget_growth(monkeypatch):
    """Real Chromium checks the saved output_scroll state and widget resizing."""
    playwright = pytest.importorskip("playwright.sync_api")
    from surfaces_tools import page_layout

    rendered = []
    monkeypatch.setattr(page_layout, "display", rendered.append)
    assert page_layout.apply_page_layout() is None
    style = rendered[0].data
    with playwright.sync_playwright() as manager:
        try:
            browser = manager.chromium.launch(
                executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"),
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
        except playwright.Error as error:
            if "Executable doesn't exist" in str(error):
                pytest.skip("Chromium is not installed; run the maintained GUI suite")
            raise
        try:
            page = browser.new_page(viewport={"width": 900, "height": 400})
            page.set_content("""
                <style>
                body {margin:0}
                #site {height:100vh;overflow:auto}
                .output_scroll {height:100px;max-height:100px;overflow:auto}
                .widget-output {height:150px;max-height:150px;overflow:auto}
                textarea {height:50px;overflow:auto}
                </style>
                <main id="site"><div id="notebook">
                <div class="output_scroll"><div class="widget-output">
                <div id="rows" style="height:2000px">Long search results</div>
                </div></div>
                <textarea>Input control</textarea>
                </div></main>
                """)
            assert page.locator(".output_scroll").evaluate(
                "e => e.scrollHeight > e.clientHeight"
            )
            page.locator("body").evaluate(
                "(e, style) => e.insertAdjacentHTML('beforeend', style)", style
            )
            for height in (2000, 5000):
                page.locator("#rows").evaluate("(e,h) => e.style.height=h+'px'", height)
                for selector in (".output_scroll", ".widget-output"):
                    assert page.locator(selector).evaluate("""e => {
                        const s=getComputedStyle(e);
                        return s.overflowY === 'visible' && s.maxHeight === 'none'
                            && e.scrollHeight <= e.clientHeight + 1;
                    }""")
                assert page.locator("#site").evaluate(
                    "e => e.scrollHeight > e.clientHeight"
                )
            assert (
                page.locator("textarea").evaluate("e => getComputedStyle(e).overflowY")
                == "auto"
            )
            page.mouse.move(200, 200)
            page.mouse.wheel(0, 600)
            page.wait_for_timeout(150)
            assert page.locator("#site").evaluate("e => e.scrollTop") > 0
            assert page.locator(".output_scroll").evaluate("e => e.scrollTop") == 0
        finally:
            browser.close()
