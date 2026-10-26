# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown>=3.7", "playwright>=1.48"]
# ///
"""Render report/report.md to report/report.pdf in the official template's layout.

Matches report-template.pdf: US Letter, 1-inch margins, Old Standard TT 11pt body, 14pt section headings.
Run from the repo root: `uv run scripts/render_report.py` (once first: `uvx playwright install chromium`).
"""

import re
import tempfile
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

SOURCE = Path("report/report.md")
OUTPUT = Path("report/report.pdf")
PAGE_MARGIN = "1in"
FOOTNOTE = re.compile(r'<div class="first-page-footnote">(.*?)</div>\n?', re.S)
FONT_LOAD_MS = 3000  # Google Fonts must load before printing, or Chromium falls back to Times

CSS = f"""
@import url(https://fonts.googleapis.com/css2?family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&display=block);
body {{ font-family: 'Old Standard TT'; font-size: 11pt; line-height: 1.15; margin: 0; }}
h1 {{ font-size: 20pt; }}
h2 {{ font-size: 14pt; }}
h3 {{ font-size: 12pt; }}
.title-block {{ text-align: center; }}
img {{ max-width: 100%; }}
table {{ border-collapse: collapse; font-size: 9pt; margin: 6pt 0; }}
td, th {{ border: 0.5pt solid #999; padding: 2px 4px; }}
blockquote {{ border: 0.5pt solid #888; padding: 4px 10px; margin: 8px 0; }}
pre {{ white-space: pre-wrap; font-size: 8pt; }}
table, img, blockquote, li {{ break-inside: avoid; }}
h1, h2, h3, img {{ break-after: avoid; }}
"""


def render(source: Path, output: Path) -> None:
    text = source.read_text()
    # The first-page footnote goes in page 1's bottom margin, so it can't overlap the body text
    footnote = FOOTNOTE.search(text)
    footnote_css = ""
    if footnote:
        note = re.sub(r"<sup>(.*?)</sup>\s*", r"\1 ", footnote.group(1)).replace('"', "'")
        footnote_css = f'@page :first {{ @bottom-left {{ content: "{note}"; font-family: "Old Standard TT"; font-size: 9pt; vertical-align: top; }} }}'
        text = FOOTNOTE.sub("", text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "md_in_html"])
    page = f"<html><head><meta charset=utf-8><base href='{source.parent.resolve().as_uri()}/'><style>{CSS}{footnote_css}</style></head><body>{body}</body></html>"
    with tempfile.NamedTemporaryFile("w", suffix=".html") as html, sync_playwright() as playwright:
        _ = html.write(page)
        html.flush()
        browser = playwright.chromium.launch()
        tab = browser.new_page()
        _ = tab.goto(Path(html.name).as_uri())
        tab.wait_for_timeout(FONT_LOAD_MS)
        margin = {side: PAGE_MARGIN for side in ("top", "bottom", "left", "right")}
        _ = tab.pdf(path=str(output), format="Letter", margin=margin)
        browser.close()


render(SOURCE, OUTPUT)
print(f"wrote {OUTPUT}")
