from bs4 import BeautifulSoup
from pathlib import Path

html_file = Path("test.html")

# Read the exported HTML
if not html_file.exists():
    raise FileNotFoundError(f"{html_file} not found")

html_content = html_file.read_text()
soup = BeautifulSoup(html_content, "html.parser")

# Match both JupyterLab and classic templates
selectors = [".jp-Cell-inputWrapper", ".input"]

# Wrap each matched code cell in a <details><summary>...</summary></details>
for selector in selectors:
    for code_cell in soup.select(selector):
        # Skip if this cell contains heading tags
        if code_cell.find(["h1", "h2", "h3", "h4", "h5", "h6"]):
            continue
        details = soup.new_tag("details")
        summary = soup.new_tag("summary")
        summary.string = ""
        details.append(summary)
        code_cell.wrap(details)

# Write back modified HTML
html_file.write_text(str(soup))
print(f"✅ Per-cell toggle wrappers (no global button) injected into {html_file}")