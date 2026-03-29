from __future__ import annotations

import os
import sys
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

project = "arbPlusJAX"
author = "arbPlusJAX contributors"
release = "1.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

templates_path = [str(DOCS_DIR / "_templates")]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
root_doc = "README"

html_theme = "sphinx_book_theme"
html_title = "arbPlusJAX"
html_static_path = [str(DOCS_DIR / "_static")]
html_baseurl = os.environ.get("DOCS_HTML_BASEURL", "")
html_theme_options = {
    "repository_url": os.environ.get(
        "DOCS_REPOSITORY_URL",
        "https://github.com/philipturner/arbplusJAX",
    ),
    "use_repository_button": True,
    "use_issues_button": False,
    "use_edit_page_button": False,
    "home_page_in_toc": True,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]
myst_heading_anchors = 3

nb_execution_mode = "off"
nb_render_markdown_format = "myst"

latex_engine = "lualatex"
latex_documents = [
    (
        "README",
        "arbplusjax-docs.tex",
        "arbPlusJAX Documentation",
        author,
        "manual",
    )
]
