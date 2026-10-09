"""Sphinx configuration for the unified DiFfRG documentation site.

The site bundles three languages under one theme:

* **C++**   — Doxygen renders the API to HTML; ``doxysphinx`` wraps that HTML
  into this Sphinx tree (see ``build.sh``). Output lives in ``doxygen/`` and is
  generated, not committed.
* **Python** — ``sphinx-autoapi`` parses the package source statically.
* **Mathematica** — ``gen_mathematica.py`` extracts ``::usage`` strings into the
  ``_generated_mathematica/`` MyST pages (generated, not committed).
"""

from pathlib import Path

HERE = Path(__file__).parent.resolve()
DIFFRG_DIR = HERE.parent  # the DiFfRG/ library subfolder

# -- Project information ------------------------------------------------------

project = "DiFfRG"
author = "Franz R. Sattler"
copyright = "2025, Franz R. Sattler"


def _read_version() -> str:
    """Read the single source of truth for the version (repo-root VERSION file)."""
    for candidate in (DIFFRG_DIR.parent / "VERSION", DIFFRG_DIR / "VERSION"):
        try:
            return candidate.read_text().strip()
        except OSError:
            continue
    return "unknown"


release = _read_version()
version = release

# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "autoapi.extension",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"

# Do not let Sphinx read the README symlink, the bundled theme, the Doxygen
# config, or build artefacts. The generated doxygen/ rst (from doxysphinx) and
# _generated_mathematica/ pages MUST stay visible, so they are not excluded.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "main.md",
    "toc.md",
    "doxygen-awesome-css",
    "requirements.txt",
    # Never scan a virtualenv that happens to live in the source tree.
    ".venv*",
    "**/site-packages/**",
]

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = [
    "dollarmath",   # $...$ and $$...$$
    "amsmath",      # \begin{aligned} ... \end{aligned}
    "deflist",
    "colon_fence",
    "attrs_inline",
    "fieldlist",
    "html_image",
]
myst_heading_anchors = 3

# Pygments cannot strictly lex the Wolfram-Language signatures extracted from
# ::usage strings (patterns with ':', descriptions on the signature line) or the
# elided JSON snippets in the tutorials; the highlighter retries in relaxed mode
# and renders fine, so silence the cosmetic failures.
suppress_warnings = ["misc.highlighting_failure"]

# -- AutoAPI (Python) ---------------------------------------------------------
# Static source parsing (astroid) — does not import the package, so heavy
# runtime deps (vtk, seaborn, jupyter, ...) need not be installed to build docs.

autoapi_type = "python"
autoapi_dirs = [str(DIFFRG_DIR / "python" / "DiFfRG")]
autoapi_root = "autoapi"
autoapi_add_toctree_entry = False  # mounted manually under the Python tab
autoapi_keep_files = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML output --------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "DiFfRG"
html_show_sourcelink = False  # drop the "Show Source" link

html_static_path = ["_static"]
# Make the embedded Doxygen (doxygen-awesome) pages follow the site's dark theme.
html_css_files = ["doxygen-darkmode.css"]

# The pydata theme builds the top navbar from the root document's top-level
# toctree entries — these become the C++ / Python / Mathematica tabs.
html_theme_options = {
    "github_url": "https://github.com/satfra/DiFfRG",
    "navbar_align": "left",
    # Auto-expand the left ("section navigation") sidebar two levels deep so the
    # API trees aren't a single collapsed entry.
    "show_nav_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "use_edit_page_button": False,
    # Single sidebar: fold the in-page "On this page" TOC away (no right column).
    "secondary_sidebar_items": [],
    "icon_links": [
        {
            "name": "arXiv",
            "url": "https://arxiv.org/abs/2412.13043",
            "icon": "fa-solid fa-book",
        },
    ],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
