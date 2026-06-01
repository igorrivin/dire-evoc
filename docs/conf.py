"""Sphinx configuration for dire-evoc."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dire_evoc  # noqa: E402

project = "dire-evoc"
author = "Igor Rivin"
copyright = "2026, Igor Rivin"
release = dire_evoc.__version__
version = dire_evoc.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = True

html_title = "dire-evoc documentation"
html_static_path = ["_static"]
