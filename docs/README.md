Last updated: 2026-03-29T00:00:00Z

# Documentation

This folder is the Markdown-first source of truth for active repo documentation.
It is authored and maintained directly in `docs/`.

When the publication layer is used, Sphinx + MyST renders this same folder into HTML and PDF output.
That publish layer must adapt to this docs tree rather than requiring a second parallel documentation corpus.

## Contents

- overview: [project_overview.md](/docs/project_overview.md)
- governance: [README.md](/docs/governance/README.md)
- standards: [README.md](/docs/standards/README.md)
- theory: [README.md](/docs/theory/README.md)
- practical: [README.md](/docs/practical/README.md)
- examples: [README.md](/examples/README.md)
- implementation: [README.md](/docs/implementation/README.md)
- reports: [README.md](/docs/reports/README.md)
- status: [README.md](/docs/status/README.md)
- objects: [README.md](/docs/objects/README.md)
- specs: [README.md](/docs/specs/README.md)
- notation: [README.md](/docs/notation/README.md)

## Build Model

- source of truth: `docs/`
- HTML renderer: Sphinx + MyST
- HTML host: GitHub Pages
- PDF renderer: Sphinx `latexpdf`/LuaLaTeX lane
- publication manuscripts: [papers/README.md](/papers/README.md)

## Site Navigation

The publication layer uses this `README.md` as the docs root document.

```{toctree}
:maxdepth: 2
:hidden:

project_overview
governance/README
standards/README
theory/README
practical/README
examples/README
implementation/README
reports/README
status/README
objects/README
specs/README
notation/README
```

This file is generated and should be refreshed through `python tools/generate_docs_indexes.py` before commit/push.
