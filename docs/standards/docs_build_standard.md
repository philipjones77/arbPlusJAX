Last updated: 2026-03-29T00:00:00Z

# Docs Build Standard

Status: active

## Purpose

This document defines the canonical build model for repository documentation.

It is narrower than general documentation governance and narrower than release
publishing. It owns the source/build/rewrite model that turns the Markdown
documentation tree into a published documentation site and PDF output.

## Required Model

The required documentation build model is:

- `docs/` is the Markdown-first source of truth
- `docs/README.md` is the docs-root landing page
- Sphinx + MyST is the canonical render/build layer
- GitHub Pages is the canonical HTML host
- Sphinx LuaLaTeX output is the canonical PDF lane
- `papers/` remains the separate manuscript lane for formal paper projects

The repository must not create a second parallel documentation corpus just to
support publishing.

## README Rule

The repository prefers folder `README.md` files over `index.md` files as the
human-facing landing pattern.

Therefore:

- `docs/README.md` is the docs-root source document
- generated section landing pages remain `README.md`
- Sphinx configuration should use `README` as the root document

If Sphinx or another renderer wants an index-like root, that adaptation belongs
in configuration, not in author-facing file naming.

## Required Toolchain

The repo should provide:

- `docs/conf.py`
- `docs/_static/`
- `docs/_templates/`
- standard Sphinx entrypoints such as `docs/Makefile` and `docs/make.bat`
- a preprocessing step that prepares a publication-safe source tree
- a build helper that can produce HTML and PDF outputs reproducibly

## Link Rewrite Rule

The authored docs tree may continue to use repo-root links such as:

- `/docs/...`
- `/src/...`
- `/tests/...`
- `/benchmarks/...`
- `/examples/...`

The publishing layer must adapt those links during the build process rather
than requiring authors to hand-maintain a separate publication-specific link
style.

That means:

- internal `docs -> docs` links should become Sphinx/MyST-resolvable links in
  the prepared source tree
- repo links to source, tests, benchmarks, examples, tools, contracts, papers,
  and similar non-docs folders should become GitHub blob/tree links or another
  explicitly governed published target
- the rewrite should happen in a temporary build source tree, not in-place in
  the authored Markdown tree

## Notebook Rule

Canonical checked-in notebooks are first-class publication artifacts.

The docs build layer should support notebook rendering through MyST notebook
support, but heavy executed notebook outputs may remain retained artifacts
rather than becoming the primary site source.

## Output Rule

Required output kinds:

- HTML site build
- LuaLaTeX PDF build

Preferred build commands:

- HTML: `sphinx-build -b html ...`
- PDF: `sphinx-build -M latexpdf ...`

## Validation Rule

The documentation build system is compliant only if it can validate:

- docs source preparation succeeds
- the HTML docs build succeeds
- internal doc links resolve under the prepared build tree
- generated docs section `README.md` files are current
- prepared output does not retain unresolved repo-root `/docs/...` links

## Related Standards

- [documentation_governance.md](/docs/governance/documentation_governance.md)
- [docs_publishing_standard.md](/docs/standards/docs_publishing_standard.md)
- [generated_documentation_standard.md](/docs/standards/generated_documentation_standard.md)
- [repo_standards.md](/docs/standards/repo_standards.md)
