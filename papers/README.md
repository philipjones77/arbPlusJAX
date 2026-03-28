# Papers

This folder is the repo-root home for long-form LaTeX paper projects and other
publication-grade manuscript builds.

Use `papers/` for:

- standalone LaTeX paper sources
- manuscript-specific figures, bibliographies, and build files
- appendix-grade derivations or writeups that need paper-style structure rather
  than docs-tree structure

Do not use `papers/` for:

- the main documentation tree
- repo standards, status, or reports
- ordinary implementation notes
- notebook outputs or benchmark artifacts

Relationship to the docs tree:

- `docs/` remains the Markdown-first source of truth for repo documentation
- `papers/` is the publication lane for long-form manuscript builds
- paper text may reference or derive from `docs/theory/`, but it should not
  replace the docs-tree authority structure

Typical future structure:

- `papers/<paper-name>/main.tex`
- `papers/<paper-name>/sections/`
- `papers/<paper-name>/figures/`
- `papers/<paper-name>/bib/`
- `papers/<paper-name>/Makefile` or build helper

The intended workflow is:

1. develop theory, standards, implementation notes, and reports in `docs/`
2. stabilize the public/runtime surface
3. build publication-facing manuscripts in `papers/` from that stabilized base

This is the intended correct structure for this repo, not an experimental
alternative:

- `docs/` owns the active Markdown-first documentation workflow
- `papers/` owns publication-grade LaTeX manuscript workflows
