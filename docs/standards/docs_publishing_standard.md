Last updated: 2026-03-29T00:00:00Z

# Docs Publishing Standard

Status: active

## Purpose

This document defines the production publishing layer for repository
documentation.

arbPlusJAX keeps Markdown in `docs/` as the source-of-truth authoring surface.
The publishing layer is an additional build/output system, not a replacement
for the current development workflow.

## Required Model

The docs publishing stack should preserve:

- Markdown-first authoring in `docs/`
- generated indexes and governed repo-root link policy
- separate publication/manuscript work under `papers/`

The publish layer should add:

- HTML site output
- PDF output
- build CI
- link and cross-reference validation

## Required Outputs

The intended production outputs are:

- Sphinx/MyST HTML site for GitHub Pages
- LuaLaTeX PDF build for governed long-form docs output

## Required Tooling Structure

The repo should provide:

- `docs/README.md` as the docs-root landing page
- `docs/conf.py`
- `docs/Makefile` and `docs/make.bat`
- docs build configuration
- isolated docs dependency group
- a Markdown-preserving prebuild rewrite layer
- a docs build workflow
- a link/crossref validation lane
- a Pages deployment workflow or equivalent scaffold

## Authoring Preservation Rule

The repository must not require authors to abandon the governed Markdown-first
workflow in order to publish documentation.

If a build-time rewrite or adapter layer is needed for Sphinx/MyST, it belongs
in the publishing toolchain rather than in ad hoc manual authoring changes.

The build layer should therefore:

- preserve repo-authored `README.md` landing pages instead of forcing
  `index.md`
- use `docs/README.md` as the docs-root source page
- rewrite repo-root Markdown links in a temporary prepared source tree rather
  than mutating authored docs in place

## Papers Rule

Publication-style manuscripts remain under `papers/`.

The docs publish system may render long-form governed docs, but it does not
replace the separate LaTeX manuscript lane.

## Required Evidence

The repo is compliant only if it has:

- a governed docs publishing standard
- a governed docs build standard
- a docs build workflow scaffold or implementation
- a docs dependency group
- a place for HTML and PDF build configuration
- a documented relationship between `docs/` and `papers/`
