Last updated: 2026-03-29T00:00:00Z

# Docs Publishing TODO

Status: `in_progress`

## Scope

This file tracks the production docs publishing layer that sits on top of the
existing Markdown-first docs workflow.

## Landed Scaffolding

- `docs/conf.py`
- `docs/README.md` as the docs-root landing page
- `docs/Makefile` and `docs/make.bat`
- Sphinx/MyST HTML build lane scaffold
- LuaLaTeX PDF workflow scaffold
- docs build CI scaffold
- build-time rewrite/preparation layer for repo-root Markdown links

## Remaining Hardening

- tighten internal cross-reference coverage and reduce build warnings
- decide which notebook subset should be in the published nav by default
- harden linkcheck expectations for GitHub-hosted source links
- decide whether PDF becomes release-attached or artifact-only
