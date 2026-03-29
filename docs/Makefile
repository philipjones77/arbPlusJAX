# Minimal Sphinx makefile that preserves the Markdown-first docs workflow.

SPHINXBUILD ?= sphinx-build
PYTHON ?= python
PREPARE ?= $(PYTHON) ../tools/prepare_sphinx_docs.py
BUILDER ?= $(PYTHON) ../tools/build_docs.py

.PHONY: help prep html linkcheck latexpdf clean

help:
	@echo "make prep       Prepare the temporary Sphinx source tree"
	@echo "make html       Build the HTML documentation site"
	@echo "make linkcheck  Build with Sphinx linkcheck"
	@echo "make latexpdf   Build the LuaLaTeX PDF output"
	@echo "make clean      Remove generated docs build outputs"

prep:
	$(PREPARE)

html:
	$(BUILDER) --builder html

linkcheck:
	$(BUILDER) --builder linkcheck

latexpdf:
	$(BUILDER) --builder latexpdf

clean:
	rm -rf ../outputs/docs_site
