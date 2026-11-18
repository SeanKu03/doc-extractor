PY ?= .venv/bin/python

.PHONY: setup eval test

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

eval:
	$(PY) eval.py

test:
	$(PY) -m pytest -q
