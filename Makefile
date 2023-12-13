# Sprawdzenie systemu operacyjnego
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE := .venv\Scripts\activate
else
    VENV_ACTIVATE := .venv/bin/activate
endif
PIP_INDEX := https://nexus.services.idea.edu.pl/repository/pypi-all/simple

.PHONY: install lint unit test clean

$(VENV_ACTIVATE): requirements.txt requirements-dev.txt .pre-commit-config.yaml
	python3.11 -m venv .venv
	. $(VENV_ACTIVATE) && pip install --upgrade pip \
		&& pip install -U -r requirements.txt -i $(PIP_INDEX) \
		&& pip install -U -r requirements-dev.txt -i $(PIP_INDEX)
	. $(VENV_ACTIVATE) && pre-commit install

install: $(VENV_ACTIVATE)

lint: $(VENV_ACTIVATE)
	. $(VENV_ACTIVATE) && black . && pylama -l mccabe,pycodestyle,pyflakes,radon,mypy zefir_analytics tests --async

unit: $(VENV_ACTIVATE) lint
	. $(VENV_ACTIVATE) && pytest -vvv tests/unit && tox -e fast_integration --skip-pkg-install

test: $(VENV_ACTIVATE) lint unit
	. $(VENV_ACTIVATE) && tox -e integration --skip-pkg-install

clean:
	rm -rf $(VENV_ACTIVATE) .mypy_cache .pytest_cache .tox
	find . | grep -E "(/__pycache__$$|\.pyc$|\.pyo$$)" | xargs rm -rf
