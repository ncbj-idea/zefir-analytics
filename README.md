# Zefir Analytics

## Setup Development Environment

### Make setup

```bash
# Creating Project Environment
make --version

sudo apt install make # if make is not installed

make install
```

### Manual setup

```bash
# Create and source virtual Environment
python -m venv .venv
source .venv/bin/active

# Install all requirements and dependencies
pip install -r requirements.txt -i https://nexus.services.idea.edu.pl/repository/pypi-all/simple
pip install -r requirements-dev.txt -i https://nexus.services.idea.edu.pl/repository/pypi-all/simple

# Init pre-commit hook
pre-commit install
```

### Linters
```bash
# using make
make lint

# without make
source .venv/bin/active
pylama zefir_analytics tests --async
```

### Tests
```bash
# using make
make test

# without make
source .venv/bin/active
pytest -vvv tests
```
