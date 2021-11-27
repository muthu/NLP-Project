VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(VENV)/bin/activate
	$(PYTHON) source.py

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt


clean:
	rm -rf __pycache__
	rm -rf $(VENV)
