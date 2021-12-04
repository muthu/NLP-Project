VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt
	$(PYTHON) source.py
	$(PYTHON) Data_Clean.py
	$(PYTHON) naive_bayes.py
	$(PYTHON) rnn_model.py
	$(PYTHON) transfer_learning.py

$(VENV)/bin/activate:
	python3 -m venv $(VENV)




clean:
	rm -rf __pycache__
	rm -rf $(VENV)
