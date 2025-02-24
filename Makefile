install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv tests/test_*.py

format:
	black library/*.py tests/*.py *.py

lint:
	pylint --disable=R,C library/*.py tests/*.py *.py

all: install lint test