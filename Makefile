test:
	cd tests/unit && python -m pytest . -s --disable-warnings

integration-test:
	cd tests/integration && python -m pytest . -s --disable-warnings

lint:
	python -m pylint deepface/ --fail-under=10 && mypy deepface/

coverage:
	pip install pytest-cov && cd tests/unit && python -m pytest --cov=deepface