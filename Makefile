test:
	cd tests && python -m pytest . -s --disable-warnings

lint:
	python -m pylint deepface/ --fail-under=10 && mypy deepface/

coverage:
	pip install pytest-cov && cd tests && python -m pytest --cov=deepface