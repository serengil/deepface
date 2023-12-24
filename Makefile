test:
	cd tests && python -m pytest . -s --disable-warnings

lint:
	python -m pylint deepface/ --fail-under=10