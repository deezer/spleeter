# =======================================================
# Library lifecycle management.
#
# @author Deezer Research <spleeter@deezer.com>
# @licence MIT Licence
# =======================================================

FEEDSTOCK = spleeter-feedstock
FEEDSTOCK_REPOSITORY = https://github.com/deezer/$(FEEDSTOCK)
FEEDSTOCK_RECIPE = $(FEEDSTOCK)/recipe/spleeter/meta.yaml
PYTEST_CMD = pytest -W ignore::FutureWarning -W ignore::DeprecationWarning -vv --forked

all: clean build test deploy

clean:
	rm -Rf *.egg-info
	rm -Rf dist

build: clean
	sed -i "s/project_name = '[^']*'/project_name = 'spleeter'/g" setup.py
	sed -i "s/tensorflow_dependency = '[^']*'/tensorflow_dependency = 'tensorflow'/g" setup.py
	python3 setup.py sdist

build-gpu: clean
	sed -i "s/project_name = '[^']*'/project_name = 'spleeter-gpu'/g" setup.py
	sed -i "s/tensorflow_dependency = '[^']*'/tensorflow_dependency = 'tensorflow-gpu'/g" setup.py
	python3 setup.py sdist

test:
	$(PYTEST_CMD) tests/

deploy:
	pip install twine
	twine upload --skip-existing dist/*
