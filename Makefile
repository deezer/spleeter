# =======================================================
# Library lifecycle management.
#
# @author Deezer Research <research@deezer.com>
# @licence MIT Licence
# =======================================================

FEEDSTOCK = spleeter-feedstock
FEEDSTOCK_REPOSITORY = https://github.com/deezer/$(FEEDSTOCK)
FEEDSTOCK_RECIPE = $(FEEDSTOCK)/recipe/spleeter/meta.yaml
PYTEST_CMD = pytest -W ignore::FutureWarning -W ignore::DeprecationWarning -vv 

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
	#$(foreach file, $(wildcard tests/test_*.py), $(PYTEST_CMD) $(file);)
	#$(PYTEST_CMD) tests/test_eval.py
	$(PYTEST_CMD) tests/test_ffmpeg_adapter.py
	$(PYTEST_CMD) tests/test_github_model_provider.py
	$(PYTEST_CMD) --boxed tests/test_separator.py
	
	

deploy:
	pip install twine
	twine upload --skip-existing dist/*
