# =======================================================
# Library lifecycle management.
#
# @author Deezer Research <research@deezer.com>
# @licence MIT Licence
# =======================================================

clean:
	rm -Rf *.egg-info
	rm -Rf dist


build:
	@echo "=== Build CPU bdist package" 
	python3 setup.py sdist
	@echo "=== CPU version checksum"
	@openssl sha256 dist/*.tar.gz

build-gpu:
	@echo "=== Build GPU bdist package" 
	python3 setup.py sdist --target gpu
	@echo "=== GPU version checksum"
	@openssl sha256 dist/*.tar.gz

pip-dependencies:
	pip install twine

test: pip-dependencies
	pytest -W ignore::FutureWarning -W ignore::DeprecationWarning -vv --forked

test-distribution: pip-dependencies
	bash tests/test_pypi_sdist.sh

deploy: pip-dependencies
	twine upload dist/*

all: clean test build build-gpu upload
