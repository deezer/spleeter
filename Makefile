# =======================================================
# Build script for distribution packaging.
#
# @author Deezer Research <research@deezer.com>
# @licence MIT Licence
# =======================================================

clean:
	rm -Rf *.egg-info
	rm -Rf dist

build:
	@echo "=== Build CPU bdist package" 
	@python3 setup.py sdist
	@echo "=== CPU version checksum"
	@openssl sha256 dist/*.tar.gz

build-gpu:
	@echo "=== Build GPU bdist package" 
	@python3 setup.py sdist --target gpu
	@echo "=== GPU version checksum"
	@openssl sha256 dist/*.tar.gz

upload:
	twine upload dist/*

test-upload:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

all: clean build build-gpu upload