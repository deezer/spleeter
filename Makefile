# =======================================================
# Library lifecycle management.
#
# @author Deezer Research <research@deezer.com>
# @licence MIT Licence
# =======================================================

FFEDSTOCK = spleeter-feedstock
FEEDSTOCK_REPOSITORY = https://github.com/deezer/$(FEEDSTOCK)
FEEDSTOCK_RECIPE = $(FEEDSTOCK)/recipe/spleeter/meta.yaml

all: clean build test deploy

clean:
	rm -Rf *.egg-info
	rm -Rf dist

build:
	python3 setup.py sdist

test:
	pytest -W ignore::FutureWarning -W ignore::DeprecationWarning -vv --forked

feedstock: build
	$(eval VERSION = $(shell grep 'project_version = ' setup.py | cut -d' ' -f3 | sed "s/'//g"))
	$(eval CHECKSUM = $(shell openssl sha256 dist/spleeter-$(VERSION).tar.gz | cut -d' ' -f2))
	git clone $(FEEDSTOCK_REPOSITORY)
	sed 's/{% set version = "[0-9]*\.[0-9]*\.[0-9]*" %}/{% set version = "$(VERSION)" %}/g' $(FEEDSTOCK_RECIPE)
	sed 's/sha256: [0-9a-z]*/sha: $(CHECKSUM)/g' $(FEEDSTOCK_RECIPE)
	git config credential.helper 'cache --timeout=120'
	git config user.email "research@deezer.com"
	git config user.name "spleeter-ci"
	git add recipe/spleeter/meta.yaml
	git commit --allow-empty -m "feat: update spleeter version from CI"
	git push -q https://$$FEEDSTOCK_TOKEN@github.com/deezer/$(FEEDSTOCK)
	hub pull-request -m "Update spleeter version to $(VERSION)"

deploy:
	pip install twine
	twine upload --skip-existing dist/*