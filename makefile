define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

help:
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)



init: ## sets up environment and installs requirements
init:
	pip install -r requirements.txt

install: ## Installs development requirments
install:
	python -m pip install --upgrade pip
	# Used for packaging and publishing
	pip install setuptools wheel twine build
	# Used for linting
	pip install black
	# Used for testing
	pip install pytest
	# For testing
	pip install tox
	# For documentation

lint: ## Runs black on src and test, exit if critical rules are broken
lint:
	black ./torchmimic
	black ./tests

package-test: ## Package and test all 
package-test: package test-all

package: ## Create package in dist
package: clean
	python -m build
	python setup.py sdist bdist_wheel

upload-test: ## Create package and upload to test.pypi
upload-test: package
	python -m twine upload -r testpypi dist/* 

upload: ## Create package and upload to pypi
upload: package
	python -m twine upload dist/* --non-interactive

clean: ## Remove build and cache files
clean:
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	rm -rf .pytest_cache
	# Remove all pycache
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

document: # build documentation
document:
	sphinx-build -b html ./docs/source/ ./docs/build/html

test-all:
	tox