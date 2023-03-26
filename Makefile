typecheck:
	ruff check visual_openllm
	mypy -p visual_openllm \
		--ignore-missing-imports \
		--warn-unreachable \
		--implicit-optional \
		--allow-redefinition \
		--disable-error-code abstract

format-check:
	black --check .

format:
	black .

build: all
	rm -fr dist
	poetry build -f sdist

all: format-check typecheck
