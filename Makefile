
lint:
	export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
	poetry run isort .
	poetry run black . --config pyproject.toml
	poetry run ruff check . --fix
	poetry run mypy .
