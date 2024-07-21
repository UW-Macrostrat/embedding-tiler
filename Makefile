all:
	poetry run fastapi dev macrostrat/embedding_tiler

test:
	poetry run pytest tests
