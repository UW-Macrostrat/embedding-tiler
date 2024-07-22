all:
	poetry run fastapi dev macrostrat/embedding_tiler

test:
	poetry run pytest tests

models:
	# Huggingface model downloads
	poetry run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('baai/bge-base-en-v1.5')"
	poetry run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('iaross/cm_bert')"
	# Models needed for nrcan_p2 preprocessing
	#poetry run spacy download en_core_web_sm
	#poetry run spacy download en_core_web_lg

install:
	# Install necessary models for pre-training
	brew install enchant
