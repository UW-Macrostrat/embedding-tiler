from mapbox_vector_tile import decode, encode
from .utils import timer
from .text_pipeline import preprocess_text, rank_polygons
from sentence_transformers import SentenceTransformer
from geopandas import GeoDataFrame
from contextvars import ContextVar
from os import environ

# Note: for now, the model is downloaded from the Hugging Face model hub
# on the first run. This will take a while. We will pre-load the model
# in the future.

models = {
    'BAAI/bge-base-en-v1.5': 'A standard model trained on the BGE dataset',
    'iaross/cm_bert': 'A model trained on a corpus of mining reports from xDD'
}

# Get  PyTorch device
pytorch_device = environ.get("PYTORCH_DEVICE", "cpu")


def create_embed_models():
    model_index = {}
    for model in models:
        embed_model = SentenceTransformer(model, device=pytorch_device)
        model_index[model] = embed_model
    return model_index


embed_models = ContextVar("embed_models", default=create_embed_models())


def get_model(model_name='BAAI/bge-base-en-v1.5'):
    try:
        return embed_models.get()[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} not found. Available models are: {models.keys()}")


@timer("Process tile")
def process_vector_tile(content: bytes, term: str, model=None):
    """
    Decode a vector tile and return the features.
    """
    tile = decode(content)

    df = get_data_frame(tile)
    data = preprocess_text(df, ['name', 'age', 'lith', 'descrip', 'comments'])

    if model is None:
        model = 'BAAI/bge-base-en-v1.5'

    _embed_model = get_model(model)

    gpd_data = rank_polygons(term, _embed_model, data)

    tile["units"]["features"] = get_geojson(gpd_data)

    layers = create_layer_list(tile)
    return encode(layers)


async def process_vector_tile_async(loop, content, term, model=None):
    # None uses the default executor (ThreadPoolExecutor)
    return await loop.run_in_executor(None, process_vector_tile, content, term, model)


def create_layer_list(tile):
    layers = []
    for key, layer in tile.items():
        layer["name"] = key
        layers.append(layer)
    return layers


def get_data_frame(tile):
    return GeoDataFrame.from_features(tile['units']['features'])


def get_geojson(df):
    return list(ensure_geojson(df.to_dict('records')))


def ensure_geojson(data):
    for feature in data:
        yield {
            'type': 'Feature',
            'geometry': feature.pop('geometry'),
            'properties': feature.pop('properties', feature)
        }
