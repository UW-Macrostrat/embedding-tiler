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

hf_model = 'BAAI/bge-base-en-v1.5'

# Get  PyTorch device
pytorch_device = environ.get("PYTORCH_DEVICE", "cpu")

embed_model = ContextVar("embed_model", default=SentenceTransformer(hf_model, device=pytorch_device))


@timer("Process tile")
def process_vector_tile(content: bytes):
    """
    Decode a vector tile and return the features.
    """
    tile = decode(content)

    df = get_data_frame(tile)
    data = preprocess_text(df, ['name', 'age', 'lith', 'descrip', 'comments'])

    _embed_model = embed_model.get()

    gpd_data = rank_polygons("Magmatic sulfide PGE", _embed_model, data)

    tile["units"]["features"] = get_geojson(gpd_data)

    layers = create_layer_list(tile)
    return encode(layers)


async def process_vector_tile_async(loop, content):
    # None uses the default executor (ThreadPoolExecutor)
    return await loop.run_in_executor(None, process_vector_tile, content)


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
