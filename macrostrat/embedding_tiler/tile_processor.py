from mapbox_vector_tile import decode, encode
from .utils import timer
from .text_pipeline import preprocess_text
from sentence_transformers import SentenceTransformer
from geopandas import GeoDataFrame

# Note: for now, the model is downloaded from the Hugging Face model hub
# on the first run. This will take a while. We will pre-load the model
# in the future.
hf_model = 'BAAI/bge-base-en-v1.5'
embed_model = SentenceTransformer(hf_model)


@timer("Process tile")
def process_vector_tile(content: bytes):
    """
    Decode a vector tile and return the features.
    """
    tile = decode(content)

    df = get_data_frame(tile)
    data = preprocess_text(df, ['name', 'age', 'lith', 'descrip', 'comments'])

    tile["units"]["features"] = get_geojson(df)

    layers = create_layer_list(tile)
    return encode(layers)


def create_layer_list(tile):
    layers = []
    for key, layer in tile.items():
        layer["name"] = key
        layers.append(layer)
    return layers


def get_data_frame(tile):
    return GeoDataFrame.from_features(tile['units']['features'])


def get_geojson(df):
    return df.to_dict('records')
