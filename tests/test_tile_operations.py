from pytest import fixture
from pathlib import Path
from mapbox_vector_tile import decode, encode
from macrostrat.embedding_tiler.text_pipeline import preprocess_text, rank_polygon
from macrostrat.embedding_tiler.tile_processor import create_layer_list, process_vector_tile, get_data_frame, \
    get_geojson, embed_model, systems_dict
from geopandas import GeoDataFrame

__here__ = Path(__file__).parent


@fixture
def tile_data():
    file = __here__ / "fixtures" / "carto-z5-x6-y12.pbf"
    with file.open("rb") as f:
        return f.read()


def test_read_tile(tile_data):
    tile = decode(tile_data)
    assert tile.keys() == {"units", "lines"}

    units = tile["units"]

    assert units["extent"] == 4096
    assert len(units["features"]) > 10

    first_feature = units["features"][0]
    properties = first_feature["properties"]

    # Check that properties match expected types
    assert isinstance(properties["map_id"], int)
    assert isinstance(properties["legend_id"], int)
    assert isinstance(properties["descrip"], str)


def test_encode_tile(tile_data):
    tile = decode(tile_data)
    layers = create_layer_list(tile)
    assert layers[0]["name"] == "units"
    res = encode(layers)

    assert len(res) >= len(tile_data)


def test_encode_decode_dataframe(tile_data):
    tile = decode(tile_data)
    assert tile.keys() == {"units", "lines"}

    df = get_data_frame(tile)
    assert len(df) > 10
    tile["units"]["features"] = get_geojson(df)

    layers = create_layer_list(tile)
    encode(layers)


def test_decode_encode_round_trip(tile_data):
    tile_data2 = process_vector_tile(tile_data)
    tile = decode(tile_data2)
    assert tile.keys() == {"units", "lines"}


def test_polygon_ranking(tile_data):
    tile = decode(tile_data)
    input_cols = ['name', 'age', 'lith', 'descrip', 'comments']

    df = get_data_frame(tile)
    data = preprocess_text(df, input_cols)

    deposit_type = 'porphyry_copper'
    _embed_model = embed_model.get()

    gpd_data, cos_sim = rank_polygon(systems_dict[deposit_type], _embed_model, data)

    assert len(gpd_data) > 10
    for col in input_cols:
        assert col in gpd_data.columns
    assert "bge_all" in gpd_data.columns

    # Convert to a geodataframe
    gpd_data = GeoDataFrame(gpd_data, geometry='geometry')

    gd = get_geojson(gpd_data)

    assert gd[0]["properties"]["bge_all"] == gpd_data["bge_all"].iloc[0]

    assert len(gd) == len(df)
