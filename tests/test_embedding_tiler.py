from pytest import fixture
from pathlib import Path
from mapbox_vector_tile import decode, encode

__here__ = Path(__file__).parent


@fixture
def tile_data():
    file = __here__ / "fixtures" / "carto-z5-x6-y12.pbf"
    with file.open("rb") as f:
        return f.read()


def test_read_tile(tile_data):
    tile = decode(tile_data)
    assert tile.keys() == {"units", "lines"}
