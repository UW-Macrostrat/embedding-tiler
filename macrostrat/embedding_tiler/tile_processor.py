from mapbox_vector_tile import decode, encode

def process_vector_tile(content: bytes):
    """
    Decode a vector tile and return the features.
    """
    tile = decode(content)

    print(tile.keys())


    res = encode(list(tile.values()))

    return tile
