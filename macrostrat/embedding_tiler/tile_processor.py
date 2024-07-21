from mapbox_vector_tile import decode, encode


def process_vector_tile(content: bytes):
    """
    Decode a vector tile and return the features.
    """
    tile = decode(content)

    layers = create_layer_list(tile)
    return encode(layers)


def create_layer_list(tile):
    layers = []
    for key, layer in tile.items():
        layer["name"] = key
        layers.append(layer)
    return layers
