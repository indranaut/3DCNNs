import nets as nets


NETWORK_DICTIONARY = {
    'i3d' : nets.i3d.Inception3D,
}


def parse_network(network_proto_config):
    network_name = network_proto_config.network_name
    if network_name not in NETWORK_DICTIONARY.keys():
        raise ValueError('The network {} was not found in the dictionary of valid networks.')

    numframes = network_proto_config.numframes
    input_height = network_proto_config.input_height
    input_width = network_proto_config.input_width
    num_classes= network_proto_config.num_classes
    return network_name, numframes, input_height, input_width, num_classes


def build_network(network_proto_config, is_training=False):
    network_name, numframes, input_height, input_width = parse_network(network_proto_config)
