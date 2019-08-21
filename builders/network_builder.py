import logging
from nets.i3d import Inception3D
from google.protobuf.json_format import MessageToDict

NETOWRK_MAP = dict(
    i3d=Inception3D
)


def parse_network_config(network_proto_config):
    network_dict = MessageToDict(message=network_proto_config,
                                 including_default_value_fields=True,
                                 preserving_proto_field_name=True)
    network_name = network_proto_config.WhichOneof('network_details')
    data_format = network_dict['data_format'].lower()
    network_dict['data_format'] = data_format
    network_dict['name'] = network_name
    return network_dict


def build_network(network_proto_config, is_training=False):
    network_dictionary = parse_network_config(network_proto_config)
    network_fn = NETOWRK_MAP[network_dictionary['name']]
    network = network_fn(num_classes=network_dictionary['num_classes'],
                         is_training=is_training, data_format=network_dictionary['data_format'],
                         **network_dictionary[network_dictionary['name']])

    if network_dictionary['data_format'] == 'channels_last':
        input_shape = (None, network_dictionary['num_frames'],
                       network_dictionary['input_height'],
                       network_dictionary['input_width'],
                       3)
    else:
        input_shape = (None, 3, network_dictionary['numframes'],
                       network_dictionary['input_height'],
                       network_dictionary['input_width'])

    network.build(input_shape=input_shape)
    logging.info(network.summary())

    return network
