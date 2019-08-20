import tensorflow as tf
import logging
from protos import network_pb2
from nets.i3d import Inception3D

NETOWRK_MAP = dict(
    i3d=Inception3D
)

def parse_network_config(network_proto_config):
    numframes = network_proto_config.numframes
    input_height = network_proto_config.input_height
    input_width = network_proto_config.input_width
    num_classes = network_proto_config.num_classes
    data_format = network_proto_config.WhichOneof('data_format')
    data_format = data_format.lower()
    network_type = network_proto_config.WhichOneof('network_details')
    network_parameters = network_proto_config.network_type
    network_dictionary = dict(
        name=network_type,
        network_parameters=network_parameters,
        data_format=data_format,
        input_height=input_height,
        input_width=input_width,
        numframes=numframes,
        num_classes=num_classes
    )
    return network_dictionary


def build_network(network_proto_config, is_training=False):
    network_dictionary = parse_network_config(network_proto_config)
    network_fn = NETOWRK_MAP[network_dictionary['name']]
    network = network_fn(num_classes=network_dictionary['num_classes'],
                         is_training=is_training, data_format=network_dictionary['data_format'],
                         **network_dictionary['network_parameters'])

    if network_dictionary['data_format'] == 'channels_last':
        input_shape = (None, network_dictionary['numframes'],
                       network_dictionary['input_height'],
                       network_dictionary['input_width'],
                       3)
    else:
        input_shape = (None, network_dictionary['numframes'],3,
                        network_dictionary['input_height'],
                        network_dictionary['input_width'])

    network.build(input_shape=input_shape)
    logging.info(network.summary())

    return network






