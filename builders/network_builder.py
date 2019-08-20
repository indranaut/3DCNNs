from protos import network_pb2


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
