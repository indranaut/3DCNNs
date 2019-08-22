import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import tensorflow as tf
from google.protobuf import text_format
from builders.network_builder import *
from protos import network_pb2
from nets.i3d import Inception3D
import numpy as np


class NetworkBuilderTest(tf.test.TestCase):
    def setUp(self):
        text_proto = """
                numframes : 64
                input_height : 224
                input_width : 224
                num_classes : 400
                i3d {
                }
                data_format : CHANNELS_FIRST
                """
        proto_msg = network_pb2.Network()
        text_format.Merge(text_proto, proto_msg)
        self._msg = proto_msg

    def test_parser(self):
        proto_msg = self._msg
        network_dict = parse_network_config(proto_msg)
        network_dict_expected = dict(
            name='i3d',
            i3d=dict(
                spatial_squeeze=True,
                dropout_keep_prob=0.5,
                name='inception_i3d'
            ),
            numframes=64,
            input_height=224,
            input_width=224,
            num_classes=400,
            data_format='channels_first'
        )
        self.assertDictEqual(network_dict, network_dict_expected)

    def test_network_creation(self):
        network = build_network(network_proto_config=self._msg,
                                is_training=False)

        inputs = tf.keras.backend.random_uniform(shape=(1, 3, 64, 224, 224),
                                                 dtype=tf.float32)

        output_shape = network(inputs)
        expected_output_shape = np.zeros(shape=(1, 400))
        self.assertShapeEqual(expected_output_shape, output_shape)


if __name__ == "__main__":
    tf.test.main()
