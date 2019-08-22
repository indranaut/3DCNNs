import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from builders.augmentation_builder import *
from google.protobuf import text_format
from protos import augmentation_pb2
import tensorflow as tf


#TODO : Add tests for augmentation functions.
class AugmentationBuilderTest(tf.test.TestCase):
    def setUp(self):
        proto_txt = """
        random_brightness{
            max_delta : 0.5
        }
        random_contrast{
            lower:0.0
            upper:0.3
        }
        horizontal_flip : true
        random_crop{
            resize_min {
                target_size : 256
            }
        }
        """
        proto_msg = augmentation_pb2.Augmentations()
        text_format.Merge(proto_txt, proto_msg)
        self._proto_msg = proto_msg

    def test_augmentation_config(self):
        augmentation_dict = parse_augmentation_config(self._proto_msg)
        expected_augmentation_dict = dict(
            random_brightness=dict(
                max_delta=0.5
            ),
            random_contrast=dict(
                lower=0.0,
                upper=0.3
            ),
            horizontal_flip=True,
            random_crop=dict(
                resize_min=dict(
                    target_size=256
                )
            )
        )
        self.assertDictEqual(augmentation_dict, expected_augmentation_dict, msg=self._proto_msg)




if __name__ == "__main__":
    tf.app.main()