import tensorflow as tf
from nets.i3d import Inception3D
import numpy as np


class TestI3DShape(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestI3DShape, self).__init__( *args, **kwargs)

    def test_i3d_output_shape(self):
        model = Inception3D(num_classes=400,
                            data_format='channels_last',
                            input_shape=(None, 64, 224, 224, 3))
        inputs = tf.keras.backend.random_uniform(shape=(1, 64, 224, 224, 3),
                                                 dtype=tf.float32)
        correct_shape = np.zeros(shape=(1, 400))
        output = model(inputs)
        self.assertShapeEqual(correct_shape, output)


if __name__ == "__main__":
    tf.test.main()
