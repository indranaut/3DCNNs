import tensorflow as tf


class Unit3D(tf.keras.layers.Layer):
    def __init__(self, output_channels,
                 name,
                 conv_name,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn='relu',
                 use_batch_norm=True,
                 use_bias=False,
                 is_training=False,
                 input_shape=None,
                 data_format='channels_last',
                 ):
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._activation = activation_fn
        self._use_batch_norm = use_batch_norm
        self._use_bias = use_bias
        self._is_training = is_training
        self._data_format = data_format
        self._conv_name = conv_name
        if input_shape is None:
            self._conv = tf.keras.layers.Conv3D(
                filters=self._output_channels,
                kernel_size=self._kernel_shape,
                strides=self._stride,
                padding='same',
                use_bias=self._use_bias,
                data_format=self._data_format,
                name=self._conv_name
            )
        else:
            self._conv = tf.keras.layers.Conv3D(
                filters=self._output_channels,
                kernel_size=self._kernel_shape,
                strides=self._stride,
                padding='same',
                use_bias=self._use_bias,
                data_format=self._data_format,
                input_shape=input_shape,
                name=self._conv_name
            )
        self._pipeline = [self._conv]
        if self._use_batch_norm:
            self._pipeline.append(tf.keras.layers.BatchNormalization(
                axis=-1 if self._data_format == 'channels_last' else 1,
                fused=False,
                name='BatchNorm'
            ))

        if self._activation is not None:
            self._pipeline.append(tf.keras.layers.Activation(
                activation=self._activation,
            ))

    def call(self, input, training=False):
        for ind, layer in enumerate(self._pipeline):
            if ind == 0:
                out = layer(input)
            elif ind == 1 and self._use_batch_norm:
                out = layer(out, training=training)
            else:
                out = layer(out)

        return out


class Inception3D(tf.keras.Model):
    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 is_training=False,
                 dropout_keep_prob=0.5,
                 input_shape=None,
                 data_format='channels_last',
                 name='inception_i3d'
                 ):
        super(Inception3D, self).__init__(name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._is_training = is_training
        self._data_format = data_format
        self._dropout_keep_prob = dropout_keep_prob
        self._pipeline_main = []
        self._pipeline_main.append(Unit3D(output_channels=64,
                                          kernel_shape=[7, 7, 7],
                                          stride=[2, 2, 2],
                                          is_training=self._is_training,
                                          data_format=self._data_format,
                                          input_shape=input_shape,
                                          conv_name='Conv3d_1a_7x7',
                                          name='MainBranch'))
        self._pipeline_main.append(tf.keras.layers.MaxPool3D(
            pool_size=[1, 3, 3],
            strides=[1, 2, 2],
            padding='same',
            data_format=self._data_format,
            name='MaxPool3d_2a_3x3'
        ))
        self._pipeline_main.append(Unit3D(
            output_channels=64,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            conv_name='Conv3d_2b_1x1',
            name='MainBranch'
        ))
        self._pipeline_main.append(Unit3D(
            output_channels=192,
            kernel_shape=[3, 3, 3],
            is_training=self._is_training,
            data_format=self._data_format,
            conv_name='Conv3d_2c_3x3',
            name='MainBranch'
        ))
        self._pipeline_main.append(tf.keras.layers.MaxPool3D(
            pool_size=[1, 3, 3],
            strides=[1, 2, 2],
            padding='same',
            data_format=self._data_format,
            name='MaxPool3d_3a_3x3'
        ))

        self._pipeline_mixed3b0 = Unit3D(
            output_channels=64,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_3b/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )
        self._pipeline_mixed3b1 = [Unit3D(output_channels=96,
                                          kernel_shape=[1, 1, 1],
                                          is_training=self._is_training,
                                          data_format=self._data_format,
                                          name='Mixed_3b/Branch_1',
                                          conv_name='Conv3d_0a_1x1'),
                                   Unit3D(output_channels=128,
                                          kernel_shape=[3, 3, 3],
                                          is_training=self._is_training,
                                          data_format=self._data_format,
                                          name='Mixed_3b/Branch_1',
                                          conv_name='Conv3d_0b_3x3')
                                   ]

        self._pipeline_mixed3b2 = [
            Unit3D(output_channels=16,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3b/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=32,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3b/Branch_2',
                   conv_name='Conv3d_0b_3x3')

        ]

        self._pipeline_mixed3b3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_3b/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=32,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3b/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._pipeline_mixed3c0 = Unit3D(
            output_channels=128,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_3c/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed3c1 = [
            Unit3D(output_channels=128,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3c/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=192,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3c/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed3c2 = [
            Unit3D(output_channels=32,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3c/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=96,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3c/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed3c3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_3c/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=64,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_3c/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._mixed_4a = tf.keras.layers.MaxPool3D(
            pool_size=[3, 3, 3],
            strides=[2, 2, 2],
            padding='same',
            data_format=self._data_format,
            name='Mixed_4aMaxPool3d_0a_3x3'
        )

        self._pipeline_mixed4b0 = Unit3D(
            output_channels=192,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_4b/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed4b1 = [
            Unit3D(output_channels=96,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4b/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=208,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4b/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4b2 = [
            Unit3D(output_channels=16,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4b/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=48,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4b/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4b3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_4b/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=64,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4b/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._pipeline_mixed4c0 = Unit3D(
            output_channels=160,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_4c/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed4c1 = [
            Unit3D(output_channels=112,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4c/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=224,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4c/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4c2 = [
            Unit3D(output_channels=24,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4c/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=64,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4c/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4c3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_4c/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=64,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4c/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._pipeline_mixed4d0 = Unit3D(
            output_channels=128,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_4d/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed4d1 = [
            Unit3D(output_channels=128,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4d/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=256,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4d/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4d2 = [
            Unit3D(output_channels=24,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4d/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=64,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4d/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4d3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_4d/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=64,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4d/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._pipeline_mixed4e0 = Unit3D(
            output_channels=112,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_4e/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed4e1 = [
            Unit3D(output_channels=144,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4e/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=288,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4e/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4e2 = [
            Unit3D(output_channels=32,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4e/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=64,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4e/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4e3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_4e/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=64,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4e/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._pipeline_mixed4f0 = Unit3D(
            output_channels=256,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_4f/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed4f1 = [
            Unit3D(output_channels=160,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4f/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=320,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4f/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4f2 = [
            Unit3D(output_channels=32,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4f/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=128,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4f/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed4f3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_4f/Branch_2/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=128,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_4f/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._mixed_5a = tf.keras.layers.MaxPool3D(
            pool_size=[2, 2, 2],
            strides=[2, 2, 2],
            padding='same',
            data_format=self._data_format,
            name='Mixed_5a/MaxPool3d_0a_3x3'
        )

        self._pipeline_mixed5b0 = Unit3D(
            output_channels=256,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_5b/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed5b1 = [
            Unit3D(output_channels=160,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5b/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=320,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5b/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed5b2 = [
            Unit3D(output_channels=32,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5b/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=128,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5b/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed5b3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_5b/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=128,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5b/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._pipeline_mixed5c0 = Unit3D(
            output_channels=384,
            kernel_shape=[1, 1, 1],
            is_training=self._is_training,
            data_format=self._data_format,
            name='Mixed_5c/Branch_0',
            conv_name='Conv3d_0a_1x1'
        )

        self._pipeline_mixed5c1 = [
            Unit3D(output_channels=192,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5c/Branch_1',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=384,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5c/Branch_1',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed5c2 = [
            Unit3D(output_channels=48,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5c/Branch_2',
                   conv_name='Conv3d_0a_1x1'),
            Unit3D(output_channels=128,
                   kernel_shape=[3, 3, 3],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5c/Branch_2',
                   conv_name='Conv3d_0b_3x3')
        ]

        self._pipeline_mixed5c3 = [
            tf.keras.layers.MaxPool3D(
                pool_size=[3, 3, 3],
                strides=[1, 1, 1],
                padding='same',
                data_format=self._data_format,
                name='Mixed_5c/Branch_3/MaxPool3d_0a_3x3'
            ),
            Unit3D(output_channels=128,
                   kernel_shape=[1, 1, 1],
                   is_training=self._is_training,
                   data_format=self._data_format,
                   name='Mixed_5c/Branch_3',
                   conv_name='Conv3d_0b_1x1')
        ]

        self._pipeline_end = [
            tf.keras.layers.AveragePooling3D(
                pool_size=[2, 7, 7],
                strides=[1, 1, 1],
                padding='valid',
                data_format=self._data_format,
                name='avgpool'
            ),
            tf.keras.layers.Dropout(
                rate=self._dropout_keep_prob,
                name='dropout'
            ),
            Unit3D(output_channels=self._num_classes,
                   kernel_shape=[1, 1, 1],
                   activation_fn=None,
                   use_batch_norm=False,
                   use_bias=True,
                   data_format=self._data_format,
                   name='Logits',
                   conv_name='Conv3d_0c_1x1'
                   ),
        ]

        if self._spatial_squeeze:
            self._pipeline_end.append(
                tf.keras.layers.Lambda(lambda x:
                                       tf.squeeze(x,
                                                  axis=[2, 3] if self._data_format == 'channels_last'
                                                  else [3, 4], name='spatial_squeeze'
                                                  ))
            )

        self._pipeline_end.append(
            tf.keras.layers.Lambda(
                lambda x:
                tf.reduce_mean(x, axis=1 if self._data_format == 'channels_last' else 2, name='logits_mean')
            )
        )

    def compute_pipeline(self, pipeline, inputs):
        out = inputs
        for l in pipeline:
            if isinstance(l, Unit3D):
                out = l(out, training=self._is_training)
            else:
                out = l(out)
        return out

    @tf.function
    def call(self, inputs, training=False):
        out = self.compute_pipeline(self._pipeline_main, inputs)

        # Pipeline for Mixed3b block of I3D. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
        pipeline_mixed3b0 = self._pipeline_mixed3b0(out, training=self._is_training)

        pipeline_mixed3b1 = self.compute_pipeline(self._pipeline_mixed3b1, out)

        pipeline_mixed3b2 = self.compute_pipeline(self._pipeline_mixed3b2, out)

        pipeline_mixed3b3 = self.compute_pipeline(self._pipeline_mixed3b3, out)

        mixed_3b_out = tf.keras.layers.Concatenate(axis=1 if self._data_format == 'channels_first' else
        -1)([pipeline_mixed3b0,
             pipeline_mixed3b1,
             pipeline_mixed3b2,
             pipeline_mixed3b3])

        # Pipeline for Mixed3b block finished.

        # Pipeline for Mixed3c block of I3D. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
        pipeline_mixed3c0 = self._pipeline_mixed3c0(mixed_3b_out,
                                                    training=self._is_training)

        pipeline_mixed3c1 = self.compute_pipeline(self._pipeline_mixed3c1,
                                                  mixed_3b_out)

        pipeline_mixed3c2 = self.compute_pipeline(self._pipeline_mixed3c2,
                                                  mixed_3b_out)

        pipeline_mixed3c3 = self.compute_pipeline(self._pipeline_mixed3c3,
                                                  mixed_3b_out)

        mixed_3c_out = tf.keras.layers.Concatenate(axis=1 if self._data_format == 'channels_first' else
        -1)([pipeline_mixed3c0,
             pipeline_mixed3c1,
             pipeline_mixed3c2,
             pipeline_mixed3c3])

        # Pipeline for Mixed3c block finished.

        # Pipeline for Mixed4a block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
        mixed_4a = self._mixed_4a(mixed_3c_out)

        # Pipeline for Mixed4a block finished.

        # Pipeline for Mixed4b block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        pipeline_mixed4b0 = self._pipeline_mixed4b0(mixed_4a,
                                                    training=self._is_training)
        pipeline_mixed4b1 = self.compute_pipeline(self._pipeline_mixed4b1,
                                                  mixed_4a)

        pipeline_mixed4b2 = self.compute_pipeline(self._pipeline_mixed4b2,
                                                  mixed_4a)

        pipeline_mixed4b3 = self.compute_pipeline(self._pipeline_mixed4b3,
                                                  mixed_4a)
        mixed_4b_out = tf.keras.layers.Concatenate(axis=1 if self._data_format == 'channels_first' else
        -1)([pipeline_mixed4b0,
             pipeline_mixed4b1,
             pipeline_mixed4b2,
             pipeline_mixed4b3])

        # Pipeline for Mixed4b finished.

        # Pipeline for Mixed4c. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        pipeline_mixed4c0 = self._pipeline_mixed4c0(mixed_4b_out, training=self._is_training)
        pipeline_mixed4c1 = self.compute_pipeline(self._pipeline_mixed4c1,
                                                  mixed_4b_out)

        pipeline_mixed4c2 = self.compute_pipeline(self._pipeline_mixed4c2,
                                                  mixed_4b_out)

        pipeline_mixed4c3 = self.compute_pipeline(self._pipeline_mixed4c3,
                                                  mixed_4b_out)

        mixed_4c_out = tf.keras.layers.Concatenate(axis=1 if self._data_format == 'channels_first' else
        -1)([pipeline_mixed4c0,
             pipeline_mixed4c1,
             pipeline_mixed4c2,
             pipeline_mixed4c3])

        # Pipeline for Mixed4c block finished.

        # Pipeline for Mixed4d block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        pipeline_mixed4d0 = self._pipeline_mixed4d0(mixed_4c_out, training=self._is_training)
        pipeline_mixed4d1 = self.compute_pipeline(self._pipeline_mixed4d1,
                                                  mixed_4c_out)

        pipeline_mixed4d2 = self.compute_pipeline(self._pipeline_mixed4d2,
                                                  mixed_4c_out)

        pipeline_mixed4d3 = self.compute_pipeline(self._pipeline_mixed4d3,
                                                  mixed_4c_out)

        mixed_4d_out = tf.keras.layers.Concatenate(axis=-1 if self._data_format == 'channels_last' else 1)(
            [pipeline_mixed4d0,
             pipeline_mixed4d1,
             pipeline_mixed4d2,
             pipeline_mixed4d3])

        # Pipeline for Mixed4d block finished.

        # Pipeline for Mixed4e block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        pipeline_mixed4e0 = self._pipeline_mixed4e0(mixed_4d_out, training=self._is_training)
        pipeline_mixed4e1 = self.compute_pipeline(self._pipeline_mixed4e1,
                                                  mixed_4d_out)

        pipeline_mixed4e2 = self.compute_pipeline(self._pipeline_mixed4e2,
                                                  mixed_4d_out)
        pipeline_mixed4e3 = self.compute_pipeline(self._pipeline_mixed4e3,
                                                  mixed_4d_out)

        mixed_4e_out = tf.keras.layers.Concatenate(axis=-1 if self._data_format ==
                                                              'channels_last' else 1)([pipeline_mixed4e0,
                                                                                       pipeline_mixed4e1,
                                                                                       pipeline_mixed4e2,
                                                                                       pipeline_mixed4e3])

        # Pipeline for Mixed4e block finished.

        # Pipeline for Mixed4f block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        pipeline_mixed4f0 = self._pipeline_mixed4f0(mixed_4e_out, training=self._is_training)
        pipeline_mixed4f1 = self.compute_pipeline(self._pipeline_mixed4f1,
                                                  mixed_4e_out)
        pipeline_mixed4f2 = self.compute_pipeline(self._pipeline_mixed4f2,
                                                  mixed_4e_out)
        pipeline_mixed4f3 = self.compute_pipeline(self._pipeline_mixed4f3,
                                                  mixed_4e_out)
        mixed_4f_out = tf.keras.layers.Concatenate(axis=-1 if self._data_format ==
                                                              'channels_last' else 1)([pipeline_mixed4f0,
                                                                                       pipeline_mixed4f1,
                                                                                       pipeline_mixed4f2,
                                                                                       pipeline_mixed4f3])

        # Pipeline for Mixed4f block finished.

        # Pipeline for Mixed5a block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        mixed_5a = self._mixed_5a(mixed_4f_out)

        # Pipeline for Mixed5a block finished.

        # Pipeline for Mixed5b block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        pipeline_mixed5b0 = self._pipeline_mixed5b0(mixed_5a, training=self._is_training)

        pipeline_mixed5b1 = self.compute_pipeline(self._pipeline_mixed5b1,
                                                  mixed_5a)
        pipeline_mixed5b2 = self.compute_pipeline(self._pipeline_mixed5b2,
                                                  mixed_5a)
        pipeline_mixed5b3 = self.compute_pipeline(self._pipeline_mixed5b3,
                                                  mixed_5a)
        mixed_5b_out = tf.keras.layers.Concatenate(axis=-1 if self._data_format ==
                                                              'channels_last' else 1)([pipeline_mixed5b0,
                                                                                       pipeline_mixed5b1,
                                                                                       pipeline_mixed5b2,
                                                                                       pipeline_mixed5b3])

        # Pipeline for Mixed5b block finished.

        # Pipeline for Mixed5c block. See https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py

        pipeline_mixed5c0 = self._pipeline_mixed5c0(mixed_5b_out, training=self._is_training)
        pipeline_mixed5c1 = self.compute_pipeline(self._pipeline_mixed5c1,
                                                  mixed_5b_out)
        pipeline_mixed5c2 = self.compute_pipeline(self._pipeline_mixed5c2,
                                                  mixed_5b_out)
        pipeline_mixed5c3 = self.compute_pipeline(self._pipeline_mixed5c3,
                                                  mixed_5b_out)
        mixed_5c_out = tf.keras.layers.Concatenate(axis=-1 if self._data_format ==
                                                              'channels_last' else 1)([pipeline_mixed5c0,
                                                                                       pipeline_mixed5c1,
                                                                                       pipeline_mixed5c2,
                                                                                       pipeline_mixed5c3])

        # Pipeline for Mixed5c block finished.

        out = self.compute_pipeline(self._pipeline_end,
                                    mixed_5c_out)

        return out
