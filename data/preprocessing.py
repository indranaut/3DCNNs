import tensorflow as tf
import logging


def flip_left_right():
    return lambda x: tf.image.random_flip_left_right(x)


def random_brightness(args_dict):
    max_delta = args_dict['max_delta']
    assert (max_delta >= 0), logging.fatal("'max_delta' argument to "
                                           "tf.image.random_brightness must be non-negative.")
    return lambda x: tf.image.random_brightness(x, max_delta=max_delta)


def random_contrast(args_dict):
    lower = args_dict['lower']
    upper = args_dict['upper']
    assert (lower < 0), logging.fatal("'lower' parameter to tf.image.random_contrast"
                                      "must be non-negative.")

    assert (upper >= lower), logging.fatal("'upper>=lower' must be satisfied for the parameters of "
                                           "tf.image.random_contrast.")
    return lambda x: tf.image.random_contrast(x, lower=lower,
                                              upper=upper)


def build_resize_min(target_size):
    def resize_minimum(image):
        image_height, image_width, _ = tf.shape(image)
        image_height = tf.cast(image_height, tf.float32)
        image_width = tf.cast(image_width, tf.float32)
        aspect_ratio = image_height / image_width
        new_height, new_width = tf.cond(
            image_height < image_width,
            lambda: target_size, tf.cast(
                target_size / aspect_ratio,
                tf.int32
            ),
            lambda: tf.cast(
                target_size * aspect_ratio, tf.int32
            ), target_size
        )
        return tf.image.resize(image, [new_height, new_width])

    return lambda x: resize_minimum(x)


def build_resize_max(target_size):
    def resize_maximum(image):
        image_height, image_width, _ = tf.shape(image)
        image_height = tf.cast(image_height, tf.float32)
        image_width = tf.cast(image_width, tf.float32)
        aspect_ratio = image_height / image_width
        new_height, new_width = tf.cond(
            image_height < image_width,
            lambda: tf.cast(
                target_size * aspect_ratio,
                tf.int32
            ), target_size,
            lambda: target_size, tf.cast(
                target_size / aspect_ratio, tf.int32
            )
        )
        return tf.image.resize(image, [new_height, new_width])

    return lambda x: resize_maximum(x)


def build_resize_exact(target_height, target_width):
    return lambda x: tf.image.resize(x, [target_height, target_width])


def build_random_crop(args_dict):
    RESIZE_DICT = dict(
        resize_min=build_resize_min,
        resize_max=build_resize_max,
        resize_exact=build_resize_exact
    )
    resize_method = args_dict.pop('name')
    input_height = args_dict.pop('input_height')
    input_width = args_dict.pop('input_width')
    resize_fn = RESIZE_DICT[resize_method](**args_dict)
    return lambda x: tf.image.random_crop(resize_fn(x), [input_height, input_width])
