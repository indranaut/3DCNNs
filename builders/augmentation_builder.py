from google.protobuf.json_format import MessageToDict
from data import preprocessing
import functools


def parse_augmentation_config(augmentation_config):
    augmentation_dict = MessageToDict(augmentation_config,
                                      including_default_value_fields=True,
                                      preserving_proto_field_name=True)

    return augmentation_dict


def build_augmentations(augmentation_config, input_height, input_width):
    augmentation_dict = parse_augmentation_config(augmentation_config)
    augmentations = []
    if 'random_crop' in augmentation_dict:
        crop_type = list(augmentation_dict['random_crop'].keys())[0]
        args_dict = dict(
            name=crop_type,
            input_height=input_height,
            input_width=input_width,
        )
        args_dict = {**args_dict, **augmentation_dict['random_crop']}
        augmentations.append(
            preprocessing.build_random_crop(args_dict)
        )

    if 'random_brightness' in augmentation_dict:
        args_dict = augmentation_dict['random_brightness']
        augmentations.append(
            preprocessing.random_brightness(args_dict)
        )

    if 'random_contrast' in augmentation_dict:
        args_dict = augmentation_dict['random_contrast']
        augmentations.append(
            preprocessing.random_contrast(args_dict)
        )

    if 'horizontal_flip' in augmentation_dict:
        augmentations.append(
            preprocessing.flip_left_right()
        )

    augmentation_fn = functools.reduce(
        lambda x, y: y(x),
        augmentations
    )

    return augmentation_fn
