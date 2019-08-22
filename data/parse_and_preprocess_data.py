import tensorflow as tf
from data import preprocessing
from builders.augmentation_builder import build_augmentations


def feature_dict(subset):
    features = dict(
        video_file_name=tf.io.FixedLenFeature(
            shape=(), dtype=tf.string, default_value=''
        ),
        video=tf.io.FixedLenFeature(
            shape=(), dtype=tf.string, default_value=''
        ),
        numframes=tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=-1
                                        ),
        subset=tf.io.FixedLenFeature(shape=(), dtype=tf.string,
                                     default_value=''),
        height=tf.io.FixedLenFeature(shape=(), dtype=tf.int64,
                                     default_value=-1),
        width=tf.io.FixedLenFeature(shape=(), dtype=tf.int64,
                                    default_value=-1)
    )
    if subset in ['train', 'val']:
        additional_dict = dict(
            label=tf.io.FixedLenFeature(shape=(), dtype=tf.int64,
                                        default_value=-1
                                        ),
            label_text=tf.io.FixedLenFeature(shape=(), dtype=tf.string,
                                             default_value=''
                                             )
        )
        features = {**features, **additional_dict}
        return features

    return features


def parse_single_record(record, numclasses, subset):
    features = feature_dict(subset)
    parsed_record = tf.io.parse_single_example(serialized=record,
                                               features=features)

    video = parsed_record['video']
    video = tf.image.decode_image(contents=video,
                                  channels=3,
                                  dtype=tf.uint8)

    height = parsed_record['height']
    width = parsed_record['width']
    numframes = parsed_record['numframes']
    video_shape = tf.stack([numframes, height, width, 3], axis=0)
    video = tf.reshape(video, video_shape)
    filename = parsed_record['filename']
    filename = tf.strings.unicode_decode(input=filename,
                                         input_encoding='UTF-8')

    output_data = dict(
        video=video,
        filename=filename
    )

    if subset in ['train', 'val']:
        label = parsed_record['label']
        label = tf.one_hot(indices=label,
                           depth=numclasses)
        label_text = parsed_record['label_text']
        label_text = tf.strings.unicode_encode(input=label_text,
                                               output_encoding='UTF-8')

        additional_output = dict(
            label=label,
            label_text=label_text
        )
        output_data = {**output_data, **additional_output}

    return output_data



def append_frames_by_repeating(video, numframes_video, numframes_out):
    numframes_to_append = numframes_out - numframes_video
    last_frame_index = numframes_video - 1
    _, frame_height, frame_width, _ = tf.shape(video)
    last_frame = tf.slice(input_=video,
                          begin=[last_frame_index, 0, 0, 0],
                          size=[1, frame_height, frame_width, 3])

    appended_frames = tf.stack([last_frame] * numframes_to_append,
                               axis=0)
    output_video = tf.concat([video, appended_frames], axis=0)
    return output_video

def select_random_continuous_frames(video, num_frames):
    numframes_video, frame_height, frame_width, _ = tf.shape(video)[0]
    starting_frame_index = tf.random.uniform(
        shape=(),
        minval=0,
        maxval=numframes_video,
        dtype=tf.int32
    )

    sliced_video = tf.slice(
        input_=video,
        begin=[starting_frame_index, 0, 0, 0],
        size=[num_frames, frame_height, frame_width, 3]
    )
    return sliced_video


def preprocess_train_eval(parsed_record, num_frames, augmentation_fn, repeat_if_less=False):
    video = parsed_record['video']
    numframes_video = tf.shape(video)[0]
    if repeat_if_less:
        video = tf.cond(
            tf.less(numframes_video, num_frames),
            lambda : append_frames_by_repeating(video,
                                                numframes_video, num_frames),
            lambda : video
        )

    numframes_video = tf.shape(video)[0]
    video = select_random_continuous_frames(video, num_frames=num_frames)
    video = augmentation_fn(video)
    parsed_record['video'] = video
    return parsed_record










