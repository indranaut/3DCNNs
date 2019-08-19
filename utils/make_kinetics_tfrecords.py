import os
import glob
import multiprocessing as mp
import argparse
import logging
import tensorflow as tf
from itertools import compress
import cv2
import numpy as np

from utils.logging_utils import ExitOnExceptionHandler

#TODO: Test the code on TF 2.0

parser = argparse.ArgumentParser('--prog',
                                 'Code to create TFRecords for the Kinetics Dataset.')
parser.add_argument('--source_dir', required=True,
                    help='Folder where the kinetics dataset is stored.')
parser.add_argument('--tfrecord_dir', required=True,
                    help='Folder where the TFRecord files are stored.')
parser.add_argument('--shards_train', required=False, type=int,
                    default=6144, help='Number of training shards.')
parser.add_argument('--shards_val', required=False, type=int,
                    default=2048, help='Number of validation shards.')
parser.add_argument('--shards_test', required=False, type=int,
                    default=512, help='Number of testing shards.')
parser.add_argument('--subset', required=False, type=str,
                    default='all', nargs='?', choices=['train', 'val', 'test', 'all'],
                    const='all',
                    help='The subset of Kinetics for which the TFRecords will be created.')

LABELMAP = None


def num_digits(number):
    """
    Returns number of digits in an integer.
    :param number: Integer
    :return: Number of digits
    """
    return len(str(number))


def validate_dir_presence(source_dir):
    """
    Returns True if source_dir is a found folder else False
    :param source_dir: Folder name
    :return: True if source_dir is a found folder else False
    """
    return os.path.isdir(source_dir)


def validate_dataset_presence_coarse(source_dir, subset):
    """
    Determines if the Kinetics dataset structure is found.
    This structure determination is coarse in nature as explained next.

    The kinetics dataset is in a folder (source_dir) inside which are three folders :
    "train", "val" and "test".
    Depending upon the value passed to the argument --subset during execution,
    this function determines if these subset folders indeed exist.
    If they are not found, the code exists with a log of level ERROR.
    :param source_dir: Base folder inside which Kinetics dataset is supposed to exist.
    :param subset: The value of the command line argument --subset.
    :return: None if the specified subset exists. Else, logs an error and exits the program.
    """
    if not validate_dir_presence(source_dir):
        logging.error('The kinetics dataset was supposed to be present at {}'
                      'but was not found.'.format(source_dir))
    if subset == 'all':
        subset = ['train', 'val', 'test']
    else:
        subset = [subset]

    absence = [not validate_dir_presence(os.path.join(source_dir, x)) for x in subset]
    absent_indices = list(compress(range(len(absence)), absence))
    if not absent_indices:
        return None

    absent_subsets = [subset[x] for x in absent_indices]
    logging.error('Some or all the specified subsets were not found. '
                  'Those not found are {}.'.format(absent_subsets))

    return None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_video(video_file):
    frames = list()
    videocap = cv2.VideoCapture(video_file)
    success, frame = videocap.read()
    numframes = 0
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        numframes += 1
        success, frame = videocap.read()

    frame_tensor = np.stack(frames, axis=0)
    logging.debug('{} frames in {}.'.format(numframes, video_file))
    return frame_tensor, numframes

def get_targets(source_dir, subset):
    targets_train = []
    targets_val = []
    targets_test = []
    if subset == 'all':
        subset = ['train', 'val', 'test']
    else:
        subset = [subset]

    for s in subset:
        files = glob.glob(
            os.path.join(source_dir, s, '**', '*.mp4'),
            recursive=True
        )
        if s == 'train':
            targets_train+=files
        elif s == 'val':
            targets_val+=files
        else:
            targets_test+=files
    return targets_train, targets_val, targets_test


def get_labelmap(source_dir):
    global LABELMAP
    folder = os.path.join(source_dir, 'train')
    if not validate_dir_presence(folder):
        logging.warning('The training subset cannot be found and so '
                        'cannot construct the labelmap.')
        return None

    dirs = os.listdir(folder)
    LABELMAP = dict(zip(dirs, list(range(len(dirs)))))
    return None

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]


def serialize_record(video_file, subset):
    global LABELMAP
    frame_tensor, numframes = read_video(video_file)

    feature = {
        'video_file_name': _bytes_feature(
            tf.compat.as_bytes(os.path.basename(video_file))
        ),
        'video': _bytes_feature(
            tf.compat.as_bytes(
                frame_tensor.tostring()
            )
        ),
        'numframes': _int64_feature(numframes),
        'subset': _bytes_feature(
            tf.compat.as_bytes(subset)
        )
    }

    if subset in ['train', 'val']:
        labelname = os.path.basename(
            os.path.dirname(video_file)
        )
        labelnum = LABELMAP[labelname]
        feature['label'] = _int64_feature(labelnum)
        feature['label_text'] = _bytes_feature(
            tf.compat.as_bytes(labelname)
        )
    return feature


def create_one_record_file(filename, videofiles, subset):
    with tf.io.TFRecordWriter(filename) as writer:
        for video_file in videofiles:
            serialized_record = serialize_record(video_file, subset)
            example_proto = tf.train.Example(
                features=tf.train.Features(feature=serialized_record)
            )
            serialized_example = example_proto.SerializeToString()
            writer.write(serialized_example)

    logging.info('The file {} was written with {} records.'.format(filename, len(videofiles)))
    return None

if __name__ == "__main__":
    args = parser.parse_args()
    source_dir = args.source_dir
    tfrecord_dir = args.tfrecord_dir
    shards_train = args.shards_train
    shards_val = args.shards_val
    shards_test = args.shards_test
    subset = args.subset

    logging_filename = os.path.join(tfrecord_dir, '{}.log'.format(subset))
    logging.basicConfig(level=logging.DEBUG,
                       handlers=[ExitOnExceptionHandler()],
                       filename=logging_filename)
    validate_dir_presence(source_dir)

    os.makedirs(tfrecord_dir, exist_ok=True)
    get_labelmap(source_dir)
    targets_train, targets_val, targets_test = get_targets(source_dir, subset)

    if subset == 'all':
        subset = ['train', 'val', 'test']
    else:
        subset = [subset]

    for s in subset:
        if s == 'train':
            numshards = shards_train
            targets = targets_train
        elif s == 'val':
            numshards = shards_val
            targets = targets_val
        else:
            numshards = shards_test
            targets = targets_test

        filenames = list(
            map(
                lambda x : os.path.join(tfrecord_dir,
                                        '{}-tfrecord-{}-of-{}.record'.format(
                                            s, str(x).zfill(num_digits(numshards)),
                                            numshards
                                        )),
                list(range(1, numshards + 1))
            )
        )
        targets = chunkify(targets, numshards)
        subset_arg = [s] * len(filenames)
        pool_arguments = list(zip(filenames,
                                  targets,
                                  subset_arg))
        pool = mp.Pool(processes=mp.cpu_count())
        pool.starmap(create_one_record_file, pool_arguments)
