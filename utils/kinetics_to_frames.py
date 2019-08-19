import os
import subprocess
import glob
import multiprocessing as mp
import logging
import argparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser('--prog', 'Code to divide the Kinetics dataset'
                                           'into frames.')
parser.add_argument('--source_dir', required=True,
                    type=str, help='Source folder where the Kinetics dataset is stored.')
parser.add_argument('--dest_dir', required=True,
                    type=str, help='Destination folder where the frames will be stored.')

SUBSETDIRS = ['train', 'val', 'test']

SUBSET_VALIDITY = None
DEST_DIR = None


def validate_source_dir(source_dir):
    if not os.path.isdir(source_dir):
        raise OSError('The source folder {} does not exist.'.format(source_dir))
    subdirs = list(map(lambda x: os.path.basename(x),
                       os.listdir(source_dir)))
    global SUBSET_VALIDITY
    SUBSET_VALIDITY = dict()
    for subd in SUBSETDIRS:
        if subd in subdirs:
            SUBSET_VALIDITY[subd] = True
        else:
            SUBSET_VALIDITY[subd] = False
            logging.warning('The subset {} was not found inside {}.'.format(subd,
                                                                            source_dir))

    return None


def get_labels(source_dir):
    global SUBSET_VALIDITY
    if not SUBSET_VALIDITY['train']:
        logging.error('Training subset was not found in the source folder {}.'
                      'Cannot get the labels.'.format(source_dir))
        return None

    train_path = os.path.join(source_dir, 'train')
    dirs = os.listdir(train_path)
    dirs = list(map(lambda x: os.path.basename(x),
                    dirs))
    labelnum = list(range(len(dirs)))
    labelmap = dict(zip(dirs, labelnum))
    return labelmap


def save_labelmap(labelmap, dest_dir):
    labelmap_filename = os.path.join(dest_dir, 'labelmap.txt')
    try:
        with open(labelmap_filename, 'w') as fid:
            for classname, classlabel in labelmap.items():
                fid.write('{},{}\n'.format(classname, classlabel))
    except Exception as e:
        logging.error(e)

    return None


def get_targets(source_dir, subset):
    global SUBSET_VALIDITY
    if not SUBSET_VALIDITY[subset]:
        return None

    if subset in ['train', 'val']:
        targets = glob.glob(os.path.join(source_dir, subset, '**', '*.mp4'),
                            recursive=True)
        destinations = list(map(lambda x :
                                os.path.basename(os.path.dirname(x)),
                                targets))
    else:
        targets = glob.glob(os.path.join(source_dir, '*.mp4'),
                            recursive=True)
        destinations = None
    return targets, destinations


def unpack_frames(video_file, dest_dir):
    filename = os.path.splitext(
        os.path.basename(video_file))[0]
    dest_dir = os.path.join(dest_dir, filename)
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.call([
        'ffmpeg', '-i',
        '{}'.format(video_file),
        os.path.join(dest_dir, 'frame%04d.png'),
        '-hide_banner'
    ])
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    source_dir = args.source_dir
    DEST_DIR = args.dest_dir
    os.makedirs(DEST_DIR, exist_ok=True)

    logging.info('Validating the source folder.')
    validate_source_dir(source_dir)
    logging.info('Getting the label map for the dataset.')
    label_map = get_labels(source_dir)
    logging.info('Saving the labelmap')
    save_labelmap(label_map, DEST_DIR)

    for subset in SUBSETDIRS:
        logging.info('Processing the subset "{}"'.format(subset))
        logging.info('Getting the targets and destinations.')
        targets, destinations = get_targets(source_dir, subset)
        logging.info('Total number of targets = {}'.format(len(targets)))
        dest_folder = os.path.join(DEST_DIR, subset)
        logging.info('Creating destination folder {}.'.format(dest_folder))
        os.makedirs(dest_folder, exist_ok=True)
        if not destinations:
            destinations = [dest_folder] * len(targets)
        else:
            destinations = list(map(lambda x : os.path.join(dest_folder, x),
                                    destinations))
            list(map(lambda x : os.makedirs(x, exist_ok=True), destinations))
        logging.info('Processing in parallel with {} processes.'.format(mp.cpu_count()))
        pool = mp.Pool(processes=mp.cpu_count())
        arguments = zip(targets, destinations)
        pool.starmap(unpack_frames, arguments)
