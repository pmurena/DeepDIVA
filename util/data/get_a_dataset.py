"""
This script allows for creation of a validation set from the training set.

"""

# Utils
import argparse
import inspect
import os
import shutil
import sys
import urllib
import zipfile
import json

import torch
import torchvision
from PIL import Image

from util.data.dataset_splitter import split_dataset


def mnist(args):
    # Use torchvision to download the dataset
    torchvision.datasets.MNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'MNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)

    return


def coco(args):
    """Downloads the 2017 COCO dataset and stores it on local disc.

    This function downloads the 2017 COCO dataset and stores it in the
    following file structure on disc:
        folder
            |__ annotations
            |__ train
            |__ valid
            |__ test

    Where "folder" is as defined by "args.output_folder" or "./data" if
    "args.output_folder" is not set. Train, valid, and test hold the images and
    annotation files for training, validation, and testing. The annotation
    files generated, data_info.json, are stored in the same folder as the
    respective images and have the following format:
        {
            image_id (int): COCO image id
            {
                file_name (str): coco image file name,
                labels (list): conjunction of coco categories and stuff
                captions (list): all captions of the current image
            }
        }

    coco(args) will only alter the original COCO folder structure; the data
    split remains unchanged.

    All original descriptions files, as well as license information, are stored
    in "args.output_folder/annotations". For more details on COCO see:
        http://cocodataset.org/

    Args:
        args (ArgumentParser): Command line arguments as set in main().

    Returns:
        None

    """
    # Build dataset folder structure
    root_dir = os.path.join(args.output_folder, 'COCO')
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'valid')
    test_dir = os.path.join(root_dir, 'test')
    anno_dir = os.path.join(root_dir, 'annotations')

    # Define coco urls
    base_url = 'http://images.cocodataset.org/'
    annotations_url = base_url + 'annotations/'
    images_url = base_url + 'zips/'

    # Build download dictionary as: [source_url, target_dir, is_image_flag]
    to_download = []
    to_download.append((
        annotations_url + 'annotations_trainval2017.zip',
        root_dir,
        False
    ))
    to_download.append((
        annotations_url + 'stuff_annotations_trainval2017.zip',
        root_dir,
        False
    ))
    to_download.append((
        annotations_url + 'image_info_test2017.zip',
        root_dir,
        False
    ))
    to_download.append((images_url + 'train2017.zip', train_dir, True))
    to_download.append((images_url + 'val2017.zip', valid_dir, True))
    to_download.append((images_url + 'test2017.zip', test_dir, True))

    # Download and extract files from to_download dictionary
    for source, target, is_image in to_download:
        print('Processing: {}'.format(source))

        # Download archive
        responce = urllib.request.urlretrieve(
            source,
            './tmp_file_19800223',
            _download_reporthook
        )

        # Extract archive
        with zipfile.ZipFile(responce[0]) as zip_file:
            print('\tinflating to {}'.format(target))

            # Store images to DeepDiva standard folder structure
            if is_image:  # If zip file holds images
                for img in zip_file.namelist():
                    if img[-1] == '/':
                        continue
                    _, img_name = os.path.split(img)
                    target_img = os.path.join(target, img_name)
                    _make_folder_if_not_exists(target)
                    with open(target_img, 'wb') as img_file:
                        img_file.write(zip_file.read(img))

            # Extract image informations
            else:  # If zip file holds image annotations
                zip_file.extractall(target)

        # Release memory and disk space
        urllib.request.urlcleanup()

    print(
        'Build data annotation files and store them in the image directories'
    )
    instances_file_name = 'instances_{}2017.json'
    captions_file_name = 'captions_{}2017.json'
    stuff_file_name = 'stuff_{}2017.json'
    phases = ['train', 'val']
    anno_files = {
        ph: [
            os.path.join(anno_dir, instances_file_name.format(ph)),
            os.path.join(anno_dir, captions_file_name.format(ph)),
            os.path.join(anno_dir, stuff_file_name.format(ph))
        ] for ph in phases
    }

    for phase in phases:
        data_dir = train_dir if phase == 'train' else valid_dir

        print('\tload {} annotations'.format(phase))
        data = list()
        for file_in in anno_files[phase]:
            with open(file_in) as f:
                data.append(json.load(f))
        instances, captions, stuff = data

        print('\tbuild {} object dictionary'.format(phase))
        data_obj = {
            img['id']: {
                'file_name': img['file_name'],
                'labels': set(),
                'captions': list()
            } for img in instances['images']
        }

        # Build annotation dictionaries
        categories = {
            cat['id']: cat['name'] for cat in instances['categories']
        }
        stuff_names = {st['id']: st['name'] for st in stuff['categories']}

        print('\tadd labels to {} object'.format(phase))
        for inst in instances['annotations']:
            data_obj[inst['image_id']]['labels'].update(
                [categories[inst['category_id']]]
            )

        print('\tadd stuff to {} object'.format(phase))
        for st in stuff['annotations']:
            s = stuff_names[st['category_id']]
            s = s.split('-') if '-' in s else [s]
            data_obj[st['image_id']]['labels'].update(
                [el for el in s if el != 'other']
            )

        # convert labels set to list
        for img in data_obj:
            data_obj[img]['labels'] = list(data_obj[img]['labels'])

        print('\tadd caption to {} object'.format(phase))
        for caption in captions['annotations']:
            data_obj[caption['image_id']]['captions'].append(
                caption['caption']
            )

        print('\twrite {} object to {}'.format(phase, data_dir))
        with open(os.path.join(data_dir, 'data_info.json'), 'w') as file_out:
            json.dump(data_obj, file_out)

    print('all done, COCO dataset is now available in {}'.format(root_dir))
    return


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _download_reporthook(blocknum, blocksize, totalsize):
    """Report hook to print downloading progress of urllib.request.urlretrieve()

    Defines the report hook used to call urllib.request.urlretrieve(url,
    filename=None, reporthook=None, data=None). This function will be called by
    urlretrieve() after each block is received. Thus, the downloading
    percentage is computed and displayed to the standard output for each
    incoming block.
    Args:
        blocknum (int): Number of received blocks
        blocksize (int): Size of the incoming network blocks
        totalsize (int): Size of the incoming file

    Returns:
        None

    """
    # Compute download progress
    progress = int(blocknum * blocksize / totalsize * 100)

    # Choose between carriage return and new line
    if progress < 100:
        str_end = '\r'
    else:
        str_end = '\n'

    # Write download progress to standard output
    sys.stdout.write('\tDownloading: {:3d}% {}'.format(
            progress,
            str_end
        )
    )

    return


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('dataset',
                        help='name of the dataset',
                        type=str,
                        choices=downloadable_datasets)
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=False,
                        type=str,
                        default='./data/')
    args = parser.parse_args()

    getattr(sys.modules[__name__], args.dataset)(args)
