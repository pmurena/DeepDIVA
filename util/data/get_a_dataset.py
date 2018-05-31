"""
This script allows for creation of a validation set from the training set.

"""

# Utils
import argparse
import inspect
import os
import shutil
import sys

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
    "args.output_folder" is not set. Train, valid, and test hold the image
    files for training, validation, and testing. coco(args) will only alter the
    original COCO folder structure; the data split remains unchanged. All
    descriptions files, later used to build the data loader, are stored in the
    annotations folder.

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
        responce = request.urlretrieve(
            source,
            './tmp_file_19800223',
            _download_reporthook
        )

        # Extract archive
        with ZipFile(responce[0]) as zip_file:
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
        request.urlcleanup()

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
    stdout.write('\tDownloading: {:3d}% {}'.format(
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
