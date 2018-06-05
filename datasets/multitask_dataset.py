"""
Load a dataset of bidimensional points by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
from skimage import io
from matplotlib import pyplot as plt


import numpy as np
import json
# Torch related stuff
import torch.utils.data as data


def load_dataset(dataset_folder):
    """
    Loads the dataset from file system and provides the dataset splits for
    train validation and test

    The dataset is expected to be in the following structure, where
    'dataset_folder' has to point to
    the root of the three folder train/val/test.

    Example:

        dataset_folder = "~/../../data/coco"

    which contains the splits sub-folders as follow:

        'dataset_folder'/train
        'dataset_folder'/val
        'dataset_folder'/test

    Parameters
    ----------
    dataset_folder (string): Path to the dataset on the file System

    Returns
    -------
    train_ds (data.Dataset): Dataset containing the training split
    val_ds (data.Dataset): Dataset containing the valitation split
    test_ds (data.Dataset): Dataset containing the test split
    """
    # Get the splits folders
    train_dir = os.path.join(dataset_folder, 'train')
    val_dir = os.path.join(dataset_folder, 'val')
    test_dir = os.path.join(dataset_folder, 'test')

    # Sanity check on the splits folders
    error_msg = "data_info.json not found in the dataset_folder={}"
    error_msg = error_msg.format(dataset_folder)
    error_msg = '{} ' + error_msg
    if not os.path.exists(train_dir):
        logging.error(error_msg.format('Train'))
        sys.exit(-1)
    if not os.path.exists(val_dir):
        logging.error(error_msg.format('Val'))
        sys.exit(-1)

    # Get the datasets
    return Multitask(train_dir), Multitask(val_dir)


class Multitask(data.Dataset):
    """
    This class loads the data_info.json file and prepares it as a dataset.
    """

    def __init__(self, path, transform=None, target_transform=None):
        """
        Load the data_info.json file and prepare it as a dataset.

        Parameters
        ----------
        path (string): Path to the dataset on the file System
        transform (torchvision.transforms): Transformation to apply on the data
        target_transform (torchvision.transforms):
                Transformation to apply on the labels
        """
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # Read data from the json file
        with open(os.path.join(path, 'data_info.json')) as f:
            self.data = json.load(f)

        # Shuffle the data once
        # (otherwise you get clusters of samples of same
        # class in each minibatch for val and test)
        np.random.shuffle(self.data)

        # Set expected class attributes
        self.vocabulary = set([
            word for elems in self.data for word in elems['labels']
        ])
        self.corpus = '\n'.join([
            caption for elems in self.data for caption in elems['captions']
        ])

    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Parameters
        ----------
        index (int): The index of the image to retrieve

        Returns
        -------

        """
        sample = self.data[index]
        img = io.imread(os.path.join(self.path, sample['file_name']))
        return img, sample['labels'], sample['captions']

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    print('main() is used for testing only and should not be called otherwise')
    train, val = load_dataset('data/coco')

    print(len(train))
    print(val[0][1])
    print('\n'.join(val[0][2]))
    plt.imshow(val[0][0])
