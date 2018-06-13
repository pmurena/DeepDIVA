"""
Load a dataset of multitask data by specifying the folder where its located.
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


def load_dataset(dataset_folder, file_name='data_info.json'):
    """ Loads a dataset from disc for each phase: training, validation, and test

    Retrieves annotations and creates Multitask datasets for each phase.
    load_dataset(args) assumes the data is split into train, valid, test on
    disc. If the mentioned folder is not found None will be returned instead of
    a Multitask object.

    Args:
        dataset_folder  (str): Path to the root dataset folder on the file
            system
        file_name (str): Annotation file name. Default: 'data_info.json'
        source_key (str): Key name to retrieve the source image in file_name
        tasks_keys (list(str)): Key names for each task groun-truth in
            file_name

    Returns:
    train (Multitask): training dataset or None if training annotations
        not found
    valid (Multitask): validation dataset or None if training annotations
        not found
    test (Multitask): test dataset or None if training annotations not found
    """
    phases = ['train', 'val', 'test']
    r_val = list()

    # Sanity check on the splits folders
    error_msg = "data_info.json not found in the dataset_folder={}"
    error_msg = error_msg.format(dataset_folder)
    error_msg = '{} ' + error_msg
    for ph in phases:
        dir = os.path.join(dataset_folder, ph)
        if os.path.exists(os.path.join(dir, file_name)):
            r_val.append(Multitask(dir, file_name=file_name))
        else:
            r_val.append(None)
            logging.error(error_msg.format(ph))

    # Get the datasets
    return r_val


class Multitask(data.Dataset):
    """
    This class loads the data_info.json file and prepares it as a dataset.
    """

    def __init__(
        self,
        path,  # Path to dataset root directory
        file_name='data_info.json',  # Name of the annotations file
        source_key='file_name',  # Name of the source data key in data
        tasks_keys=['labels', 'captions']  # Names of the GT keys
    ):
        """ Load the data description file and prepare it as a multitask dataset.

        Multitask learning allows learning multiple objectives at once. This
        dataset can be used to perform such learning tasks if data with several
        ground-truth types is provided. The data is read from a JSON file
        which,for each data sample, contains the image name/path as well as
        ground-truth for each learning task. In order to build itself properly
        the key names for the file name and the various learning tasks, must be
        given.

        Args:
            path  (str): Path to the root dataset folder on the file system
            file_name (str): Annotation file name. Default: 'data_info.json'
            source_key (str): Key name to retrieve the source image in
                file_name
            tasks_keys (list(str)): Key names for each task groun-truth in
                file_name
        """
        self.path = os.path.expanduser(path)
        self.source_key = source_key
        self.tasks_keys = tasks_keys

        # Read data from the json file
        with open(os.path.join(path, file_name)) as f:
            self.data = json.load(f)

        # Shuffle the data once
        # (otherwise you get clusters of samples of same
        # class in each minibatch for val and test)
        np.random.shuffle(self.data)

    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Args:
            index (int): The index of the image to retrieve

        Returns:
            img (numpy.ndarray): the source images
            ground_truth (dict): ground_truth for each task to be learned

        """
        sample = self.data[index]
        img = io.imread(os.path.join(self.path, sample[self.source_key]))
        ground_truth = {gt: sample[gt] for gt in self.tasks_keys}
        return img, ground_truth

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    print('main() is used for testing only and should not be called otherwise')
    train, val, test = load_dataset('data/coco')

    print(len(val))
    print('source type: \n\t{}\nlabels:\n\t{}\ncaptions:\n\t{}\n'.format(
            type(val[0][0]),
            '\n\t'.join(val[0][1]['labels']),
            '\n\t'.join(val[0][1]['captions'])
        )
    )
