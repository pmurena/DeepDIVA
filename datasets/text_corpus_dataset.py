"""
Load a dataset of text corpus by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
import re

import numpy as np
# Torch related stuff
import torch.utils.data as data


def load_dataset(dataset_folder):
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

    # Sanity check on the splits folders
    error_msg = "wiki files not found in the dataset_folder={}"
    error_msg = error_msg.format(dataset_folder)
    error_msg = '{} ' + error_msg
    r_val = Corpus(dataset_folder)

    if len(r_val) > 0:
        return r_val
    else:
        logging.error(error_msg.format(dataset_folder))
        return None


class Corpus(data.Dataset):
    """
    This class loads the data_info.json file and prepares it as a dataset.
    """

    def __init__(self, path):
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
        self.data = [
            os.path.join(ls[0], f)
            for ls in os.walk(self.path)
            for f in ls[2]
            if re.match('wiki_[0-9]{2}_clean', f)
        ]

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
        with open(self.data[index], 'r') as file:
            return file.read()

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    print('main() is used for testing only and should not be called otherwise')
    test = load_dataset('data/wiki')

    inst_test = lambda x: True if isinstance(x, Corpus) else False

    print(test[0])
    print(inst_test(test))
    print(inst_test('models'))
    print(inst_test('model'))
    print(len(test))
