"""
Load a dataset of text corpus by specifying the folder where its located.
"""

# Utils
import logging
import os
import sys
import re

import pickle
import collections

import numpy as np
# Torch related stuff
import torch.utils.data as data
from util.data.handlers.helpers import Folder


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
    return Corpus(Folder(dataset_folder))


class Corpus(data.Dataset):
    """
    This class loads the data_info.json file and prepares it as a dataset.
    """
    END_OF_SEQUENCE = '<eos>'
    UNKNOWN = '<unk>'

    def __init__(self, dataset_folder):
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
        err_msg = '{} expected to be of type util.data.handlers.helpers.Folder'

        self.data = list()

        with open(dataset_folder.get_file_name('train.pickle'), 'rb') as tr:
            self.train = pickle.load(tr)

        with open(
            dataset_folder.get_file_name('vocabulary.pickle'),
            'rb'
        ) as voc:
            self.voc = [
                word for (word, count) in pickle.load(voc).most_common()
                if count >= 100
            ]

        self.voc.append(Corpus.END_OF_SEQUENCE)
        self.voc.append(Corpus.UNKNOWN)

        self.w2idx = {
            word: idx
            for idx, word in enumerate(self.voc)
        }

    def encode_seq(self, sequence):
        seq = list()
        sequence += ' {}'.format(Corpus.END_OF_SEQUENCE)
        for s in sequence.split():
            try:
                seq.append(self.w2idx[s])
            except KeyError:
                seq.append(Corpus.UNKNOWN)
        return seq

    def voc_size(self):
        return(len(self.voc))

    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Args:
            index (int): The index of the image to retrieve

        Returns:
            img (numpy.ndarray): the source images
            ground_truth (dict): ground_truth for each task to be learned

        """
        seq = self.encode_seq(self.train[index])
        r_val = {
            'word': list(),
            'target': list()
        }
        for idx in range(len(seq)-1):
            r_val['word'].append(seq[idx])
            r_val['target'].append(seq[idx+1])
        return r_val

    def __len__(self):
        return len(self.train)


if __name__ == "__main__":
    print('main() is used for testing only and should not be called otherwise')
    test = load_dataset('~/storage/datasets/wiki/en')
    print(test[0])
