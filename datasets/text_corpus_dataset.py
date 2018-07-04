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
import torch
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
    dataset_folder = Folder(dataset_folder)
    train = dataset_folder.get_file_name('train.pkl')
    val = dataset_folder.get_file_name('val.pkl')
    test = dataset_folder.get_file_name('test.pkl')
    voc = dataset_folder.get_file_name('voc.pkl')
    sanity_check = [
        dataset_folder.file_exists(train),
        dataset_folder.file_exists(val),
        dataset_folder.file_exists(test),
        dataset_folder.file_exists(voc)
    ]
    error_msg = "wiki files not found in the dataset_folder={}"
    error_msg = error_msg.format(dataset_folder)
    if False in sanity_check:
        print(error_msg)
        sys.exit(-1)

    return Corpus(train, voc), Corpus(val, voc), Corpus(test, voc)


class Corpus(data.Dataset):
    """
    This class loads the data_info.json file and prepares it as a dataset.
    """
    END_OF_SEQUENCE = '<eos>'
    UNKNOWN = '<unk>'

    def __init__(self, dataset_file, vocabulary_file, word_freq=100):
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

        with open(dataset_file, 'rb') as tr:
            self.train = pickle.load(tr)

        with open(vocabulary_file, 'rb') as voc:
            self.voc = [
                word for (word, count) in pickle.load(voc).most_common()
                if count >= word_freq
            ]

        self.voc.append(Corpus.UNKNOWN)
        self.voc.append(Corpus.END_OF_SEQUENCE)

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
                seq.append(self.w2idx[Corpus.UNKNOWN])
        return torch.tensor(seq, dtype=torch.long, requires_grad=True)

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
        return seq[:-1], seq[1:]

    def __len__(self):
        return len(self.train)


if __name__ == "__main__":
    print('main() is used for testing only and should not be called otherwise')
    _, test, _ = load_dataset('~/storage/datasets/wiki/en')
    print(test[1])
    print(test.voc_size())
    test_loader = data.DataLoader(test, batch_size=1)
    for batch, data in enumerate(test_loader):
        print(data)
        if batch == 0:
            break
