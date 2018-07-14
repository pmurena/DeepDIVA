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
import random

import numpy as np
import torch
from multiprocessing import Pool
# Torch related stuff
import torch.utils.data as data
from util.data.handlers.helpers import Folder
from torch.nn.utils import rnn

from sys import getsizeof


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

        with open(dataset_file, 'rb') as tr:
            self.data = pickle.load(tr)
        self.neg_idx = random.sample(range(len(self)), int(len(self)/2))
        self.seq_len = len(max(self.data, key=len).split())
        self.voc = [Corpus.END_OF_SEQUENCE, Corpus.UNKNOWN]
        with open(vocabulary_file, 'rb') as voc:
            _ = [
                self.voc.append(word)
                for (word, count) in pickle.load(voc).most_common()
                if count >= word_freq
            ]
        self.w2idx = {
            word: idx
            for idx, word in enumerate(self.voc)
        }
        self.idx2w = {
            idx: word
            for word, idx in self.w2idx.items()
        }

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
        neg = True if index in self.neg_idx else False
        seq = list()
        for s in self.data[index].split():
            try:
                seq.append(self.w2idx[s])
            except KeyError:
                seq.append(self.w2idx[Corpus.UNKNOWN])
        if neg:
            random.shuffle(seq)
        original_seq_len = len(seq)
        eos_idx = self.w2idx[Corpus.END_OF_SEQUENCE]
        seq.extend([eos_idx] * (self.seq_len-original_seq_len))
        encoded_padded_seq = torch.tensor(seq)
        label = torch.tensor(int(neg))
        return encoded_padded_seq, original_seq_len, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    print('main() is used for testing only and should not be called otherwise')
    test = Corpus('/home/pat/storage/datasets/wiki/en/test.pkl', '/home/pat/storage/datasets/wiki/en/voc.pkl')

    test[3]
    # print(test[0][0].size())
    # print(test[0:100:30])
    test_data = torch.utils.data.DataLoader(test, batch_size=3)
    print(test_data.batch_size)
    # for data in test_data:
    #     myList = [
    #         data[0],
    #         data[1],
    #         data[2]
    #     ]
    #     input(myList)
    #    lengths, idx = data[1].sort(descending=True)
    #    sequences = list()
    #    labels = list()
    #    for i in idx:
    #        sequences.append(data[0][i].numpy())
    #        labels.append(data[2][i].numpy())
    #    packed_padded_seq = rnn.pack_padded_sequence(
    #        torch.tensor(sequences),
    #        torch.tensor(lengths),
    #        batch_first=True
    #    )
    #    input(getsizeof(packed_padded_seq))
    #print(test[3])
    #for elem in test[3:6]:
    #    print(elem)
    #for data, label in test[500:200:-20]:
    #    print(len(data))
    #for t in test.data:
    #    print(len(t))
    #for (data, label) in test:
    #    da = ''
    #    for idx, d in enumerate(data):
    #        da += ' {}'.format(test.idx2w[int(d)])
    #        if idx == 50:
    #            break
    #    input('{}: {}'.format(label, da))
