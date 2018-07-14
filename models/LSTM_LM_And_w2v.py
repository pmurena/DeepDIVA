import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import rnn

import random
from datasets import text_corpus_dataset

import pprint
'''
python template/RunMe.py --experiment-name build_wiki_lm --model-name LSTM_LM_And_w2v --batch-size 300 --output-folder log --dataset-folder ~/storage/datasets/wiki/en/ --lr 0.1 --ignoregit
'''

class LSTM_LM_And_w2v(nn.Module):
    """
    Simple feed forward neural network

    Attributes
    ----------
    fc1 : torch.nn.Linear
        Fully connected layer of the network
    cl : torch.nn.Linear
        Final classification fully connected layer
    """

    def __init__(self, output_channels=2, **kwds):
        """
        Creates an LSTM language model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        """
        super(LSTM_LM_And_w2v, self).__init__()

        self.vocabulary_size = output_channels
        self.embedding_dim = 500
        self.hidden_dim = 100

        self.word_embeddings = nn.Embedding(
            self.vocabulary_size,
            self.embedding_dim
        )
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim,
            batch_first=True
        )
        self.hidden = self.init_hidden()
        s_fac = int(self.hidden_dim / 2)
        self.hidden_to_small = nn.Linear(
            self.hidden_dim,
            s_fac
        )
        self.small_to_language = nn.Linear(
            s_fac,
            2
        )

    def init_hidden(self, batch_size=1):
        return (
            torch.zeros(1, batch_size, self.hidden_dim).cuda(async=True),
            torch.zeros(1, batch_size, self.hidden_dim).cuda(async=True)
        )

    def forward(self, data):
        sequence, lengths = data
        padded_seq = rnn.pad_sequence(
            sequence,
            batch_first=True,
            padding_value=0
        )
        embeds = self.word_embeddings(padded_seq)
        packed_padded_seq = rnn.pack_padded_sequence(
            embeds,
            lengths,
            batch_first=True
        )
        self.hidden = self.init_hidden(padded_seq.size(0))
        # self.lstm.flatten_parameters()
        lstm_out, self.hidden = self.lstm(
            packed_padded_seq,
            self.hidden
        )
        hidden = self.hidden[0]
        small = self.hidden_to_small(hidden)
        return self.small_to_language(small)


def sort_sequences_desc_order(data):
    lengths, idx = data[1].sort(descending=True)
    se = data[0].numpy()
    la = data[2].numpy()
    sequences = torch.tensor(se[idx], dtype=torch.long)
    labels = torch.tensor(la[idx], dtype=torch.long)
    return sequences, lengths, labels


if __name__ == '__main__':
    print('load data')
    pp = pprint.PrettyPrinter(indent=4)
    data_folder = '/home/pat/storage/datasets/wiki/en/'
    train, test, val = text_corpus_dataset.load_dataset(data_folder)
    print('init training')
    model = LSTM_LM_And_w2v(train.voc_size())
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.cuda(0)
    best = 0
    acc = 0
    batch_size = 3
    val_batch_size = batch_size * 4
    print('prepare data loader')
    train_data = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val, batch_size=val_batch_size)
    print('start training')
    for nb, tr_data in enumerate(train_data):
        nb = nb + 1
        sequences, lengths, labels = sort_sequences_desc_order(tr_data)

        sequences = sequences.cuda()
        lengths = lengths.cuda()
        labels = labels.cuda()

        model.zero_grad()
        res = model((sequences, lengths))
        loss = loss_function(res.view(batch_size, 2), labels)
        loss.backward()
        optimizer.step()
        pred = res.argmax(2)
        comp = pred == labels
        batch_acc = int(sum(comp[0]))/batch_size
        msg = 'batch: {:>10,} batch_acc: {:.2%} loss: {:6.4f} '
        msg = msg + 'last val acc: {:.2%} current best acc: {:.2%}'
        print(msg.format(
             nb, batch_acc, loss.item(), acc, best
         )
        )
        if nb % 1 == 0:
            torch.save(
                model.state_dict(),
                '{}save/{}.pth.tar'.format(data_folder, nb)
            )
            acc = list()
            for i, vl_data in enumerate(val_loader):
                data, length, target = sort_sequences_desc_order(vl_data)
                data = data.cuda(0)
                length = length.cuda(0)
                target = target.cuda(0)
                val_res = model((data, length))
                pred = val_res.argmax(2)
                comp = pred == target
                acc.append(int(sum(comp[0]))/val_batch_size)
                r_acc = sum(acc)/len(acc)
                print('validation: {} accuracy: {:.2%}'.format(i, r_acc))
            if best < acc:
                torch.save(
                    model.state_dict(),
                    '{}save/best.pth.tar'.format(data_folder)
                )
                best = acc
