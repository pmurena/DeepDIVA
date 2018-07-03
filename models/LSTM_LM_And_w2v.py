import torch
import torch.nn as nn


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

    def __init__(self, output_channels=248165, **kwargs):
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
        self.hidden_dim = self.embedding_dim

        self.word_embeddings = nn.Embedding(
            self.vocabulary_size,
            self.embedding_dim
        )

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.hidden2word = nn.Linear(
            self.hidden_dim,
            self.vocabulary_size
        )
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            torch.zeros(1, 1, self.hidden_dim),
            torch.zeros(1, 1, self.hidden_dim)
        )

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1),
            self.hidden
        )
        tag_space = self.hidden2word(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
