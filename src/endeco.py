import torch.nn as nn

from src.data_gen import to_ID


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=to_ID()[" "])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, sequence):
        embedding = self.word_embeddings(sequence)
        output, (last_hidden_state, last_cell_state) = self.lstm(embedding)

        # last_cell_state is a tensor, which keeps or deletes information.
        # LSTM gets 3 inputs and returns 3 outputs.
        return (last_hidden_state, last_cell_state)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=to_ID()[" "])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sequence, encoder_state):
        embedding = self.word_embeddings(sequence)
        output, state = self.lstm(embedding, encoder_state)
        output = self.hidden2linear(output)

        return output, state
