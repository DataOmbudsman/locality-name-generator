import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharLSTM(nn.Module):
    def __init__(self, vocab, config):
        super(CharLSTM, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab.chars)
        self.config = config

        self.hidden_size = config.hidden_size
        self.embed_dim = config.embed_dim
        self.n_layers = config.n_layers
        self.dropout_p = config.dropout_p
        
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=vocab.pad_ix
        )
        
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(self.dropout_p)
        
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size - 1  # don't predict PAD
        )

    def forward(self, x_padded, lengths, prev_state):
        embed = self.embedding(x_padded)
        packed_input = pack_padded_sequence(
            embed, lengths, batch_first=True, enforce_sorted=False)
        packed_output, state = self.lstm(packed_input, prev_state)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.dropout(out)
        out = self.fc(out)
        return out, state
    
    def init_state(self, batch_size=1, device="cpu"):
        return (
            torch.zeros(
                self.n_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(
                self.n_layers, batch_size, self.hidden_size).to(device)
        )
