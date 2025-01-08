from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class NamesDataset(Dataset):
    def __init__(self, data_as_char, vocab):
        self.vocab = vocab
        self.data_as_int = [
            [self.vocab.char_to_ix[char] for char in char_seq]
            for char_seq in data_as_char
        ]
        self.seq_lengths = [len(seq) - 2 for seq in self.data_as_int]

    def __len__(self):
        return len(self.data_as_int)

    def __getitem__(self, ix):
        item = self.data_as_int[ix]
        x = item[:-1]
        y = item[1:]
        return torch.tensor(x), torch.tensor(y), len(x)


def read_file(filename, token_eos, token_sos):
    data = []
        
    with open(filename) as file:
        text = file.read().lower()

    names = text.splitlines()
    for name in iter(names):
        chars = [token_sos] + list(name) + [token_eos]
        data.append(chars)
    return data


def collate_fn(batch, pad_ix, device):
    xs, ys, lengths = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=pad_ix)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=pad_ix)
    return xs_padded.to(device), ys_padded.to(device), torch.tensor(lengths)


def get_dataloader(dataset, batch_size, pad_ix, device):
    collate_fn_partial = partial(collate_fn, pad_ix=pad_ix, device=device)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_partial
    )
    return dataloader
