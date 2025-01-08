import pickle
import string

TOKEN_EOS = "<EOS>"
TOKEN_PAD = "<PAD>"
TOKEN_SOS = "<SOS>"

class Vocab:
    def __init__(self):
        self.chars = None
        self.ix_to_char = None
        self.char_to_ix = None
        self.pad_ix = None

    def create_from_file(self, filename):
        possible_chars = self._get_possible_chars(filename)
        self.chars = (
            [TOKEN_SOS, TOKEN_EOS] +
            sorted([ch for ch in possible_chars]) +
            [TOKEN_PAD]
        )
        self.ix_to_char = {ix: char for ix, char in enumerate(self.chars)}
        self.char_to_ix = {char: ix for ix, char in self.ix_to_char.items()}
        self.pad_ix = len(self.chars) - 1

    def _get_possible_chars(self, filename):
        if filename:
            with open(filename) as file:
                text = file.read().lower()
            chars = set(text)
            chars.remove('\n')
            return chars
        return string.ascii_lowercase

    def serialize(self):
        data = {
            "chars": self.chars,
            "ix_to_char": self.ix_to_char,
            "char_to_ix": self.char_to_ix,
            "pad_ix": self.pad_ix,
        }
        b = pickle.dumps(data)
        return b

    def load_serialized(self, b):
        data = pickle.loads(b)
        self.chars = data['chars']
        self.ix_to_char = data['ix_to_char']
        self.char_to_ix = data['char_to_ix']
        self.pad_ix = data['pad_ix']
