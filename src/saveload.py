import torch

from .config import ModelConfig
from .model import CharLSTM
from .vocab import Vocab


def save(model, path):
    vocab_seralized = model.vocab.serialize()
    data = {
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "vocab": vocab_seralized,
    }
    torch.save(data, path)


def load(path):
    data = torch.load(path)
    state_dict = data["model_state_dict"]

    config_dict = data["config"]
    config = ModelConfig.from_dict(config_dict)

    vocab = Vocab()
    vocab.load_serialized(data["vocab"])

    model = CharLSTM(vocab, config)
    model.load_state_dict(state_dict)
    return model, vocab
