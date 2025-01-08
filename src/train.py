from argparse import ArgumentParser

from torch import nn

from .config import ModelConfig, TrainerConfig
from .data import NamesDataset, get_dataloader, read_file
from .model import CharLSTM
from .saveload import save
from .trainer import Trainer
from .vocab import TOKEN_EOS, TOKEN_SOS, Vocab


def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument('--input', help="Text file to train on")
    parser.add_argument('--output', help="Save model here")

    return parser.parse_args()


def main(args):
    model_config = ModelConfig()
    trainer_config = TrainerConfig()
    device = trainer_config.device

    data_as_char = read_file(args.input, TOKEN_EOS, TOKEN_SOS)
    vocab = Vocab()
    vocab.create_from_file(args.input)

    dataset = NamesDataset(data_as_char, vocab)
    dataloader = get_dataloader(
        dataset,
        trainer_config.batch_size,
        vocab.pad_ix,
        device
    )

    model = CharLSTM(vocab, model_config)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_ix)
    trainer = Trainer(model, criterion, trainer_config)

    print(f"Start training with model: {model}")

    model, _ = trainer.train(
        dataloader,
        print_every=1_000
    )

    save(model, args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
