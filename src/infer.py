from argparse import ArgumentParser

from .generator import WordGenerator
from .saveload import load


def parse_arguments():
    parser = ArgumentParser()
    
    parser.add_argument('--prefix', help="How the word should start")
    parser.add_argument('--model', help="Trained model")
    parser.add_argument('--temperature', help="Temperature", default=1.0)
    parser.add_argument('--topk', help="Top k", default=10)
    parser.add_argument('--topp', help="Top p", default=0.9)
    parser.add_argument('--max_len', help="Max length", default=20)

    return parser.parse_args()


def main(args):
    model, vocab = load(args.model)
    generator = WordGenerator(
        model,
        vocab,
        block_file=None,
        top_k=args.topk,
        top_p=args.topp,
        temperature=args.temperature,
    )
    generated = generator.generate_word(
        prefix=args.prefix,
        max_length=args.max_len,
    )
    print(generated)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
