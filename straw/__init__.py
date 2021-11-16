from .encoder import Encoder


def run(args):
    print(args)

    e = Encoder()
    e.load_files(args.input_files)
    e.encode()
