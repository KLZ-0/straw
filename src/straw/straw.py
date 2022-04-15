import sys
import timeit
from pathlib import Path

import numpy as np

from straw import Encoder, Decoder


def read(file) -> (np.array, int):
    d = Decoder()
    d.load_file(Path(file))
    d.decode()
    return d.get_soundfile_compatible_array(), d.get_params().sample_rate


def write(file, data: np.array, samplerate: int):
    e = Encoder()
    e.load_data(data, samplerate)
    e.encode()
    e.save_file(Path(file))


def _encode(args):
    e = Encoder(args.flac_mode, do_corrections=True, dynamic_blocksize=args.dynamic_blocksize)

    start = timeit.default_timer()
    e.load_file(Path(args.input_files[0]))
    stop = timeit.default_timer()
    print(f"<TIME> load_file: {stop - start}", file=sys.stderr)

    mid = timeit.default_timer()
    e.encode()
    stop = timeit.default_timer()
    print(f"<TIME> encode: {stop - mid}", file=sys.stderr)

    mid = timeit.default_timer()
    e.save_file(args.output_file)
    stop = timeit.default_timer()
    print(f"<TIME> save_file: {stop - mid}", file=sys.stderr)

    e.print_stats(args.output_file)

    print(f"<TIME> total: {stop - start:.3f} seconds", file=sys.stderr)


def _decode(args):
    d = Decoder(args.flac_mode)

    start = timeit.default_timer()
    d.load_file(Path(args.input_files[0]))
    stop = timeit.default_timer()
    print(f"<TIME> load_file: {stop - start}", file=sys.stderr)

    mid = timeit.default_timer()
    d.decode()
    stop = timeit.default_timer()
    print(f"<TIME> decode: {stop - mid}", file=sys.stderr)

    mid = timeit.default_timer()
    d.save_file(args.output_file)
    stop = timeit.default_timer()
    print(f"<TIME> save_file: {stop - mid}", file=sys.stderr)

    d.test()

    print(f"<TIME> total: {stop - start:.3f} seconds", file=sys.stderr)


def run(args):
    print(args)
    if not args.decode:
        _encode(args)
    else:
        _decode(args)
