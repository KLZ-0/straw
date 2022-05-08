import sys
import timeit
from pathlib import Path

import numpy as np

from straw import Encoder, Decoder


def read(file) -> (np.array, int):
    """
    Read a Straw file (compatibility function to soundfile.read)
    :param file: Input file
    :return: soundfile compatible array and samplerate
    """
    d = Decoder()
    d.load_file(Path(file))
    d.decode()
    return d.get_soundfile_compatible_array(), d.get_params().sample_rate


def write(file, data: np.array, samplerate: int):
    """
    Write a Straw file (compatibility function to soundfile.write)
    :param file: Output file
    :param data: Data to be written
    :param samplerate: Sample rate of the given data
    :return: None
    """
    e = Encoder()
    e.load_data(data, samplerate, data.dtype.itemsize * 8)
    e.encode()
    e.save_file(Path(file).open("wb"))


def _encode(args):
    """
    Internal encoder call from the executable script
    :param args: command line args (see straw/__main__.py)
    :return: None
    """
    e = Encoder(flac_mode=False,
                dynamic_blocksize=args.dynamic_blocksize,
                min_block_size=args.min_frame_size,
                max_block_size=args.max_frame_size,
                framing_treshold=args.framing_treshold,
                framing_resolution=args.framing_resolution,
                responsiveness=args.rice_responsiveness,
                parallelize=args.parallel,
                show_progress=False)

    if not args.silent:
        print(f"Loading file ... ", end="", file=sys.stderr)
    start = timeit.default_timer()
    e.load_file(Path(args.input_files[0]))
    stop = timeit.default_timer()
    if not args.silent:
        print(f"DONE in {stop - start:.3f} s", file=sys.stderr)

    if not args.silent:
        print(f"Encoding ... ", end="", file=sys.stderr)
    mid = timeit.default_timer()
    e.encode()
    stop = timeit.default_timer()
    if not args.silent:
        print(f"DONE in {stop - mid:.3f} s", file=sys.stderr)

    if not args.silent:
        print(f"Saving to file ... ", end="", file=sys.stderr)
    mid = timeit.default_timer()
    e.save_file(args.output_file)
    stop = timeit.default_timer()
    if not args.silent:
        print(f"DONE in {stop - mid:.3f} s", file=sys.stderr)

    if args.verbose and not args.silent:
        e.print_stats(args.output_file, stream=sys.stderr)

    if not args.silent:
        print(f"Total processing time: {stop - start:.3f} s", file=sys.stderr)


def _decode(args):
    """
    Internal decoder call from the executable script
    :param args: command line args (see straw/__main__.py)
    :return: None
    """
    d = Decoder(flac_mode=False, show_progress=False)

    if not args.silent:
        print(f"Loading file ... ", end="", file=sys.stderr)
    start = timeit.default_timer()
    d.load_file(Path(args.input_files[0]))
    stop = timeit.default_timer()
    if not args.silent:
        print(f"DONE in {stop - start:.3f} s", file=sys.stderr)

    if not args.silent:
        print(f"Decoding ... ", end="", file=sys.stderr)
    mid = timeit.default_timer()
    d.decode()
    stop = timeit.default_timer()
    if not args.silent:
        print(f"DONE in {stop - mid:.3f} s", file=sys.stderr)

    if not args.silent:
        print(f"Saving to file ... ", end="", file=sys.stderr)
    mid = timeit.default_timer()
    d.save_file(args.output_file)
    stop = timeit.default_timer()
    if not args.silent:
        print(f"DONE in {stop - mid:.3f} s", file=sys.stderr)

    # if Path(args.input_files[0]).stem == "1min":
    #     d.test()

    if not args.silent:
        print(f"Total processing time: {stop - start:.3f} s", file=sys.stderr)


def run(args):
    """
    Run the encoder/decoder
    :param args: command line args (see straw/__main__.py)
    :return: None
    """
    if args.verbose and not args.silent:
        print(args, file=sys.stderr)
    if not args.decode:
        _encode(args)
    else:
        _decode(args)
