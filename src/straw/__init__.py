from .codec import Encoder, Decoder


def encode(args):
    import timeit
    import sys

    e = Encoder(args.flac_mode)

    start = timeit.default_timer()
    e.load_files(args.input_files)
    stop = timeit.default_timer()
    print(f"<TIME> load_files: {stop - start}", file=sys.stderr)

    mid = timeit.default_timer()
    e.create_frames()
    stop = timeit.default_timer()
    print(f"<TIME> create_frames: {stop - mid}", file=sys.stderr)

    mid = timeit.default_timer()
    e.encode()
    stop = timeit.default_timer()
    print(f"<TIME> encode: {stop - mid}", file=sys.stderr)

    mid = timeit.default_timer()
    e.save_file(args.output_file)
    stop = timeit.default_timer()
    print(f"<TIME> save_file: {stop - mid}", file=sys.stderr)

    mid = timeit.default_timer()
    e.restore()
    stop = timeit.default_timer()
    print(f"<TIME> restore: {stop - mid}", file=sys.stderr)

    e.print_stats(args.output_file)

    print(f"<TIME> total: {stop - start:.3f} seconds", file=sys.stderr)


def decode(args):
    import timeit
    import sys
    from pathlib import Path

    d = Decoder(args.flac_mode)

    start = timeit.default_timer()
    d.load_file(Path(args.input_files[0]))
    stop = timeit.default_timer()
    print(f"<TIME> load_files: {stop - start}", file=sys.stderr)


def run(args):
    print(args)
    if not args.decode:
        encode(args)
    else:
        decode(args)
