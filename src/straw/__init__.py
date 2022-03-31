from .encoder import Encoder


def run(args):
    import timeit, sys
    print(args)

    e = Encoder(args)

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
