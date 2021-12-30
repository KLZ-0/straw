from .encoder import Encoder


def run(args):
    import timeit, sys
    print(args)

    e = Encoder()

    start = timeit.default_timer()
    e.load_files(args.input_files)
    stop = timeit.default_timer()
    print(f"<TIME> load_files: {stop - start}", file=sys.stderr)

    start = timeit.default_timer()
    e.create_frames()
    stop = timeit.default_timer()
    print(f"<TIME> create_frames: {stop - start}", file=sys.stderr)

    start = timeit.default_timer()
    e.encode()
    stop = timeit.default_timer()
    print(f"<TIME> encode: {stop - start}", file=sys.stderr)

    start = timeit.default_timer()
    e.save_file(args.output_file)
    stop = timeit.default_timer()
    print(f"<TIME> save_file: {stop - start}", file=sys.stderr)
