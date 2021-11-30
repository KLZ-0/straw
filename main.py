import argparse

import straw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lossless multi-channel audio codec")
    parser.add_argument("-i", "--input", dest="input_files", metavar="INPUT_FILE", type=str, nargs="+",
                        help="Input files")
    parser.add_argument("-o", "--output", dest="output_file", metavar="OUTPUT_FILE", type=str,
                        help="Output file")
    parser.add_argument("-d", "--decode", dest="decode", action="store_true",
                        help="Decode")

    # TODO: remove this if done with test runs
    parser.add_argument("--figures", dest="figures", action="store_true",
                        help="Generate figures - temporary option for runs with non-consistent behavior")

    args = parser.parse_args()

    if args.figures:
        pass
    else:
        straw.run(args)

