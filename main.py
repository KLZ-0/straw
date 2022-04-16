import argparse
from pathlib import Path

import straw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lossless multi-channel audio codec")
    parser.add_argument("-i", "--input", dest="input_files", metavar="INPUT_FILE", type=str, nargs="+",
                        help="Input files", required=True)
    parser.add_argument("-o", "--output", dest="output_file", metavar="OUTPUT_FILE", type=str,
                        help="Output file")
    parser.add_argument("-d", "--decode", dest="decode", action="store_true",
                        help="Decode")
    parser.add_argument("-f", "--flac", dest="flac_mode", action="store_true",
                        help="FLAC mode")

    # TODO: remove this if done with test runs
    parser.add_argument("--figures", dest="figures", action="store_true",
                        help="Generate figures - temporary option for runs with non-consistent behavior")
    parser.add_argument("--no-dynamic-blocks", dest="dynamic_blocksize", action="store_false",
                        help="Use dynamic blocksize, by default delimited by short-term energy")
    parser.add_argument("--figures-show", dest="fig_show", action="store_true", help="Show figures")
    parser.add_argument("--figures-dir", dest="fig_dir", metavar="FIG_DIR", type=str,
                        default="outputs", help="Override figure directory (default='outputs')")

    args = parser.parse_args()

    # Fix args
    if args.output_file:
        args.output_file = Path(args.output_file)
    else:
        pth = Path(args.input_files[0])
        args.output_file = (pth.parent / pth.stem).with_suffix(".straw")

    # TODO: this part has to get proper args

    if args.figures:
        from figures import plot_all

        plot_all(args)
    else:
        straw.run(args)
