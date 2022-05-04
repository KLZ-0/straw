import argparse
from pathlib import Path

import straw
from straw.static import Default


def main():
    parser = argparse.ArgumentParser(description="Lossless multi-channel audio codec")
    parser.add_argument("-i", "--input", dest="input_files", metavar="INPUT_FILE", type=str, nargs="+",
                        help="Input files", required=True)
    parser.add_argument("-o", "--output", dest="output_file", metavar="OUTPUT_FILE", type=str,
                        help="Output file")
    parser.add_argument("-d", "--decode", dest="decode", action="store_true",
                        help="Decode")

    parser.add_argument("--figures", dest="figures", action="store_true",
                        help="Generate figures - temporary option for runs with non-consistent behavior")
    parser.add_argument("--no-dynamic-blocks", dest="dynamic_blocksize", action="store_false",
                        help="Use dynamic blocksize, by default delimited by short-term energy")
    parser.add_argument("--figures-show", dest="fig_show", action="store_true", help="Show figures")
    parser.add_argument("--figures-dir", dest="fig_dir", metavar="FIG_DIR", type=str,
                        default="outputs", help="Override figure directory (default='outputs')")

    parser.add_argument("--min-frame-size", dest="min_frame_size", metavar="MIN_FRAME_SIZE", type=int,
                        default=Default.min_frame_size, help="Override minimal frame size")
    parser.add_argument("--max-frame-size", dest="max_frame_size", metavar="MAX_FRAME_SIZE", type=int,
                        default=Default.max_frame_size, help="Override maximal frame size")
    parser.add_argument("--framing-threshold", dest="framing_treshold", metavar="THRESHOLD", type=int,
                        default=Default.framing_treshold, help="Override framing treshold")
    parser.add_argument("--framing-resolution", dest="framing_resolution", metavar="RESOLUTION", type=int,
                        default=Default.framing_resolution, help="Override framing resolution")
    parser.add_argument("--rice-responsiveness", dest="rice_responsiveness", metavar="RESPONSIVENESS", type=int,
                        default=Default.rice_responsiveness, help="Override Rice codding responsiveness")

    args = parser.parse_args()

    if args.min_frame_size > args.max_frame_size:
        raise ValueError("Minimal frame size can't be larger than maximal frame size")

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
