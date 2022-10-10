#!/usr/bin/env python

import argparse
import sys
import os
import re

sys.path.append("/home/mchristo/proj/orbitRadar/marsis_processor/src")

import marsis

# To print failure messages and exit
def fail(msg):
    print("\033[91m" + msg + "\033[0m", file=sys.stderr)
    sys.exit(1)


def cli():
    parser = argparse.ArgumentParser(
        description="Download MARSIS EDRs. Takes list of tracks from a file or stdin"
    )
    parser.add_argument(
        "-t",
        "--tracks",
        type=str,
        help="List of MARSIS EDRs to download (label file)",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Directory to save downloaded MARSIS data to (default = ./",
        default="./",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


def main():
    args = cli()

    if args.tracks is None:
        tracks = sys.stdin.readlines()
    else:
        if not os.path.isfile(args.tracks):
            fail('EDR track list "%s" cannot be found' % args.tracks)
        with open(args.tracks, "r") as fd:
            tracks = fd.readlines()

    # Check paths
    if not os.path.isdir(args.output):
        fail('Output directory "%s" cannot be found' % args.out)

    tracks = [track.strip() for track in tracks]

    # Check track file formatting
    marsis_pattern = re.compile("e_[0-9]{5}_ss3_trk_cmp_m")
    for track in tracks:
        if marsis_pattern.match(track) is None:
            fail('Invalid track "%s"' % track)

    marsis.fetch(tracks, args.output, clobber=True)

    return 0


if __name__ == "__main__":
    main()
