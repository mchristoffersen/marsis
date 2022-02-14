#!/usr/bin/env python

import argparse
import os
import sys

import numpy as np

import marsis

# CLI
parser = argparse.ArgumentParser(description="Process MARSIS EDRs")
parser.add_argument("tracks", type=str, help="File containing track IDs to process")
parser.add_argument(
    "-o", "--out", type=str, help="Output directory (Default = ./)", default="./"
)
parser.add_argument(
    "-c", "--cache", type=str, help="EDR cache (Default = ./)", default="./"
)
args = parser.parse_args()

# Check out and cache
if not os.path.isdir(args.out):
    print('Output directory "%s" cannot be found' % args.out, file=sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.cache):
    print('Cache directory "%s" cannot be found' % args.cache, file=sys.stderr)
    sys.exit(1)

# Read in track file
if not os.path.isfile(args.tracks):
    print('Track list argument "%s" cannot be found' % args.tracks, file=sys.stderr)
    sys.exit(1)

fd = open(args.tracks, "r")
tracks = fd.read().split()
fd.close()

# Fetch files
marsis.fetch(tracks, args.cache)

# Generate EDR and process each file
for track in tracks:
    lblPath = args.cache + "/" + track.lower() + ".lbl"
    edr = marsis.EDR(lblPath)
    nav = edr.geo[["SUB_SC_LATITUDE", "SUB_SC_LONGITUDE", "SPACECRAFT_ALTITUDE"]]
    with open(edr.lbld["PRODUCT_ID"].lower() + "_nav.csv", "w") as fd:
        fd.write(",".join(nav.dtype.names) + "\n")
        np.savetxt(fd, nav, "%.6f", ",")
    print(nav)
    f1, f2 = marsis.campbell(edr)

    # Write out radargram and nav products
    f1Path = args.out + "/" + track.lower() + "_f1.img"
    f2Path = args.out + "/" + track.lower() + "_f2.img"

    f1.T.tofile(f1Path)
    f2.T.tofile(f2Path)
