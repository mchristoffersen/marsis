#!/usr/bin/env python

import argparse
import multiprocessing
import os
import sys

import numpy as np

import marsis


def process(track, args):
    lblPath = args.cache + "/" + track.lower() + ".lbl"
    edr = marsis.EDR(lblPath)
    nav = edr.geo[["SUB_SC_LATITUDE", "SUB_SC_LONGITUDE", "SPACECRAFT_ALTITUDE"]]
    with open(args.out + "/" + edr.lbld["PRODUCT_ID"].lower() + "_nav.csv", "w") as fd:
        fd.write(",".join(nav.dtype.names) + "\n")
        np.savetxt(fd, nav, "%.6f", ",")
    f1, f2 = marsis.campbell(edr, args.dem, cacheIono=True, cache=args.cache)

    # Write out radargram and nav products
    f1Path = args.out + "/" + track.lower() + "_f1.img"
    f2Path = args.out + "/" + track.lower() + "_f2.img"

    f1.T.tofile(f1Path)
    f2.T.tofile(f2Path)

    marsis.gen_tiff(f1, f1Path.replace(".img", ".tif"))
    marsis.gen_tiff(f2, f2Path.replace(".img", ".tif"))

    marsis.gen_segy(f1, edr, f1Path.replace(".img", ".sgy"))
    marsis.gen_segy(f2, edr, f2Path.replace(".img", ".sgy"))

    return 0


# CLI
parser = argparse.ArgumentParser(description="Process MARSIS EDRs")
parser.add_argument("tracks", type=str, help="File containing track IDs to process")
parser.add_argument("dem", type=str, help="Global digital elevation model of Mars")
parser.add_argument(
    "-o", "--out", type=str, help="Output directory (Default = ./)", default="./"
)
parser.add_argument(
    "-c", "--cache", type=str, help="EDR cache (Default = ./)", default="./"
)
parser.add_argument(
    "-n",
    "--num_proc",
    type=int,
    help="Number of processes to spawn (Default = 1)",
    default=1,
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
marsis.fetch(tracks, args.cache, clobber=False)

# Generate EDR and process each file
with multiprocessing.Pool(args.num_proc) as pool:
    pool.starmap(process, zip(tracks, [args] * len(tracks)))
