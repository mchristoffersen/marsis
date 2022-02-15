#!/usr/bin/env python

import argparse
import os
import sys
import multiprocessing

import numpy as np

sys.path.append('/home/mchristo/proj/marsis-processor/src/')
import marsis

def process(lblPath):
    edr = marsis.EDR(lblPath)
    nav = edr.geo[["SUB_SC_LATITUDE", "SUB_SC_LONGITUDE", "SPACECRAFT_ALTITUDE"]]
    with open(edr.lbld["PRODUCT_ID"].lower() + "_nav.csv", "w") as fd:
        fd.write(",".join(nav.dtype.names) + "\n")
        np.savetxt(fd, nav, "%.6f", ",")
    f1, f2 = marsis.campbell(edr)

    # Write out radargram and nav products
    f1Path = args.out + "/" + track.lower() + "_f1.img"
    f2Path = args.out + "/" + track.lower() + "_f2.img"

    f1.T.tofile(f1Path)
    f2.T.tofile(f2Path)


# CLI
parser = argparse.ArgumentParser(description="Process MARSIS EDRs")
parser.add_argument("tracks", type=str, help="File containing track IDs to process")
parser.add_argument(
    "-o", "--out", type=str, help="Output directory (Default = ./)", default="./"
)
parser.add_argument(
    "-c", "--cache", type=str, help="EDR cache (Default = ./)", default="./"
)
parser.add_argument(
    "-n", "--num_proc", type=int, help="Number of processes to spawn (Default = 1)", default=1
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
lbls = []
for track in tracks:
    lbls.append(args.cache + "/" + track.lower() + ".lbl")

with multiprocessing.Pool(args.num_proc) as pool:
    pool.map(process, lbls)
