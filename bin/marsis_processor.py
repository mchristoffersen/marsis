#!/usr/bin/env python

import argparse
import os
import sys
import re

import numpy as np

import marsis

# To print failure messages and exit
def fail(msg):
    print("\033[91m" + msg + "\033[0m", file=sys.stderr)
    sys.exit(1)


def process(args):
    name = os.path.basename(args.track).replace(".lbl", "")

    edr = marsis.EDR(args.track)

    lat = edr.geo["SUB_SC_LATITUDE"]
    lon = (edr.geo["SUB_SC_LONGITUDE"] + 180) % 360 - 180
    alt = edr.geo["SPACECRAFT_ALTITUDE"]

    bands = np.array([1.8, 3, 4, 5])
    f1_center = bands[edr.ost["DCG_CONFIGURATION_F1"]]
    f2_center = bands[edr.ost["DCG_CONFIGURATION_F2"]]

    if args.method == "none":
        f1, f2 = marsis.plain(edr)
        info = np.dstack((lat, lon, alt, f1_center, f2_center))[0]
        names = "SUB_SC_LATITUDE,SUB_SC_LONGITUDE,SPACECRAFT_ALTITUDE,BAND_F1,BAND_F2"
        fmt = "%.6f,%.6f,%.3f,%.1f,%.1f"
    elif args.method == "campbell":
        f1, f2, f1_rate, f2_rate = marsis.campbell(edr, args.dem)
        info = np.dstack((lat, lon, alt, f1_center, f2_center, f1_rate, f2_rate))[0]
        names = "SUB_SC_LATITUDE,SUB_SC_LONGITUDE,SPACECRAFT_ALTITUDE,BAND_F1,BAND_F2,RATE_F1,RATE_F2"
        fmt = "%.6f,%.6f,%.3f,%.1f,%.1f,%.6e,%.6e"
    elif args.method == "mcmichael":
        f1, f2, f1_psis, f2_psis = marsis.mcmichael(edr, args.sim)
        info = np.dstack(
            (
                lat,
                lon,
                alt,
                f1_center,
                f2_center,
                f1_psis["psi1"],
                f1_psis["psi2"],
                f1_psis["psi3"],
                f2_psis["psi1"],
                f2_psis["psi2"],
                f2_psis["psi3"],
            )
        )[0]
        names = "SUB_SC_LATITUDE,SUB_SC_LONGITUDE,SPACECRAFT_ALTITUDE,BAND_F1,BAND_F2,p1_F1,p2_F1,p3_F1,p1_F2,p2_F2,p3_F2"
        fmt = "%.6f,%.6f,%.3f,%.1f,%.1f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e"
    with open(args.out + "/" + name + "_nav.csv", "w") as fd:
        fd.write(names + "\n")
        np.savetxt(fd, info, fmt)

    # Write out radargram and nav products
    f1Path = args.out + "/" + name + "_f1.img"
    f2Path = args.out + "/" + name + "_f2.img"

    f1.T.tofile(f1Path)
    f2.T.tofile(f2Path)

    marsis.gen_tiff(f1, f1Path.replace(".img", ".tif"))
    marsis.gen_tiff(f2, f2Path.replace(".img", ".tif"))

    marsis.gen_segy(f1, edr, f1Path.replace(".img", ".sgy"))
    marsis.gen_segy(f2, edr, f2Path.replace(".img", ".sgy"))

    return 0


def cli():
    parser = argparse.ArgumentParser(description="Process MARSIS EDRs")
    parser.add_argument(
        "track", type=str, help="MARSIS EDR file to process (label file)"
    )
    parser.add_argument(
        "method",
        type=str,
        choices=["none", "campbell", "mcmichael"],
        help="Ionosphere compensation method",
    )
    parser.add_argument(
        "-d",
        "--dem",
        type=str,
        help="Global digital elevation model of Mars (reqd. for campbell method)",
        default=None,
    )
    parser.add_argument(
        "-s",
        "--sim",
        type=str,
        help="Surface clutter simulation (reqd. for mcmichael method)",
        default=None,
    )
    parser.add_argument(
        "-o", "--out", type=str, help="Output directory (Default = ./)", default="./"
    )
    parser.add_argument(
        "-n",
        "--num_proc",
        type=int,
        help="Number of processes to spawn (Default = 1)",
        default=1,
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


def check_args(args):
    # Check user's arguments
    # TODO: Check format of track, dem, sim files

    args.out = os.path.abspath(args.out)
    args.track = os.path.abspath(args.track)

    if args.method == "campbell":
        if args.dem is None:
            fail("Global DEM unspecified. Required for campbell ionosphere correction.")
        args.dem = os.path.abspath(args.dem)
        if not os.path.isfile(args.dem):
            fail('Global DEM "%s" cannot be found' % args.dem)

    if args.method == "mcmichael":
        if args.sim is None:
            fail(
                "Surface clutter simulation unspecified. Required for mcmichael ionosphere correction."
            )
        args.sim = os.path.abspath(args.sim)
        if not os.path.isfile(args.sim):
            fail('Surface clutter simulation "%s" cannot be found' % args.sim)

    # Check paths
    if not os.path.isdir(args.out):
        fail('Output directory "%s" cannot be found' % args.out)

    if not os.path.isfile(args.track):
        fail('EDR label file "%s" cannot be found' % args.track)

    if not os.path.isfile(args.track.replace(".lbl", "_f.dat")):
        fail('EDR data file "%s" cannot be found' % args.data)

    if not os.path.isfile(args.track.replace(".lbl", "_g.dat")):
        fail('EDR data file "%s" cannot be found' % args.data)

    if args.verbose:
        print("EDR label file:\t\t%s" % args.track)
        print("EDR data file:\t\t%s" % args.track.replace(".lbl", "_f.dat"))
        print("EDR data file:\t\t%s" % args.track.replace(".lbl", "_g.dat"))
        print("Output directory:\t%s" % args.out)
        if args.dem is not None:
            print("Global DEM:\t\t%s" % args.dem)
        if args.sim is not None:
            print("Clutter simulation:\t%s" % args.sim)

    return args


def main():
    args = cli()
    args = check_args(args)
    process(args)
    return 0


if __name__ == "__main__":
    main()
