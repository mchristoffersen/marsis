import io
import os
import re
import sys
import urllib.request

import pandas as pd


def fetch(tracks, path, clobber=True):
    """Download MARSIS data from the PDS.

    Download MARSIS data from the PDS by searching the PDS
    index files to match user supplied file names.

    Parameters
    ----------
    tracks: list of str
        List of the MARSIS data files to download, each entry in the
        format "E_XXXXX_SS3_TRK_CMP_M"
    path: str
        Path to directory where the MARSIS data files will be downloaded to
    clobber: bool
        Whether to overwrite (re-download) files at path

    Returns
    -------
    files: list of str
        Tuple containing the paths of all files downloaded by this function

    Notes
    -----
    """

    # Only allow ss3_trk_cmp for now, raise error if anything different
    r = re.compile("e_[0-9]{5}_ss3_trk_cmp_m")
    for track in tracks:
        if r.match(track.lower()) is None:
            print(track)
            raise ValueError("Processor only supports E_SS3_TRK_CMP type")

    # Get index files from pds
    print("Downloading PDS index files...")

    baseurl = "https://pds-geosciences.wustl.edu/mex/"
    indxURLs = [
        baseurl + "mex-m-marsis-2-edr-v2/mexme_1001/index/index.tab",
        baseurl + "mex-m-marsis-2-edr-ext1-v2/mexme_1002/index/index.tab",
        baseurl + "mex-m-marsis-2-edr-ext2-v1/mexme_1003/index/index.tab",
        baseurl + "mex-m-marsis-2-edr-ext3-v1/mexme_1004/index/index.tab",
        baseurl + "mex-m-marsis-2-edr-ext4-v1/mexme_1005/index/index.tab",
        baseurl + "mex-m-marsis-2-edr-ext5-v1/mexme_1006/index/index.tab",
        baseurl + "mex-m-marsis-2-edr-ext6-v1/mexme_1007/index/index.tab",
        baseurl + "mex-m-marsis-2-edr-ext7-v1/mexme_1008/index/index.tab",
    ]

    index = ""

    try:
        for url in indxURLs:
            data = urllib.request.urlopen(url)
            subIndx = str(data.read(), "utf-8")
            subPath = url.split("/")[:6]
            subPath = "/".join(subPath) + "/"
            subIndx = subIndx.replace("DATA/", subPath + "DATA/").lower()
            index += subIndx
    except Exception as e:
        print(e, file=sys.stderr)
        print("Error downloading index file %s" % url, file=sys.stderr)

    # Create pandas dataframe
    print("Creating index dataframe...")

    cols = ["LBL_PATH", "ID", "TIME", "DS_ID", "RLS_ID", "REV_ID"]
    df = pd.read_csv(io.StringIO(index), sep=",", header=None, names=cols)

    # Download the data files
    print("Downloading data files...")

    files = [None] * len(tracks) * 3
    try:
        for track in tracks:
            track = track.lower()
            lbl = df[df["ID"] == track]["LBL_PATH"]
            if len(lbl) == 0:
                raise ValueError("No files found to match %s" % track)
            lbl = lbl.iloc[0]
            geo = lbl.replace(".lbl", "_g.dat")
            sci = lbl.replace(".lbl", "_f.dat")

            lbl_path = path + "/" + lbl.split("/")[-1]
            if clobber or not os.path.isfile(lbl_path):
                lbl_path, headers = urllib.request.urlretrieve(lbl, lbl_path)

            geo_path = path + "/" + geo.split("/")[-1]
            if clobber or not os.path.isfile(geo_path):
                geo_path, headers = urllib.request.urlretrieve(geo, geo_path)

            sci_path = path + "/" + sci.split("/")[-1]
            if clobber or not os.path.isfile(sci_path):
                sci_path, headers = urllib.request.urlretrieve(sci, sci_path)

            files.append(lbl_path)
            files.append(geo_path)
            files.append(sci_path)

    except Exception as e:
        print(e, file=sys.stderr)
        print("Error downloading track %s" % track, file=sys.stderr)

    return files
