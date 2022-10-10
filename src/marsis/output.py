import struct

import numpy as np
import pandas as pd
import PIL as pil
import matplotlib.pyplot as plt


def gen_tiff(rg, name):
    # Log scaling
    rg = np.log(np.power(rg, 2))
    rg -= rg.min()
    rg *= 255 / rg.max()
    rg = rg.astype(np.uint8)
    img = pil.Image.fromarray(rg)
    img.save(name)


def gen_segy(rg, edr, name):
    lat = np.copy(edr.geo["SUB_SC_LATITUDE"])
    lon = np.copy(edr.geo["SUB_SC_LONGITUDE"])

    # Convert 0-360 to -180-180
    for i in range(len(lon)):
        if lon[i] > 180:
            lon[i] = lon[i] - 360

        lon[i] = lon[i]

    # Multiply by 10000 to store as ints w/ 3 decimal digit precision in segy
    lat *= 10000
    lon *= 10000

    # Load data
    rg = np.transpose(rg).copy(order="C")  # This makes the traces C-contiguous
    spt = rg.shape[1]
    ntrace = rg.shape[0]

    # Build Segy 2.0 format files
    # https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev2_0-mar2017.pdf
    with open(name, "wb") as f:
        ### Write 3200 byte text header
        f.write(build_txthead(name.split("/")[-1].split(".")[0]))

        ### Write 400 byte binary header
        f.write(build_binhead(spt, ntrace))

        ### Write data traces
        for i in range(ntrace):
            f.write(build_trchead(spt, i + 1, int(lat[i]), int(lon[i])))
            f.write(rg[i, :].astype(">f", order="C").tobytes(order="C"))

    return 0


def build_txthead(track):
    txthead = b""

    # (0) Track number string
    track = track[0:10]
    txthead += track.encode(encoding="ASCII", errors="strict")

    # (10) Rest of text header, 3200-10 = 3190 bytes
    for i in range(398):
        txthead += struct.pack("Q", 0)

    txthead += struct.pack("I", 0)
    txthead += struct.pack("H", 0)

    return txthead


def build_binhead(spt, ntrace):
    ### Build binary header
    binhead = b""

    # (0) Job identification number
    binhead += struct.pack(">I", 1)
    # (4) Line number
    binhead += struct.pack(">I", 1)
    # (8) Reel number
    binhead += struct.pack(">I", 1)
    # (12) Traces per ensemble
    binhead += struct.pack(">H", 1)
    # (14) Auxiliary traces per ensemble
    binhead += struct.pack(">H", 0)
    # (16) Sample interval
    binhead += struct.pack(">H", 375)
    # (18) Field recording sample interval
    binhead += struct.pack(">H", 375)
    # (20) Samples per trace
    binhead += struct.pack(">H", spt)
    # (22) Field recording samples per trace
    binhead += struct.pack(">H", spt)
    # (24) Data format code
    binhead += struct.pack(">H", 5)
    # (26) Ensemble fold
    binhead += struct.pack(">H", 1)
    # (28) Trace sorting code
    binhead += struct.pack(">H", 4)
    # (30) Vertical sum code
    binhead += struct.pack(">H", 1)
    # (32) Sweep freq start
    binhead += struct.pack(">H", 0)
    # (34) Sweep freq end
    binhead += struct.pack(">H", 0)
    # (36) Sweep length
    binhead += struct.pack(">H", 0)
    # (38) Sweep type
    binhead += struct.pack(">H", 0)
    # (40) Trace number of sweep channel
    binhead += struct.pack(">H", 0)
    # (42) Sweep trace taper length at start
    binhead += struct.pack(">H", 0)
    # (44) Sweep trace taper length at end
    binhead += struct.pack(">H", 0)
    # (46) Taper type
    binhead += struct.pack(">H", 0)
    # (48) Correlated data traces, 2 is yes
    binhead += struct.pack(">H", 2)
    # (50) Binary gain recovered, 2 is no
    binhead += struct.pack(">H", 1)
    # (52) Amplitude recovery method, 1 is none
    binhead += struct.pack(">H", 4)
    # (54) Measurement system, 1 is meters
    binhead += struct.pack(">H", 1)
    # (56) Implulse signal polarity
    binhead += struct.pack(">H", 1)
    # (58) Vibratory polarity code
    binhead += struct.pack(">H", 0)
    # (60) Extended number of traces per ensemble
    binhead += struct.pack(">I", 0)
    # (64) Extended number of auxiliary traces per ensemble
    binhead += struct.pack(">I", 0)
    # (68) Extended samples per trace
    binhead += struct.pack(">I", 0)
    # (72) Extended sample interval
    binhead += struct.pack(">d", 37.5)
    # (80) Extended field recording sample interval
    binhead += struct.pack(">d", 0)
    # (88) Extended field recording samples per trace
    binhead += struct.pack(">I", 0)
    # (92) Extended ensemble fold
    binhead += struct.pack(">I", 0)
    # (96) An integer constant - see format doc
    binhead += struct.pack(">I", 16909060)
    # (100) 200 unassigned bytes
    for i in range(25):
        binhead += struct.pack("Q", 0)
    # (300) Major SEG-Y format revision number
    binhead += struct.pack(">B", 2)
    # (301) Minor SEG-Y format revision number
    binhead += struct.pack(">B", 0)
    # (302) Fixed length trace flag, 1 is yes - see format doc
    binhead += struct.pack(">H", 1)
    # (304) Number of extended text file headers
    binhead += struct.pack(">H", 0)
    # (306) Number of additional trace headers
    binhead += struct.pack(">I", 0)
    # (310) Time basis code, 4 is UTC
    binhead += struct.pack(">H", 4)
    # (312) Number of traces in file
    binhead += struct.pack(">Q", ntrace)
    # (320) Byte offset of first trace
    binhead += struct.pack(">Q", 0)
    # (328) Number of data trailer stanza records
    binhead += struct.pack(">I", 0)
    # (332) 68 unassigned bytes
    for i in range(17):
        binhead += struct.pack(">I", 0)

    return binhead


def build_trchead(spt, i, lat, lon):
    # i - trace number
    trchead = b""
    # (0) Trace sequence number within line
    trchead += struct.pack(">I", i)
    # (4) Trace sequence number within file - starts at 1
    trchead += struct.pack(">I", i)
    # (8) Original field record number
    trchead += struct.pack(">I", i)
    # (12) Trace number in original field record
    trchead += struct.pack(">I", i)
    # (16) Energy source point number
    trchead += struct.pack(">I", 0)
    # (20) Ensemble number
    trchead += struct.pack(">I", i)
    # (24) Trace number within ensemble
    trchead += struct.pack(">I", 0)
    # (28) Trace ID code, 1 is time domain
    trchead += struct.pack(">H", 1)
    # (30) Number of vertically summed traces
    trchead += struct.pack(">H", 0)
    # (32) Number of horizontally stacked traces (stacking??)
    trchead += struct.pack(">H", 0)
    # (34) Data use, 1 is production
    trchead += struct.pack(">H", 1)
    # (36) Distance from center of source point to center of rx group
    trchead += struct.pack(">I", 0)
    # (40) Elevation of rx group
    trchead += struct.pack(">I", 0)
    # (44) Surface elevation at source location
    trchead += struct.pack(">I", 0)
    # (48) Source depth below surface
    trchead += struct.pack(">I", 0)
    # (52) Seismic datum elevation at rx group
    trchead += struct.pack(">I", 0)
    # (56) Seismic datum elevation at source
    trchead += struct.pack(">I", 0)
    # (60) Water column height at source location
    trchead += struct.pack(">I", 0)
    # (64) Water column height at rx group location
    trchead += struct.pack(">I", 0)
    # (68) Scalar to be applied to last seven fields for real val - see format doc
    trchead += struct.pack(">H", 1)
    # (70) Scalar to be applied to four following fields for real val - see format doc
    trchead += struct.pack(">h", -10000)
    # (72) Source X coord
    trchead += struct.pack(">I", 0)
    # (76) Source Y coord
    trchead += struct.pack(">I", 0)
    # (80) rx group X coord
    trchead += struct.pack(">I", 0)
    # (84) rx group Y coord
    trchead += struct.pack(">I", 0)
    # (88) Coordinate units, 3 is decimal degrees
    trchead += struct.pack(">H", 3)
    # (90) Weathering velocity
    trchead += struct.pack(">H", 0)
    # (92) Subweathering velocity
    trchead += struct.pack(">H", 0)
    # (94) Uphole time at source in ms
    trchead += struct.pack(">H", 0)
    # (96) Uphole time at group in ms
    trchead += struct.pack(">H", 0)
    # (98) Source static correction in ms
    trchead += struct.pack(">H", 0)
    # (100) Group static correction in ms
    trchead += struct.pack(">H", 0)
    # (102) Total static correction applied in ms
    trchead += struct.pack(">H", 0)
    # (104) Lag time A
    trchead += struct.pack(">H", 0)
    # (106) Lag time B
    trchead += struct.pack(">H", 0)
    # (108) Delay recording time
    trchead += struct.pack(">H", 0)
    # (110) Mute time start in ms
    trchead += struct.pack(">H", 0)
    # (112) Mute time stop in ms
    trchead += struct.pack(">H", 0)
    # (114) Samples in this trace
    trchead += struct.pack(">H", spt)
    # (116) Sample interval for this trace
    trchead += struct.pack(">H", 0)
    # (118) Gain type of field instruments
    trchead += struct.pack(">H", 0)
    # (120) Instrument gain constant
    trchead += struct.pack(">H", 0)
    # (122) Instrument early or initial gain
    trchead += struct.pack(">H", 0)
    # (124) Correlated, 2 is yes
    trchead += struct.pack(">H", 2)
    # (126) Sweep freq start Hz
    trchead += struct.pack(">H", 0)
    # (128) Sweep freq end Hz
    trchead += struct.pack(">H", 0)
    # (130) Sweep length ms
    trchead += struct.pack(">H", 0)
    # (132) Sweep type, 1 is linear
    trchead += struct.pack(">H", 1)
    # (134) Sweep trace taper length start ms
    trchead += struct.pack(">H", 0)
    # (136) Sweep trace taper length end ms
    trchead += struct.pack(">H", 0)
    # (138) Taper type
    trchead += struct.pack(">H", 0)
    # (140) Alias filter freq Hz
    trchead += struct.pack(">H", 0)
    # (142) Alias filter slope
    trchead += struct.pack(">H", 0)
    # (144) Notch filter freq Hz
    trchead += struct.pack(">H", 0)
    # (146) Notch filter flope
    trchead += struct.pack(">H", 0)
    # (148) Low-cut freq Hz
    trchead += struct.pack(">H", 0)
    # (150) Hi-cut freq Hz
    trchead += struct.pack(">H", 0)
    # (152) Low-cut slope
    trchead += struct.pack(">H", 0)
    # (154) Hi-cut slope
    trchead += struct.pack(">H", 0)
    # (156) Year data recorded
    trchead += struct.pack(">H", 0)
    # (158) Day of year data recorded
    trchead += struct.pack(">H", 0)
    # (160) Hour of day
    trchead += struct.pack(">H", 0)
    # (162) Minute of hour
    trchead += struct.pack(">H", 0)
    # (164) Second of minute
    trchead += struct.pack(">H", 0)
    # (166) Time basis code
    trchead += struct.pack(">H", 0)
    # (168) Trace weighting factor
    trchead += struct.pack(">H", 0)
    # (170) Geophone group number of roll switch position one
    trchead += struct.pack(">H", 0)
    # (172) Geophone group number of trace number one within original field record
    trchead += struct.pack(">H", 0)
    # (174) Geophone group number of last trace within original field record
    trchead += struct.pack(">H", 0)
    # (176) Gap size
    trchead += struct.pack(">H", 0)
    # (178) Over travel assoc with taper at beginning of line
    trchead += struct.pack(">H", 0)
    # (180) X coordinate of ensemble position of this trace
    trchead += struct.pack(">i", lon)
    # (184)Y coordinate of ensemble position of this trace
    trchead += struct.pack(">i", lat)
    # (188) Line number (inline  number)
    trchead += struct.pack(">I", 1)
    # (192) Ensemble number (crossline number)
    trchead += struct.pack(">I", i)
    # (196) Shotpoint number
    trchead += struct.pack(">I", 0)
    # (200) Scalar to be applied to shotpoint number
    trchead += struct.pack(">H", 0)
    # (202) Trace units, 0 is unknown
    trchead += struct.pack(">H", 0)
    # (204) Transduction constant
    trchead += struct.pack(">I", 0)
    trchead += struct.pack(">H", 0)
    # (210) Transduction units
    trchead += struct.pack(">H", 0)
    # (212) Device/Trace identifier
    trchead += struct.pack(">H", 0)
    # (214) Scalar to be applied to uphole, static, and group corrections
    trchead += struct.pack(">H", 0)
    # (216) Source type/orientation
    trchead += struct.pack(">H", 0)
    # (218) Source energy direction wrt orientation
    trchead += struct.pack(">H", 0)
    trchead += struct.pack(">H", 0)
    trchead += struct.pack(">H", 0)
    # (224) Source measurement
    trchead += struct.pack(">I", 0)
    trchead += struct.pack(">H", 0)
    # (230) Source measurement unit
    trchead += struct.pack(">H", 0)
    # (232) Zero
    trchead += struct.pack(">Q", 0)

    return trchead
