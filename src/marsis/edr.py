import os
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ply import lex, yacc


class EDR:
    """Reader for experimental data record (EDR) files from the PDS.

    This object will ingest and store data from the EDR files, to be
    processed later. It only ingests data into a convenient structure,
    and de-compresses the science data, no processing occurs.
    """

    def __init__(self, lbl=None):
        """Create an EDR object.

        Create an EDR object, read in an EDR file if given.

        Parameters
        ----------
        lbl: str, optional
            Path to an EDR label file which must be in the same directory as
            the corresponding science and geometry files

        Returns
        -------
        EDR

        Notes
        -----
        """
        if lbl is None:
            return

        self.load(lbl)

    def load(self, lbl):
        """Load a set of EDR files.

        Read in the label, science, and geometry EDR files for a given
        observation.

        Parameters
        ----------
        lbl: str
            Path to an EDR label file which must be in the same directory as
            the corresponding science and geometry files

        Returns
        -------

        Notes
        -----
        """
        # Read label file
        self.lbld = self.parseLBL(lbl)

        # Science and aux file names
        sci = lbl.replace(".lbl", "_f.dat")
        geo = lbl.replace(".lbl", "_g.dat")

        # Read science file
        self.anc, self.ost, self.data = self.parseSci(sci, self.lbld)

        # Read geometry file
        self.geo = self.parseGeo(geo, self.lbld)

    def parseGeo(self, file, lbld):
        """Read in a geometry data file.

        Parameters
        ----------
        file: str
            Path to an geometry data file
        lbld: dict
            Dictionary containing data parsed from corresponding label
            file, created by the parseLBL method

        Returns
        -------

        Notes
        -----
        """
        # Set up geometry data frame

        rec_t = np.dtype(
            [
                ("SCET_FRAME_WHOLE", ">u4"),
                ("SCET_FRAME_FRAC", ">u2"),
                ("GEOMETRY_EPHEMERIS_TIME", ">f8"),
                ("GEOMETRY_EPOCH", "V23"),
                ("MARS_SOLAR_LONGITUDE", ">f8"),
                ("MARS_SUN_DISTANCE", ">f8"),
                ("ORBIT_NUMBER", ">u4"),
                ("TARGET_NAME", "V6"),
                ("TARGET_SC_POSITION_VECTOR", ">f8", 3),
                ("SPACECRAFT_ALTITUDE", ">f8"),
                ("SUB_SC_LONGITUDE", ">f8"),
                ("SUB_SC_LATITUDE", ">f8"),
                ("TARGET_SC_VELOCITY_VECTOR", ">f8", 3),
                ("TARGET_SC_RADIAL_VELOCITY", ">f8"),
                ("TARGET_SC_TANG_VELOCITY", ">f8"),
                ("LOCAL_TRUE_SOLAR_TIME", ">f8"),
                ("SOLAR_ZENITH_ANGLE", ">f8"),
                ("DIPOLE_UNIT_VECTOR", ">f8", 3),
                ("MONOPOLE_UNIT_VECTOR", ">f8", 3),
            ]
        )

        geoData = np.fromfile(file, dtype=rec_t)

        return geoData

    def decompress(self, trace, exp):
        sign = (-1) ** (trace >> 7)  # Sign bit
        mantissa = 1 + ((trace & 0x7F) / (2.0 ** 7))
        # mantissa = trace & 0x7F
        trace = sign * mantissa * (2 ** (exp - 127))
        return trace

    def deagc(self, data, agc):
        # Take in an array of marsis data
        # and a vector of AGC settings,
        # then correct for agc
        agc = agc & 0x07  # Only last three bits matter
        agc = agc * 4 + 2  # Gain in dB, per Orosei
        data = data * 10 ** (agc / 20)[np.newaxis, :]
        return data

    def parseSci(self, file, lbld):
        # Set up ancillary data dataframe

        rec_t = np.dtype(
            [
                ("SCET_STAR_WHOLE", ">u4"),
                ("SCET_STAR_FRAC", ">u2"),
                ("OST_LINE_NUMBER", ">u2"),
                ("OST_LINE", "V12"),
                ("FRAME_NUMBER", ">u2"),
                ("ANCILLARY_DATA_HEADER", "V6"),
                ("FIRST_PRI_OF_FRAME", ">u4"),
                ("SCET_FRAME_WHOLE", ">u4"),
                ("SCET_FRAME_FRAC", ">u2"),
                ("SCET_PERICENTER_WHOLE", ">u4"),
                ("SCET_PERICENTER_FRAC", ">u2"),
                ("SCET_PAR_WHOLE", ">u4"),
                ("SCET_PAR_FRAC", ">u2"),
                ("H_SCET_PAR", ">f4"),
                ("VT_SCET_PAR", ">f4"),
                ("VR_SCET_PAR", ">f4"),
                ("N_0", ">u4"),
                ("DELTA_S_MIN", ">f4"),
                ("NB_MIN", ">u2"),
                ("M_OCOG", ">f4", 2),
                ("INDEX_OCOG", ">u2", 2),
                ("TRK_THRESHOLD", ">f4", 2),
                ("INI_IND_TRK_THRESHOLD", ">u2", 2),
                ("LAST_IND_TRK_THRESHOLD", ">u2", 2),
                ("INI_IND_FSRM", ">u2", 2),
                ("LAST_IND_FSRM", ">u2", 2),
                ("SPARE_4", ">u4", 3),
                ("DELTA_S_SCET_PAR", ">f4"),
                ("NB_SCET_PAR", ">u2"),
                ("NA_SCET_PAR", ">u2", 2),
                ("A2_INI_CM", ">f4", 2),
                ("A2_OPT", ">f4", 2),
                ("REF_CA_OPT", ">f4", 2),
                ("DELTA_T", ">u2", 2),
                ("SF", ">f4", 2),
                ("I_C", ">u2", 2),
                ("AGC_SA_FOR_NEXT_FRAME", ">f4", 2),
                ("AGC_SA_LEVELS_CURRENT_FRAME", ">u1", 2),
                ("RX_TRIG_SA_FOR_NEXT_FRAME", ">u2", 2),
                ("RX_TRIG_SA_PROGR", ">u2", 2),
                ("INI_IND_OCOG", ">u2"),
                ("LAST_IND_OCOG", ">u2"),
                ("OCOG", ">f4", 2),
                ("A", ">f4", 2),
                ("C_LOL", ">i2", 2),
                ("SPARE_5", ">u2", 3),
                ("MAX_RE_EXP_MINUS1_F1_DIP", ">u1"),
                ("MAX_IM_EXP_MINUS1_F1_DIP", ">u1"),
                ("MAX_RE_EXP_ZERO_F1_DIP", ">u1"),
                ("MAX_IM_EXP_ZERO_F1_DIP", ">u1"),
                ("MAX_RE_EXP_PLUS1_F1_DIP", ">u1"),
                ("MAX_IM_EXP_PLUS1_F1_DIP", ">u1"),
                ("MAX_RE_EXP_MINUS1_F2_DIP", ">u1"),
                ("MAX_IM_EXP_MINUS1_F2_DIP", ">u1"),
                ("MAX_RE_EXP_ZERO_F2_DIP", ">u1"),
                ("MAX_IM_EXP_ZERO_F2_DIP", ">u1"),
                ("MAX_RE_EXP_PLUS1_F2_DIP", ">u1"),
                ("MAX_IM_EXP_PLUS1_F2_DIP", ">u1"),
                ("SPARE_6", ">u1", 8),
                ("AGC_PIS_PT_VALUE", ">f4", 2),
                ("AGC_PIS_LEVELS", ">u1", 2),
                ("K_PIM", ">u1"),
                ("PIS_MAX_DATA_EXP", ">u1", 2),
                ("PROCESSING_PRF", ">f4"),
                ("SPARE_7", ">u1"),
                ("REAL_ECHO_MINUS1_F1_DIP", ">u1", 512),
                ("IMAG_ECHO_MINUS1_F1_DIP", ">u1", 512),
                ("REAL_ECHO_ZERO_F1_DIP", ">u1", 512),
                ("IMAG_ECHO_ZERO_F1_DIP", ">u1", 512),
                ("REAL_ECHO_PLUS1_F1_DIP", ">u1", 512),
                ("IMAG_ECHO_PLUS1_F1_DIP", ">u1", 512),
                ("REAL_ECHO_MINUS1_F2_DIP", ">u1", 512),
                ("IMAG_ECHO_MINUS1_F2_DIP", ">u1", 512),
                ("REAL_ECHO_ZERO_F2_DIP", ">u1", 512),
                ("IMAG_ECHO_ZERO_F2_DIP", ">u1", 512),
                ("REAL_ECHO_PLUS1_F2_DIP", ">u1", 512),
                ("IMAG_ECHO_PLUS1_F2_DIP", ">u1", 512),
                ("PIS_F1", ">i2", 128),
                ("PIS_F2", ">i2", 128),
            ]
        )

        telTab = np.fromfile(file, dtype=rec_t)

        # Decode OST line bit fields - this is incomplete
        df = pd.DataFrame()
        ost = telTab["OST_LINE"]
        ost = np.array(ost.tolist())  # weird but it works to get from void to bytes_
        df["SPARE_0"] = np.vectorize(lambda s: s[0])(ost)
        df["MODE_DURATION"] = np.vectorize(
            lambda s: np.frombuffer(s[0:4], dtype=">u4") & 0x00FFFFFF
        )(ost)
        df["SPARE_1"] = np.vectorize(
            lambda s: np.frombuffer(s[4:5], dtype=">u1") & 0xC0
        )(ost)
        df["MODE_SELECTION"] = np.vectorize(
            lambda s: np.frombuffer(s[4:5], dtype=">u1") >> 2 & 0x0F
        )(ost)
        df["DCG_CONFIGURATION_F1"] = np.vectorize(
            lambda s: np.frombuffer(s[4:5], dtype=">u1") & 0x03
        )(ost)
        df["DCG_CONFIGURATION_F2"] = np.vectorize(
            lambda s: np.frombuffer(s[5:6], dtype=">u1") >> 6
        )(ost)

        # Decompress data and make radargrams
        moded = {
            "SS3_TRK_CMP": [
                "MINUS1_F1",
                "ZERO_F1",
                "PLUS1_F1",
                "MINUS1_F2",
                "ZERO_F2",
                "PLUS1_F2",
            ]
        }

        mode = lbld["INSTRUMENT_MODE_ID"].replace('"', "")

        if mode not in moded.keys():
            print("Unhandled mode, exiting")
            print(mode)
            sys.exit()

        datad = {}

        for rg in moded[mode]:
            block = np.zeros((512, len(telTab)), dtype=np.complex64)
            for i in range(len(telTab)):
                expIM = telTab["MAX_IM_EXP_" + rg + "_DIP"][i]
                expRE = telTab["MAX_RE_EXP_" + rg + "_DIP"][i]
                trIM = telTab["IMAG_ECHO_" + rg + "_DIP"][i]
                trRE = telTab["REAL_ECHO_" + rg + "_DIP"][i]

                trRE = self.decompress(trRE, expRE)
                trIM = self.decompress(trIM, expIM)

                trace = trRE + 1j * trIM

                band = int(rg.split("_")[1][1])

                block[:, i] = trace

            if "F1" in rg:
                block = self.deagc(block, telTab["AGC_SA_LEVELS_CURRENT_FRAME"][:, 0])
            elif "F2" in rg:
                block = self.deagc(block, telTab["AGC_SA_LEVELS_CURRENT_FRAME"][:, 1])

            datad[rg] = block

        return telTab, df, datad

    def buildDict(self, pdata, i):
        dd = {}

        while i < len(pdata):
            key, val = pdata[i]
            if key == "OBJECT":
                c = 0
                name = val + str(c)
                while name in dd.keys():
                    c += 1
                    name = val + str(c)

                i, dd[name] = self.buildDict(pdata, i + 1)
                continue

            if key == "END_OBJECT":
                return i + 1, dd

            dd[key] = val
            i += 1

        return dd

    def parseLBL(self, lbl):
        # Parse the label file with lex and yacc
        # Heavily based on https://github.com/mkelley/pds3

        # lexer def ###

        tokens = [
            "DSID",
            "WORD",
            "STRING",
            "COMMENT",
            "POINTER",
            "DATE",
            "INT",
            "REAL",
            "UNIT",
            "END",
        ]

        literals = ["(", ")", ",", "=", "{", "}"]

        def t_DSID(t):
            r"MEX-M-MARSIS-2-EDR(-EXT[0-9])?-V[0-9].0"  # Dataset ID
            return t

        def t_WORD(t):
            r"[A-Z][A-Z0-9:_]+"
            if t.value == "END":
                t.type = "END"
            return t

        t_STRING = r'"[^"]+"'

        def t_COMMENT(t):
            r"/\*.+\*/"
            pass

        t_POINTER = r"\^[A-Z0-9_]+"
        t_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(.[0-9]{3})?"
        t_INT = r"[+-]?[0-9]+"
        t_REAL = r"[+-]?[0-9]+\.[0-9]+([Ee][+-]?[0-9]+)?"
        t_UNIT = r"<[\w*^\-/]+>"
        t_ignore = " \t\r\n"

        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)

        lexer = lex.lex()

        # ## parser def ## #

        def p_label(p):
            """label : record
            | label record
            | label END"""
            if len(p) == 2:
                # record
                p[0] = [p[1]]
            elif p[2] == "END":
                # label END
                p[0] = p[1]
            else:
                # label record
                p[0] = p[1] + [p[2]]

        def p_record(p):
            """record : WORD '=' value
            | POINTER '=' INT
            | POINTER '=' STRING
            | POINTER '=' '(' STRING ',' INT ')'"""
            p[0] = (p[1], p[3])

        def p_value(p):
            """value : STRING
            | DATE
            | WORD
            | DSID
            | number
            | number UNIT
            | sequence"""
            # Just chuck the units for now
            p[0] = p[1]

        def p_number(p):
            """number : INT
            | REAL"""
            p[0] = p[1]

        def p_sequence(p):
            """sequence : '(' value ')'
            | '(' sequence_values ')'
            | '{' value '}'
            | '{' sequence_values '}'"""
            p[0] = p[2]

        def p_sequence_values(p):
            """sequence_values : value ','
            | sequence_values value ','
            | sequence_values value"""
            if p[2] == ",":
                p[0] = [p[1]]
            else:
                p[0] = p[1] + [p[2]]

        def p_error(p):
            if p:
                print("Syntax error at '%s'" % p.value)
            else:
                print("Syntax error at EOF")

        parser = yacc.yacc()

        # ## parse the label ## #

        fd = open(lbl, "r")
        data = fd.read()
        fd.close()

        result = parser.parse(data, lexer=lexer, debug=False)

        return self.buildDict(result, 0)
