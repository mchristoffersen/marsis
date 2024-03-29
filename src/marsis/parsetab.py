# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = "3.10"

_lr_method = "LALR"

_lr_signature = "COMMENT DATE DSID END INT POINTER REAL STRING UNIT WORDlabel : record\n            | label record\n            | label ENDrecord : WORD '=' value\n            | POINTER '=' INT\n            | POINTER '=' STRING\n            | POINTER '=' '(' STRING ',' INT ')'value : STRING\n            | DATE\n            | WORD\n            | DSID\n            | number\n            | number UNIT\n            | sequencenumber : INT\n            | REALsequence : '(' value ')'\n            | '(' sequence_values ')'\n            | '{' value '}'\n            | '{' sequence_values '}'sequence_values : value ','\n            | sequence_values value ','\n            | sequence_values value"

_lr_action_items = {
    "WORD": (
        [
            0,
            1,
            2,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            38,
        ],
        [
            3,
            3,
            -1,
            -2,
            -3,
            9,
            -10,
            -4,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            9,
            9,
            -5,
            -6,
            -13,
            9,
            9,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
            -7,
        ],
    ),
    "POINTER": (
        [
            0,
            1,
            2,
            5,
            6,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            20,
            21,
            23,
            29,
            31,
            33,
            34,
            38,
        ],
        [
            4,
            4,
            -1,
            -2,
            -3,
            -10,
            -4,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            -5,
            -6,
            -13,
            -17,
            -18,
            -19,
            -20,
            -7,
        ],
    ),
    "$end": (
        [
            1,
            2,
            5,
            6,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            20,
            21,
            23,
            29,
            31,
            33,
            34,
            38,
        ],
        [
            0,
            -1,
            -2,
            -3,
            -10,
            -4,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            -5,
            -6,
            -13,
            -17,
            -18,
            -19,
            -20,
            -7,
        ],
    ),
    "END": (
        [
            1,
            2,
            5,
            6,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            20,
            21,
            23,
            29,
            31,
            33,
            34,
            38,
        ],
        [
            6,
            -1,
            -2,
            -3,
            -10,
            -4,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            -5,
            -6,
            -13,
            -17,
            -18,
            -19,
            -20,
            -7,
        ],
    ),
    "=": (
        [
            3,
            4,
        ],
        [
            7,
            8,
        ],
    ),
    "STRING": (
        [
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            22,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
        ],
        [
            11,
            21,
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            11,
            11,
            28,
            -13,
            11,
            11,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
        ],
    ),
    "DATE": (
        [
            7,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
        ],
        [
            12,
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            12,
            12,
            -13,
            12,
            12,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
        ],
    ),
    "DSID": (
        [
            7,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
        ],
        [
            13,
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            13,
            13,
            -13,
            13,
            13,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
        ],
    ),
    "INT": (
        [
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
        ],
        [
            16,
            20,
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            16,
            16,
            -13,
            16,
            16,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            37,
            -22,
        ],
    ),
    "REAL": (
        [
            7,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
        ],
        [
            17,
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            17,
            17,
            -13,
            17,
            17,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
        ],
    ),
    "(": (
        [
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
        ],
        [
            18,
            22,
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            18,
            18,
            -13,
            18,
            18,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
        ],
    ),
    "{": (
        [
            7,
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
        ],
        [
            19,
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            19,
            19,
            -13,
            19,
            19,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
        ],
    ),
    ")": (
        [
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            23,
            24,
            25,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            37,
        ],
        [
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            -13,
            29,
            31,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
            38,
        ],
    ),
    ",": (
        [
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            23,
            24,
            26,
            28,
            29,
            31,
            32,
            33,
            34,
        ],
        [
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            -13,
            30,
            30,
            35,
            -17,
            -18,
            36,
            -19,
            -20,
        ],
    ),
    "}": (
        [
            9,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            23,
            26,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
        ],
        [
            -10,
            -8,
            -9,
            -11,
            -12,
            -14,
            -15,
            -16,
            -13,
            33,
            34,
            -17,
            -21,
            -18,
            -23,
            -19,
            -20,
            -22,
        ],
    ),
    "UNIT": (
        [
            14,
            16,
            17,
        ],
        [
            23,
            -15,
            -16,
        ],
    ),
}

_lr_action = {}
for _k, _v in _lr_action_items.items():
    for _x, _y in zip(_v[0], _v[1]):
        if not _x in _lr_action:
            _lr_action[_x] = {}
        _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {
    "label": (
        [
            0,
        ],
        [
            1,
        ],
    ),
    "record": (
        [
            0,
            1,
        ],
        [
            2,
            5,
        ],
    ),
    "value": (
        [
            7,
            18,
            19,
            25,
            27,
        ],
        [
            10,
            24,
            26,
            32,
            32,
        ],
    ),
    "number": (
        [
            7,
            18,
            19,
            25,
            27,
        ],
        [
            14,
            14,
            14,
            14,
            14,
        ],
    ),
    "sequence": (
        [
            7,
            18,
            19,
            25,
            27,
        ],
        [
            15,
            15,
            15,
            15,
            15,
        ],
    ),
    "sequence_values": (
        [
            18,
            19,
        ],
        [
            25,
            27,
        ],
    ),
}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
    for _x, _y in zip(_v[0], _v[1]):
        if not _x in _lr_goto:
            _lr_goto[_x] = {}
        _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
    ("S' -> label", "S'", 1, None, None, None),
    ("label -> record", "label", 1, "p_label", "edr.py", 369),
    ("label -> label record", "label", 2, "p_label", "edr.py", 370),
    ("label -> label END", "label", 2, "p_label", "edr.py", 371),
    ("record -> WORD = value", "record", 3, "p_record", "edr.py", 383),
    ("record -> POINTER = INT", "record", 3, "p_record", "edr.py", 384),
    ("record -> POINTER = STRING", "record", 3, "p_record", "edr.py", 385),
    ("record -> POINTER = ( STRING , INT )", "record", 7, "p_record", "edr.py", 386),
    ("value -> STRING", "value", 1, "p_value", "edr.py", 390),
    ("value -> DATE", "value", 1, "p_value", "edr.py", 391),
    ("value -> WORD", "value", 1, "p_value", "edr.py", 392),
    ("value -> DSID", "value", 1, "p_value", "edr.py", 393),
    ("value -> number", "value", 1, "p_value", "edr.py", 394),
    ("value -> number UNIT", "value", 2, "p_value", "edr.py", 395),
    ("value -> sequence", "value", 1, "p_value", "edr.py", 396),
    ("number -> INT", "number", 1, "p_number", "edr.py", 401),
    ("number -> REAL", "number", 1, "p_number", "edr.py", 402),
    ("sequence -> ( value )", "sequence", 3, "p_sequence", "edr.py", 406),
    ("sequence -> ( sequence_values )", "sequence", 3, "p_sequence", "edr.py", 407),
    ("sequence -> { value }", "sequence", 3, "p_sequence", "edr.py", 408),
    ("sequence -> { sequence_values }", "sequence", 3, "p_sequence", "edr.py", 409),
    (
        "sequence_values -> value ,",
        "sequence_values",
        2,
        "p_sequence_values",
        "edr.py",
        413,
    ),
    (
        "sequence_values -> sequence_values value ,",
        "sequence_values",
        3,
        "p_sequence_values",
        "edr.py",
        414,
    ),
    (
        "sequence_values -> sequence_values value",
        "sequence_values",
        2,
        "p_sequence_values",
        "edr.py",
        415,
    ),
]
