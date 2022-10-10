# marsis
Open-source processor for the Mars Advanced Radar for Subsurface and Ionosphere Sounding (MARSIS)

This software implements several processing routines that, when performed in sequence, transform a MARSIS Experimental Data Record (EDR) into an interpretable data product (a "radargram"). The following processing steps are currently implemented:

1. Parsing of EDR label files
2. Parsing of EDR science and auxiliary data files
3. Decompression of science data
4. Several modes of ionosphere compensation
    - No ionosphere compensation
    - Ionosphere compensation via:
        - "Campbell method" (https://doi.org/10.1002/2015JE004917)
        - "McMichael method" (https://doi.org/10.1109/RADAR.2017.7944326)
<>        - "Italian method" (https://doi.org/10.1109/RADAR.2008.4720760)
<>    - Dual band range compression with ionosphere compensation that is a hybrid of the Smithsonian and Italian methods
5. Range compression

The processor can be used as a command line tool or within a Python script to generate MARSIS radargrams and/or output metdata from the EDR files in a text format. There are several Jupyter Notebooks in the "notebooks" directory that walk through usage within Python and as a command line tool 

The Campbell method requires a global DEM of Mars, I recommend using this one:
https://mchristo.net/data/megr_32ppd.tif   (253 MB)
## Useful links
MARSIS Planetary Data System (PDS) Page - https://pds-geosciences.wustl.edu/missions/mars_express/marsis.htm

## Useful information
MARSIS sampling frequency - 2.8 MHz  
Effective sampling frequency in data files after I/Q conversion - 1.4 MHz  
Chirp center frequencies - 1.8, 3, 4, and 5 MHz  
Chirp bandwidth - 1 MHz  
Chirp duration - 250 μs  
Chirp center frequency in EDR files - 0.7 MHz
