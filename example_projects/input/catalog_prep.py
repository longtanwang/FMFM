from obspy import UTCDateTime
""" 
This script prepares the input catalog file for HASH program.
add #eventid at the beginning of each event line in the catalog file.
"""
with open('palm_catalog.pha') as f:
    ctlgs = f.readlines()

evt_id = 0
with open('example_catalog.pha','w') as f:
    for ctlg in ctlgs:
        if len(ctlg) < 70:
            evt_id += 1
            ctlg = '#%08d,'%(evt_id) + ctlg
            f.write(ctlg)
        else: 
            f.write(ctlg)
