import re, glob, os, sys
import numpy as np
from obspy import UTCDateTime
import subprocess

raw_resfile = './RawResult'
raw_resfilepaths = sorted(glob.glob(os.path.join(raw_resfile,'*.txt')))
res_outpath = 'example_polarity.dat'

f_out = open(res_outpath,'w')

data_len_for_pick = 5
for filepath in raw_resfilepaths:
    f = open(filepath,'r')
    datas = f.readlines()
    time = 0
    for data in datas:
        datastr =       data.split(' ')
        print(datastr)
        if abs(float(datastr[3][-5:]) - data_len_for_pick/2) > \
           abs(time - data_len_for_pick/2):
            continue
        evt_id =        datastr[0][:-4]
        time =          float(datastr[3][-5:])
        upprob =        float(datastr[6][-5:])
        doprob =        float(datastr[7][-5:])
        unprob =        float(datastr[8][-5:])

        #First_arrival_time = dt + time - data_len_for_pick/2

        print(time)
        if upprob >= doprob:
            if upprob > unprob:
                label = 'up'
            if upprob <= unprob:
                label = 'unknown'
        if upprob < doprob:
            if doprob > unprob:
                label = 'down'
            if doprob <= unprob:
                label = 'unknown'
    
    outputline = '%s,%.2f,%s,%.4f\n'%(evt_id,time\
                                          ,label,max(upprob,doprob,unprob))
    print(outputline)
    f_out.write(outputline)
    f.close()
f_out.close()
 
# remove raw files
#subprocess.run(['rm', os.path.join(raw_resfile,'*.txt')])