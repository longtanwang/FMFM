import os, glob, subprocess
import numpy as np
from obspy import UTCDateTime, Stream, read
import plotresult
from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass(frozen=True)
class StationPhase:
    """Stores phase arrival information for a single station."""
    net: str
    sta: str
    p_arrival: UTCDateTime
    s_arrival: UTCDateTime

@dataclass
class Event:
    """Stores a single seismic event and all its associated station phases."""
    event_id: int
    phases: List[StationPhase]

def datestr(date):
    return '%4s%02d%02d'%(date.year,date.month,date.day)

def read_fpha(file_path):
    result = []; current_group = []; evt_id = None  # Initialize event ID variable
    with open(file_path) as f:
        for line in (ln.strip() for ln in f if ln.strip()):
            if line[0] == '#':  # Event header line
                if current_group:
                    phase_count = len(current_group)
                    for record in current_group: record[4] = phase_count
                    result.extend(current_group)
                    current_group = []
                evt_id = line.split(',', 1)[0][1:]  # Extract the first field
            else:  # Phase data line
                if evt_id is None: continue
                parts = line.split(',', 4)
                if len(parts) >= 3 and parts[1] != '-1':
                    current_group.append([evt_id,parts[0],parts[1],parts[2],None])
    # the last event in the file
    if current_group:
        phase_count = len(current_group)
        for record in current_group:
            record[4] = phase_count
        result.extend(current_group)
    return result

def find_seismic_files(data_root, netsta, datetime):
    UTCdt = UTCDateTime(datetime)
    search_path = os.path.join\
            (data_root,\
            datestr(UTCdt), \
            f"{netsta}.{datestr(UTCdt)}.*Z.mseed")
    found_files = glob.glob(search_path)
    if not found_files: return None     
    return sorted(found_files)[0]

def read_seisdata(data_root, netsta, P_arr, S_arr, dlength, S_factor, filter = None):
    dpath = find_seismic_files(data_root, netsta, P_arr)
    samp_rate = 100
    win_npts = dlength * samp_rate
    start_time = UTCDateTime(P_arr) - dlength
    end_time = UTCDateTime(P_arr) + dlength
    st = read(dpath, starttime = start_time, endtime = end_time)
    try: st = read(dpath, starttime = start_time, endtime = end_time)
    except:
        print('Error in reading data! \n')
        return 0
    if filter:
        try: 
            st.detrend('constant').detrend('demean')
            st.taper(max_percentage = 0.1, max_length= 0.2)
            st = st.filter('bandpass', freqmin=filter[0], freqmax=filter[1], zerophase=False)
        except: print('Error in filting data!  \n')
    #print(st)
    data = st[0].data[int(win_npts/2):int(win_npts + (win_npts/2))]
    if S_arr != '-1':
        if UTCDateTime(S_arr) - UTCDateTime(P_arr) < dlength / 2:
            filter_Sdecrease = np.ones(win_npts)
            filter_Sdecrease[win_npts / 2 + int(samp_rate*(UTCDateTime(S_arr) - UTCDateTime(P_arr))):] = S_factor
            data = data * filter_Sdecrease
    return data

def output_resfile(a,b,c,d,c_num,name,outputpath,iswrite1 = 0,iswrite2 = 0):
    """
    Outputs the POSE results to text and optional .mat files and plots.
    """
    rawdata_outputpath = os.path.join(outputpath,'RawResult/')
    figure_outputpath  = os.path.join(outputpath,'Figure/')
    if iswrite1 == 1:
        scio.savemat('%s' % (rawdata_outputpath) + '%s.mat' % (a.name),{'transitionmatrix': np.array(b.matrix),'ampprob': np.array(b.ampprob_up).astype('float64'), 'Apeak': b.Apeak, 'samplelength': b.samplength,'eigvalue':b.eigvalue,'bigeig':b.bigeig,'threshold':a.threshold})
    for i in range(0,c_num):
        b.estimation(i)
        if iswrite1 == 1:
            scio.savemat('%s' % (rawdata_outputpath) + '%s_timeprob_%d.mat'%(a.name,i),{'timeprob': c[i]})
        if iswrite2 == 1:
            plotresult.plotprob(a, b, i, a.name, figure_outputpath)
        f = open('%s' % (rawdata_outputpath) + '%s'%(name), "a+")
        f.writelines('%s ' % (name) + 'solution id:%d '%(i) + 'arrivaltime:%.3f ' % (b.arrivalestimate) + 'overall up:%.5f ' % (float(np.sum(c[i] * d))) + 'up:%.3f '%(b.polarityup) + 'down:%.3f '%(b.polaritydown) + 'unknown:%.3f '%(b.polarityunknown) + '\n')
        #f.writelines('%s ' % (name) + 'solution id:%d '%(i) +'arrivaltime:%.3f ' % (b.arrivalestimate) + 'overall up:%.5f ' % (np.float(np.sum(c[i] * d))) +  'up:%.3f '%(b.polarityup)+  'down:%.3f '%(b.polaritydown)+ 'unknown:%.3f '%(b.polarityunknown)+'\n')
        f.close()


def merge_polarity_data(raw_resdir, output_path, seis_window=5):
    """
    Processes all raw POSE result text files to find the best polarity pick for each event.
    """
    raw_resfilepaths = sorted(glob.glob(os.path.join(raw_resdir, '*.txt')))
    with open(output_path, 'w') as f_out:
        for filepath in raw_resfilepaths:
            with open(filepath, 'r') as f: datas = f.readlines()
                
            best_time = 0; best_label = 'unknown'; max_prob = 0.0; evt_id = None
            for data in datas:
                datastr = data.split()
                if len(datastr) < 9: continue
                try:
                    current_time = float(datastr[3][-5:])
                    current_upprob = float(datastr[6][-5:])
                    current_doprob = float(datastr[7][-5:])
                    current_unprob = float(datastr[8][-5:])
                except (IndexError, ValueError):
                    continue
                
                time_diff = abs(current_time - seis_window/2)
                if time_diff > abs(best_time - seis_window/2): continue
                
                evt_id = datastr[0][:-4]
                best_time = current_time
                
                if current_upprob >= current_doprob:
                    if current_upprob > current_unprob:
                        best_label = 'up'
                        max_prob = current_upprob
                    else:
                        best_label = 'unknown'
                        max_prob = current_unprob
                else:
                    if current_doprob > current_unprob:
                        best_label = 'down'
                        max_prob = current_doprob
                    else:
                        best_label = 'unknown'
                        max_prob = current_unprob
            if evt_id:
                output_line = f"{evt_id},{best_time:.2f},{best_label},{max_prob:.4f}\n"
                f_out.write(output_line)
    # delete origin file
    # subprocess.run(['rm', os.path.join(raw_resdir, '*.txt')])
