import os, glob, subprocess
import numpy as np
from obspy import UTCDateTime
from obspy.core import read
import plotresult

def datestr(date):
    """
    Formats an ObsPy UTCDateTime object into a YYYYMMDD string.

    Args:
        date (UTCDateTime): The datetime object to format.

    Returns:
        str: The formatted date string (e.g., "20230101").
    """
    return '%4s%02d%02d'%(date.year,date.month,date.day)

def parse_seismic_data(file_path):
    """
    Parses a PALM-like seismic .pha file, grouping phase arrivals by event.

    It reads through the file, identifies events marked by a '#' header,
    collects all subsequent phase lines for that event, and then flattens
    the result into a single list where each record is augmented with the
    total phase count for its parent event.

    Args:
        file_path (str): The path to the input catalog file.

    Returns:
        list: A list of lists, where each inner list represents a phase record:
              [event_id, station_code, p_arrival, s_arrival, total_phase_count].
    """
    result = []
    current_group = []
    evt_id = None  # Initialize event ID variable
    with open(file_path) as f:
        for line in (ln.strip() for ln in f if ln.strip()):
            if line[0] == '#':  # Event header line
                # When a new event is found, save the phases of the previous event
                if current_group:
                    phase_count = len(current_group)
                    for record in current_group: record[4] = phase_count
                    result.extend(current_group)
                    current_group = []
                # Always update the event ID regardless of previous data
                evt_id = line.split(',', 1)[0][1:]  # Extract the first field
            else:  # Phase data line
                if evt_id is None: continue
                parts = line.split(',', 4)
                if len(parts) > 3 and parts[1] != '-1':
                    current_group.append([evt_id,parts[0],parts[1],parts[2],None])
    # Process the last event in the file
    if current_group:
        phase_count = len(current_group)
        for record in current_group:
            record[4] = phase_count
        result.extend(current_group)
    return result

def find_sac_files(data_root, netsta, datetime):
    """
    Constructs a file path pattern and finds the first matching SAC file.

    The path is constructed based on the data root, year, date, and station code
    to locate the relevant waveform data.

    Args:
        data_root (str): The root directory where waveform data is stored.
        netsta (str): The network and station code (e.g., "MS.D024").
        datetime_str (str): The datetime string used to determine year and julian day.

    Returns:
        str: The full path to the first SAC file found.
    """
    UTCdt = UTCDateTime(datetime)
    search_path = os.path.join\
            (data_root,\
            # str(UTCdt.year),\
            datestr(UTCdt), \
            f"{netsta}.{datestr(UTCdt)}.*Z.mseed")
    found_files = glob.glob(search_path)
    
    if not found_files:
        print(f"WARNING: No data file found for pattern: {search_path}")
        return None
    return sorted(found_files)[0]

def read_sac(data_root, netsta, P_arr, S_arr, dlength, S_factor, filter = None):
    """
    Reads a SAC file for a specified time window around a P-wave arrival,
    applies optional filtering, and performs a custom S-wave amplitude reduction.

    Args:
        data_root (str): Root directory of SAC files.
        netsta (str): Network and station code.
        P_arr (str): P-wave arrival time string.
        S_arr (str): S-wave arrival time string.
        dlength (float): Length of the data segment to read in seconds.
        S_factor (float): A factor to reduce the amplitude of the S-wave signal.
        filter (list, optional): A list [min_freq, max_freq] for a bandpass filter.

    Returns:
        numpy.ndarray: The processed waveform data, or 0 on failure.
    """
    # dlength : second
    dpath = find_sac_files(data_root, netsta, P_arr)
    start_time = UTCDateTime(P_arr) - dlength / 2
    end_time = UTCDateTime(P_arr) + dlength / 2
    st = read(dpath, starttime = start_time, endtime = end_time)
    try:
        st = read(dpath, starttime = start_time, endtime = end_time)
    except:
        print('Error in reading data! \n')
        return 0
    if filter:
        try:
            st = st.detrend('constant').detrend('demean')\
                .filter('bandpass', freqmin=filter[0], freqmax=filter[1], zerophase=False)
        except:
            print('Error in filting data!  \n')
    data = st[0].data[0:500]
    if UTCDateTime(S_arr) - UTCDateTime(P_arr) < dlength / 2:
        filter_Sdecrease = np.ones(500)
        filter_Sdecrease[250 + int(100*(UTCDateTime(S_arr) - UTCDateTime(P_arr))):] = S_factor
        data = data * filter_Sdecrease
    return data

def output_resfile(a,b,c,d,c_num,name,outputpath,iswrite1 = 0,iswrite2 = 0):
    """
    Outputs the analysis results to text and optional .mat files and plots.

    This function iterates through all calculated solutions for an event,
    saves raw data if requested, generates plots if requested, and appends
    a summary of each solution to a text file.

    Args:
        a: Waveform data object.
        b: Analysis state object.
        c: Markov matrix or related result.
        d: Amplitude probability result.
        c_num: Number of solutions.
        name (str): The name of the event/station.
        outputpath (str): The root directory for output.
        iswrite1 (int, optional): Flag to write .mat files (1=yes, 0=no).
        iswrite2 (int, optional): Flag to generate plots (1=yes, 0=no).
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
        f = open('%s' % (rawdata_outputpath) + '%s.txt'%(name), "a+")
        f.writelines('%s ' % (name) + 'solution id:%d '%(i) + 'arrivaltime:%.3f ' % (b.arrivalestimate) + 'overall up:%.5f ' % (float(np.sum(c[i] * d))) + 'up:%.3f '%(b.polarityup) + 'down:%.3f '%(b.polaritydown) + 'unknown:%.3f '%(b.polarityunknown) + '\n')
        #f.writelines('%s ' % (name) + 'solution id:%d '%(i) +'arrivaltime:%.3f ' % (b.arrivalestimate) + 'overall up:%.5f ' % (np.float(np.sum(c[i] * d))) +  'up:%.3f '%(b.polarityup)+  'down:%.3f '%(b.polaritydown)+ 'unknown:%.3f '%(b.polarityunknown)+'\n')
        f.close()


def process_polarity_data(raw_resdir, output_path, segment_length=5):
    """
    Processes all raw result text files to find the best polarity pick for each event.

    It reads through all intermediate .txt files in the raw result directory,
    finds the solution for each event that is closest to the center of the time
    window, and determines the final polarity label based on probability values.

    Args:
        raw_resdir (str): The directory containing raw result .txt files.
        output_path (str): The path for the final aggregated output file.
        segment_length (float, optional): The length of the data window in seconds.
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
                
                time_diff = abs(current_time - segment_length/2)
                if time_diff > abs(best_time - segment_length/2): continue
                
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
