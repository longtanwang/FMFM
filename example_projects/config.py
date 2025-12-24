""" 
Configure file
"""
import sys, os
import multiprocessing as mp
sys.path.append('../src')
import utils_POSE as uP
import utils_SPRatio as uSP

class Config:
    def __init__(self):
        # Config of data_path and catalog_path
        self.seis_data_dir= './input/example_data'
        self.catalog_path= './input/example_catalog.pha'
        self.station_path= './input/example_station.sta'
        self.min_phase_num= 8

        # Config for 1.run_POSE.py
        self.output_root_dir= './output/polarity/'
        self.output_polarity_file= 'example_polarity.dat'
        self.log_file_path_POSE= './logs/example_POSE.log' 
        self.raw_result_dir = os.path.join(self.output_root_dir, 'RawResult')
        self.pols_merge_path = os.path.join(self.output_root_dir, self.output_polarity_file)
        self.seis_window= 5.0  #second
        self.s_factor= 0.5
        self.freq_band_POSE= [1, 20]
        self.max_processes = 20
        self.save_mat_format = False
        self.generate_plots = False
        self.max_concurrent_processes = min(mp.cpu_count() or 1, self.max_processes)

        # Config for 2.calc_SP_AmpRatip.py
        self.SPR_path = './output/focal_mechanisms/HASH_io/example.amp'
        self.log_file_path_SPR= './logs/example_calc_SPR.log' 
        # --- Parameters for calculating S/P ratio --- 
        self.read_window = 10.0       # time padding before P and after S
        self.offset_npts = 50         # npts shifted before the P&S arrival
        self.amp_window_npts = 200    # window length npts for P and S
        self.snr_threshold = 3.0      # P/noise SNR threshold
        self.freq_band_SPR = [1, 10]  # bandpass filter

        # Config for 3.prep_HASH.py
        self.polarity_probability_threshold = 0.95
        self.polarity_path = './output/polarity/example_polarity.dat'
        self.output_dir = './output/focal_mechanisms/HASH_io'
        self.hash_station_path = 'example.station'
        self.hash_reverse_path = 'example.reverse'
        self.hash_statcor_path = 'example.statcor'
        self.hash_phase_path = 'example.phase'
        self.is_calc_sta_corr = False
        self.H_UNCERTAINTY = 0.07
        self.V_UNCERTAINTY = 0.10

        # Config for 4.run_HASH.py
        self.hash_executable = '../bin/FMFM_hash_driver'
        self.hash_input_file = './input/example.inp'
        self.hash_output_raw_file = './output/focal_mechanisms/example_raw.out'
        self.hash_fms_csv_file = './output/focal_mechanisms/example_fms.csv'

        # 3. data interface
        self.read_fpha = uP.read_fpha
        self.merge_polarity_data = uP.merge_polarity_data
        self.read_seisdata = uP.read_seisdata
        self.read_event_catalog = uSP.read_event_catalog
        self.load_and_prepare_stream = uSP.load_and_prepare_stream
        self.calculate_amplitude_metrics = uSP.calculate_amplitude_metrics

