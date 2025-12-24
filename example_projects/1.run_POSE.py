# ==============================================================================
#  Copyright (c) 2025. Longtan Wang and Weilai Pei.
#
#  This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#  International License. To view a copy of this license, visit at
#  http://creativecommons.org/licenses/by-nc-sa/4.0/
# ==============================================================================
#
#  Authors:
#  - Longtan Wang
#    Department of Earth and Space Sciences,
#    Southern University of Science and Technology, Shenzhen, China
#
#  - Weilai Pei
#    Sinopec Petroleum Exploration and Production Research Institute,
#    Beijing, China
#
#  Contact: wanglt@sustech.edu.cn
#
#  This script is part of the FMFM program.
#  It is responsible for POSE moduel driver.
# ==============================================================================

import sys, os
sys.path.append('../src')
import multiprocessing as mp
import warnings
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from tqdm import tqdm
import config
import time
import numpy as np
import logging
from single import calc_solu
warnings.filterwarnings("ignore")

# --- Set thread environment variables for numerical libraries ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def process_event_worker(task_args):
    event_info, config = task_args
    evt_id = event_info['evt_id']
    station_code = event_info['station_code']
    # Initialize a logger for the worker process.
    log_dir = os.path.dirname(config.log_file_path_POSE)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    worker_logger = logging.getLogger(f'worker_{os.getpid()}')
    worker_logger.setLevel(logging.INFO)
    # Prevent adding multiple handlers if called multiple times in the same process
    if not any(isinstance(h, logging.FileHandler) for h in worker_logger.handlers):
        file_handler = logging.FileHandler(config.log_file_path_POSE)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        worker_logger.addHandler(file_handler)
    # Remove any existing StreamHandlers to ensure no output to console from workers
    for handler in list(worker_logger.handlers):
        if isinstance(handler, logging.StreamHandler):
            worker_logger.removeHandler(handler)
    try:
        p_arrival = event_info['p_arrival']
        s_arrival = event_info['s_arrival']
        stream = config.read_seisdata(
            data_root=config.seis_data_dir, netsta=station_code, P_arr=p_arrival,
            S_arr=s_arrival, dlength=config.seis_window,
            S_factor=config.s_factor, filter=config.freq_band_POSE
        )
        if stream is None or len(stream) == 0:
            worker_logger.warning(f"SKIPPED: {evt_id}_{station_code} - Failed to read data or no data found.")
            return f"SKIPPED: {evt_id}_{station_code} - Failed to read data."
        output_filename = f"{evt_id}_{station_code}.txt"
        calc_solu().solutionset(
            output_filename, stream, config.output_root_dir,
            config.save_mat_format, config.generate_plots
        )
        worker_logger.info(f"SUCCESS: {evt_id}_{station_code}")
        return f"SUCCESS: {evt_id}_{station_code}"
    except Exception:
        # Capture the full traceback and log it
        tb_str = traceback.format_exc()
        try:
            evt_id_err = task_args[0]['evt_id']
            st_code_err = task_args[0]['station_code']
            error_message = f"FAILED: {evt_id_err}_{st_code_err}\n--- TRACEBACK ---\n{tb_str}--- END ---"
            worker_logger.error(error_message)
            return f"FAILED: {evt_id_err}_{st_code_err} - An error occurred. Check log for details."
        except Exception:
            error_message = f"FAILED: Worker process crashed with args {task_args}\n--- TRACEBACK ---\n{tb_str}--- END ---"
            worker_logger.error(error_message)
            return f"FAILED: Worker process crashed. Check log for details."


class POSE_Processor:
    def __init__(self, config):
        self.config = config
        self.seismic_events = []
        self._setup_logging() # Setup logging when the processor is initialized
    
    def _setup_logging(self):
        """Sets up the main logger for the application."""
        log_dir = os.path.dirname(self.config.log_file_path_POSE)
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists
        # Configure the root logger
        logging.basicConfig(
            level=logging.INFO, # Set the logging level (INFO, WARNING, ERROR, DEBUG, CRITICAL)
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file_path_POSE)
                # Removed StreamHandler to stop printing logs to console
            ]
        )
        self.logger = logging.getLogger('POSE_Processor') # Get a specific logger for this class

    def _prepare(self):
        """Preparation phase: parse catalog, create output directories."""
        self.logger.info("--- Step 1: Preparing ---")
        self.phases = self.config.read_fpha(config.catalog_path)
        os.makedirs(self.config.raw_result_dir, exist_ok=True)
        self.logger.info(f"Found {len(self.seismic_events)} total events in '{self.config.read_fpha}'.")
        self.logger.info(f"Results will be saved to '{self.config.output_root_dir}'.")

    def _run_parallel_processing(self):
        task_list = []
        for phase in self.phases:
            #print(phase)
            evt_id, station_code, p_arrival, s_arrival, phase_count = phase
            # Skip events with insufficient phase counts
            if phase_count < config.min_phase_num: continue
            event_info = {
                'evt_id': evt_id, 'station_code': station_code,
                'p_arrival': p_arrival, 's_arrival': s_arrival
            }
            # Each task's argument is a tuple: (event_info_dict, config_object)
            task_list.append((event_info, config))

        if not task_list:
            self.logger.warning("No valid tasks found that meet the criteria.")
            return
        self.logger.info(f"Filtered {len(task_list)} valid tasks to be processed in parallel on {self.config.max_concurrent_processes} cores...")
    
        with mp.Pool(processes=config.max_concurrent_processes) as pool:
            results_iterator = pool.imap_unordered(process_event_worker, task_list)
            for result in tqdm(results_iterator, total=len(task_list), desc="Processing Events"):
                if "FAILED" in result:
                    self.logger.error(f"Worker reported a FAILED task: {result}")
                elif "SKIPPED" in result:
                    self.logger.warning(f"Worker reported a SKIPPED task: {result}")
                else:
                    self.logger.info(f"Worker reported SUCCESS: {result}")

    def _post_process(self):
        """Post-processing phase: aggregate all individual result files."""
        self.logger.info("\n--- Step 3: Aggregating Final Results ---")
        try:
            self.config.merge_polarity_data(
                raw_resdir=self.config.raw_result_dir,
                output_path=self.config.pols_merge_path,
                seis_window=self.config.seis_window
            )
            self.logger.info(f"Successfully aggregated results to '{self.config.pols_merge_path}'.")
        except Exception as e:
            self.logger.error(f"Error during post-processing: {e}", exc_info=True)
    
    def run(self):
        """Executes the complete workflow."""
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Automatic P-Wave Polarity Determination Started")
        self._prepare()
        self._run_parallel_processing()
        self._post_process()
        end_time = time.time()
        self.logger.info(f"All processes completed! Total elapsed time: {end_time - start_time:.2f} seconds.")
        self.logger.info("=" * 60)

if __name__ == "__main__":
    # Set the multiprocessing start method (fork is faster but spawn is safer on Unix-like systems)
    config = config.Config()
    processor = POSE_Processor(config)
    processor.run()