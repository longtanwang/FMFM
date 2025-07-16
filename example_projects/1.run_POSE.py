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
#  This script is part of the POSE-FMS program.
#  It is responsible for POSE moduel driver.
# ==============================================================================

import sys, os
import multiprocessing as mp
import warnings
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from tqdm import tqdm
import time
import numpy as np
import logging # 导入 logging 模块

warnings.filterwarnings("ignore")
# --- Set thread environment variables for numerical libraries ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Add project source directory to the system path ---
sys.path.append('../src')

# --- Import data processing modules ---
from data_pipeline import parse_seismic_data, read_sac, process_polarity_data
from single import calc_solu

# ==============================================================================
# Configuration Module
# ==============================================================================
@dataclass(frozen=True)
class Config:
    """Configuration class to store all script parameters."""
    # --- Input/Output Paths ---
    phase_catalog_path: str = './input/example_catalog.pha'
    raw_data_dir: str = '/data5/Turkey/Turkey_clean'
    output_root_dir: str = './output/polarity/'
    output_polarity_file: str = 'example_polarity.dat'
    # Added log file path
    log_file_path: str = './logs/polarity_processing.log' 
    
    # --- Data Processing Parameters ---
    phase_num_threshold: int = 6
    data_segment_length: float = 5.0
    s_factor: float = 0.5
    frequency_band: List[float] = field(default_factory=lambda: [1, 20])
    MAX_CPU_CORE_LIMIT: int = 20
    save_mat_format: bool = False
    generate_plots: bool = True
    
    max_concurrent_processes: int = min(mp.cpu_count() or 1, MAX_CPU_CORE_LIMIT)

    # --- Derived Paths (read-only properties) ---
    @property
    def raw_result_dir(self) -> str:
        """Output directory for raw, individual results."""
        return os.path.join(self.output_root_dir, 'RawResult')

    @property
    def final_output_path(self) -> str:
        """Full path for the final aggregated polarity file."""
        return os.path.join(self.output_root_dir, self.output_polarity_file)

# ==============================================================================
# Worker Function
# ==============================================================================
def process_event_worker(task_args: Tuple[Dict[str, Any], Config]) -> str:
    """
    Executes the complete data processing workflow for a single seismic event.
    This function will run in a separate child process.
    """
    event_info, config = task_args
    evt_id = event_info['evt_id']
    station_code = event_info['station_code']

    # Initialize a logger for the worker process.
    # This is important because child processes don't inherit the parent's loggers directly.
    # Ensure log directory exists within the worker process as well.
    log_dir = os.path.dirname(config.log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    worker_logger = logging.getLogger(f'worker_{os.getpid()}')
    worker_logger.setLevel(logging.INFO)
    # Prevent adding multiple handlers if called multiple times in the same process
    if not any(isinstance(h, logging.FileHandler) for h in worker_logger.handlers):
        file_handler = logging.FileHandler(config.log_file_path)
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
        
        # Create a calculator instance inside the child process
        polarity_calculator = calc_solu()
        
        # 1. Read waveform data
        waveform_data = read_sac(
            data_root=config.raw_data_dir, netsta=station_code, P_arr=p_arrival,
            S_arr=s_arrival, dlength=config.data_segment_length,
            S_factor=config.s_factor, filter=config.frequency_band
        )

        if waveform_data is None or len(waveform_data) == 0:
            worker_logger.warning(f"SKIPPED: {evt_id}_{station_code} - Failed to read data or no data found.")
            return f"SKIPPED: {evt_id}_{station_code} - Failed to read data."

        # 2. Perform the calculation
        output_filename = f"{evt_id}_{station_code}.txt"
        polarity_calculator.solutionset(
            output_filename, waveform_data, config.output_root_dir,
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

# ==============================================================================
# Main Processor Class
# ==============================================================================
class PolarityProcessor:
    """Main class that encapsulates the entire polarity processing workflow."""
    def __init__(self, config: Config):
        self.config = config
        self.seismic_events = []
        self._setup_logging() # Setup logging when the processor is initialized

    def _setup_logging(self):
        """Sets up the main logger for the application."""
        log_dir = os.path.dirname(self.config.log_file_path)
        os.makedirs(log_dir, exist_ok=True) # Ensure log directory exists

        # Configure the root logger
        logging.basicConfig(
            level=logging.INFO, # Set the logging level (INFO, WARNING, ERROR, DEBUG, CRITICAL)
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file_path)
                # Removed StreamHandler to stop printing logs to console
            ]
        )
        self.logger = logging.getLogger('PolarityProcessor') # Get a specific logger for this class

    def _prepare(self):
        """Preparation phase: parse catalog, create output directories."""
        self.logger.info("--- Step 1: Preparing ---")
        self.seismic_events = parse_seismic_data(self.config.phase_catalog_path)
        os.makedirs(self.config.raw_result_dir, exist_ok=True)
        self.logger.info(f"Found {len(self.seismic_events)} total events in '{self.config.phase_catalog_path}'.")
        self.logger.info(f"Results will be saved to '{self.config.output_root_dir}'.")

    def _run_parallel_processing(self):
        """Parallel processing phase: use a process pool to dispatch and execute tasks."""
        self.logger.info("\n--- Step 2: Starting Parallel Processing ---")
        
        # Prepare a list of task arguments containing only simple data types
        task_list = []
        for event in self.seismic_events:
            evt_id, station_code, p_arrival, s_arrival, phase_count = event
            if phase_count >= self.config.phase_num_threshold:
                event_info = {
                    'evt_id': evt_id, 'station_code': station_code,
                    'p_arrival': p_arrival, 's_arrival': s_arrival
                }
                # Each task's argument is a tuple: (event_info_dict, config_object)
                task_list.append((event_info, self.config))
        
        if not task_list:
            self.logger.warning("No valid tasks found that meet the criteria.")
            return

        self.logger.info(f"Filtered {len(task_list)} valid tasks to be processed in parallel on {self.config.max_concurrent_processes} cores...")

        # Use imap_unordered for better performance and progress feedback
        with mp.Pool(processes=self.config.max_concurrent_processes) as pool:
            # tqdm provides a nice progress bar
            results_iterator = pool.imap_unordered(process_event_worker, task_list)
            for result in tqdm(results_iterator, total=len(task_list), desc="Processing Events"):
                # The worker function now logs detailed errors directly.
                if "FAILED" in result:
                    # No need to print traceback here, it's already in the log.
                    self.logger.error(f"Worker reported a FAILED task: {result}")
                elif "SKIPPED" in result:
                    self.logger.warning(f"Worker reported a SKIPPED task: {result}")
                else:
                    self.logger.info(f"Worker reported SUCCESS: {result}")

    def _post_process(self):
        """Post-processing phase: aggregate all individual result files."""
        self.logger.info("\n--- Step 3: Aggregating Final Results ---")
        try:
            process_polarity_data(
                raw_resdir=self.config.raw_result_dir,
                output_path=self.config.final_output_path,
                segment_length=self.config.data_segment_length
            )
            self.logger.info(f"Successfully aggregated results to '{self.config.final_output_path}'.")
        except Exception as e:
            self.logger.error(f"Error during post-processing: {e}", exc_info=True)

    def run(self):
        """Executes the complete workflow."""
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Automatic P-Wave Polarity Inversion Workflow Started")
        self.logger.info("=" * 60)
        
        self._prepare()
        self._run_parallel_processing()
        self._post_process()
        
        end_time = time.time()
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"All processes completed! Total elapsed time: {end_time - start_time:.2f} seconds.")
        self.logger.info("=" * 60)

# ==============================================================================
# Execution Entry Point
# ==============================================================================
def main_entry():
    """The main entry point for the script."""
    config = Config()
    processor = PolarityProcessor(config)
    processor.run()

if __name__ == "__main__":
    # Set the multiprocessing start method (fork is faster but spawn is safer on Unix-like systems)
    if sys.platform != 'win32':
        try:
            mp.set_start_method("fork")
        except RuntimeError:
            pass # Ignore if it has been already set

    # Ignore warning messages for a cleaner output
    
    main_entry()