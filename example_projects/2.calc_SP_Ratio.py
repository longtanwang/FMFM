# ==============================================================================
#  Copyright (c) 2025. Longtan Wang and Weilai Pei.
#
#  This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#  International License. To view a copy of this license, visit at
#  http://creativecommons.org/licenses/by-nc-sa/4.0/
# ==============================================================================
#
#  Authors:
#  - Longtan Wang
#    Department of Earth and Space Sciences,
#    Southern University of Science and Technology, Shenzhen, China
#
#  - Weilai Pei
#    Sinopec Petroleum Exploration and Production Research Institute,
#    Beijing, China
#
#  Contact: wanglt@sustech.edu.cn
#
#  This script is part of the FMFM program for calculating S/P amplitude ratio.
# ==============================================================================

import os, sys, config, logging
from dataclasses import dataclass, field
from typing import List, Any
from tqdm import tqdm

sys.path.append('../src')

class AmplitudeProcessor:
    def __init__(self, config):
        self.config = config
        self.events = []
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Configures the logger to write to the path specified in config."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():  logger.handlers.clear()
        log_dir = os.path.dirname(self.config.log_file_path_SPR)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
            
        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(self.config.log_file_path_SPR, mode='w')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def load_catalog(self):
        """Loads the event catalog using the function from the config module."""
        self.events = config.read_event_catalog(self.config.catalog_path)

    def process_and_write_results(self):
        """
        Iterates through all events and their stations, processes the data,
        and writes the results to the output file.
        """
        self.logger.info("Starting event processing...")
        output_directory = os.path.dirname(self.config.SPR_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        with open(self.config.SPR_path, 'w') as f_out:
            # Wrap the events iterable with tqdm for a progress bar
            for event in tqdm(self.events, desc="Processing Events"):
                valid_station_metrics = []
                if len(event.phases) < self.config.min_phase_num: continue
                
                for station_phase in event.phases:
                    # Read and process waveform
                    stream = config.load_and_prepare_stream(
                        station_phase, self.config.seis_data_dir,
                        self.config.read_window, self.config.freq_band_SPR
                    )
                    
                    if not stream: continue

                    # Calculate amplitude metrics
                    metrics = config.calculate_amplitude_metrics(stream, station_phase, self.config)
                    if not metrics: continue
                    
                    valid_station_metrics.append(metrics)
                
                # Write results for the current event if any stations were valid
                if valid_station_metrics:
                    f_out.write(f"{event.event_id:08d}{len(valid_station_metrics):7d}\n")
                    for m in valid_station_metrics:
                        f_out.write(
                            f"{m.station_name:5}HHZ {m.network_code:2} "
                            f"{m.sp_amplitude_ratio_log10:26.3f} {m.p_wave_snr:10.3f}\n"
                        )
                    self.logger.info(f"Event {event.event_id:08d} processed with {len(valid_station_metrics)} stations.")

    def run(self):
        """Executes the complete workflow."""
        self.load_catalog()
        self.process_and_write_results()
        self.logger.info(f"Processing complete. Results are saved to {self.config.SPR_path}")

if __name__ == "__main__":
    # Create a configuration instance
    config = config.Config()
    
    # Create a processor instance and run the workflow
    processor = AmplitudeProcessor(config)
    processor.run()