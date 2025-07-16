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
#  This script is part of the POSE_HASH program for calculating S/P amplitude ratio.
# ==============================================================================

import os, sys
from dataclasses import dataclass, field
from typing import List, Any
# --- Add project source directory to the system path ---
sys.path.append('../src')
from utils_SPRatio import (
    parse_event_catalog,
    load_and_prepare_stream,
    calculate_amplitude_metrics
)

# =========================
# Configuration Area
# =========================
@dataclass(frozen=True)
class Config:
    """io path"""
    DATA_ROOT: str = '/data5/Turkey/Turkey_clean'
    CATALOG_PATH: str = './input/example_catalog.pha'
    OUTPUT_PATH: str = './output/focal_mechanisms/HASH_io/example.amp'

    # Parameters for calculating S/P ratio
    TIME_PADDING_SEC: float = 10.0      # time padding before P and after S
    OFFSET_NPTS: int = 50               # npts shifted before the P&S arrival
    AMP_WINDOW_NPTS: int = 200           # window length npts for P and S
    SNR_THRESHOLD: float = 3.0          # P/noise SNR threshold
    FILTER_BAND: List[float] = field(default_factory=lambda: [1, 10]) # Bandpass filter

# =========================
# Main Processor
# =========================
class AmplitudeProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.events = []

    def load_catalog(self):
        self.events = parse_event_catalog(self.config.CATALOG_PATH)

    def process_and_write_results(self):
        """
        Iterates through all events and their stations, processes the data,
        and writes the results to the output file.
        """
        print("INFO: Starting event processing...")
        output_directory = os.path.dirname(self.config.OUTPUT_PATH)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        with open(self.config.OUTPUT_PATH, 'w') as f_out:
            for event in self.events:
                valid_station_metrics = []
                for station_phase in event.phases:
                    # Step 1: Read and process waveform
                    stream = load_and_prepare_stream(
                        station_phase, self.config.DATA_ROOT,
                        self.config.TIME_PADDING_SEC, self.config.FILTER_BAND
                    )
                    if not stream:
                        continue
                    
                    # Step 2: Calculate
                    metrics = calculate_amplitude_metrics(stream, station_phase, self.config)
                    if not metrics:
                        continue
                    
                    valid_station_metrics.append(metrics)
                
                # Step 3: Write results for the current event if any stations were valid
                if valid_station_metrics:
                    f_out.write(f"{event.event_id:08d}{len(valid_station_metrics):7d}\n")
                    for m in valid_station_metrics:
                        f_out.write(
                            f"{m.station_name:5}HHZ {m.network_code:2} "
                            f"{m.sp_amplitude_ratio_log10:26.3f} {m.p_wave_snr:10.3f}\n"
                        )
                    print(f"INFO: Event {event.event_id:08d} processed with {len(valid_station_metrics)} valid stations.")

    def run(self):
        """Executes the complete workflow."""
        self.load_catalog()
        self.process_and_write_results()
        print(f"\nSUCCESS: Processing complete. Results are saved to {self.config.OUTPUT_PATH}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    # Create a configuration instance
    config = Config()
    
    # Create a processor instance and run the workflow
    processor = AmplitudeProcessor(config)
    processor.run()