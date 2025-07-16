# ==============================================================================
#  Copyright (c) 2025. Longtan Wang and Weilai Pei.
#
#  This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#  International License. To view a copy of this license, visit at
#  http://creativecommons.org/licenses/by-nc-sa/4.0/.
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
#  This script is part of the POSE-FMS program.
#  It serves as the main driver for the HASH pipeline, executing the HASH
#  program and processing its output into a user-friendly format.
#
# ------------------------------------------------------------------------------
#  Acknowledgement of Integrated Software: HASH
# ------------------------------------------------------------------------------
#  This software (POSE-FMS) utilizes the HASH_v1.2 for focal mechanism
#  inversion. HASH is a third-party program developed by Hardebeck and Shearer
#  and is subject to its own terms of use.
#
#  Users of this software should cite the original HASH publication:
#
#  Hardebeck, J. L., & Shearer, P. M. (2002). A new method for determining
#  first-motion focal mechanisms. Bulletin of the Seismological Society of
#  America, 92(6), 2264-2276.
# ==============================================================================

import os, sys
import csv
from dataclasses import dataclass
# --- Add project source directory to the system path ---
sys.path.append('../src')
from HASH_utils import run_external_program, HashOutputParser

# =========================
# Configuration Area
# =========================
@dataclass(frozen=True)
class Config:
    """Stores all file paths and parameters for the HASH pipeline."""
    # --- Input files ---
    hash_input_file: str = './output/focal_mechanisms/example.inp'
    hash_output_raw_file: str = './output/focal_mechanisms/example_raw.out'
    
    # --- Output file ---
    final_fms_csv_file: str = './output/focal_mechanisms/example_fms.csv'
    
    # --- Path to HASH executable ---
    hash_executable: str = '../bin/hash_driver'

# =========================
# Main Pipeline Orchestrator
# =========================
class HashPipeline:
    """A class to orchestrate the HASH execution and output processing workflow."""
    def __init__(self, config: Config):
        self.config = config

    def run_hash_driver(self) -> bool:
        """Step 1: Run the external HASH executable."""
        return run_external_program(
            executable_path=self.config.hash_executable,
            input_file_path=self.config.hash_input_file
        )

    def process_and_save_results(self):
        """Step 2: Parse the raw HASH output and save it as a clean CSV."""
        try:
            parser = HashOutputParser(self.config.hash_output_raw_file)
            best_solutions = parser.parse()

            # Write the results to a CSV file
            output_dir = os.path.dirname(self.config.final_fms_csv_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            header = [
                "event_id", "time", "longitude", "latitude", "depth", "magnitude",
                "strike", "dip", "rake", "quality_rank", "a_plane_uncertainty",
                "b_plane_uncertainty", "probability", "p_phase_num",
                "p_misfit", "s_phase_num", "s_misfit"
            ]
            
            with open(self.config.final_fms_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for sol in sorted(best_solutions.values(), key=lambda s: s.origin_time):
                    writer.writerow([
                        sol.event_id, sol.origin_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z',
                        f"{sol.longitude:.4f}", f"{sol.latitude:.4f}", f"{sol.depth:.2f}", f"{sol.magnitude:.2f}",
                        sol.strike, sol.dip, sol.rake, sol.quality,
                        sol.a_plane_uncertainty, sol.b_plane_uncertainty, sol.probability,
                        sol.p_phase_count, sol.p_misfit, sol.s_phase_count, sol.s_misfit
                    ])
            print(f"SUCCESS: Formatted results saved to '{self.config.final_fms_csv_file}'")

        except FileNotFoundError as e:
            print(f"ERROR: Cannot process results, file not found: {e}", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during result processing: {e}", file=sys.stderr)

    def run(self):
        """Executes the full pipeline: run HASH, then process outputs."""
        if self.run_hash_driver():
            self.process_and_save_results()
        else:
            print("FATAL: HASH driver execution failed. Skipping result processing.", file=sys.stderr)

# =========================
# 3. Execution Entry Point
# =========================
if __name__ == "__main__":
    config = Config()
    pipeline = HashPipeline(config)
    pipeline.run()