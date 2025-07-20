import numpy as np
import os
from typing import Any, List
from Waveform import Waveform
from data_pipeline import output_resfile

class calc_solu:
    def solutionset(
        self, 
        event_identifier: str, 
        raw_waveform_data: np.ndarray, 
        output_dir: str, 
        save_as_mat: bool, 
        generate_plot: bool
    ) -> None:
        """
        Executes the complete polarity analysis and result output workflow 
        for a single event's waveform data.

        Args:
            event_identifier (str): A unique identifier for the event (e.g., "20250628_ST01").
            raw_waveform_data (np.ndarray): The raw three-component waveform data.
            output_dir (str): The root directory for all output files.
            save_as_mat (bool): Whether to save the results in .mat format.
            generate_plot (bool): Whether to generate result plots.
        """
        # --- 1. Define Constants and Parameters ---
        # It is not recommended to modify this unless you need to test the algorithm.
        INITIAL_DT = 0.01            
        TIME_AMP_RATIO = 5 / 2       
        INTERP_FACTOR = 20       

        # --- 2. Prepare Output Paths ---
        raw_results_dir = os.path.join(output_dir, "RawResult")
        figures_dir = os.path.join(output_dir, "Figure")
        os.makedirs(raw_results_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        # --- 3. Core Data Processing Workflow ---
        waveform_processor = Waveform(f'{event_identifier}')
        waveform_processor.importdata(raw_waveform_data, INITIAL_DT) 
        waveform_processor.denseunique()
        waveform_processor.denselong(TIME_AMP_RATIO, INTERP_FACTOR)
        waveform_processor.extremearr()
        waveform_processor.densebin()
        analysis_state = waveform_processor.constructstate()
        markov_matrix, markov_state_count = analysis_state.markovmatrix()
        amplitude_probabilities = analysis_state.ampprobcalculate()
        
        # --- 4. Output Results ---
        output_resfile(
            waveform_processor,
            analysis_state,
            markov_matrix,
            amplitude_probabilities,
            markov_state_count,         # 'c_num'
            event_identifier,           # 'name'
            output_dir,                 # 'outdir'
            save_as_mat,                # 'is_mat'
            generate_plot               # 'is_plot'
        )
        return