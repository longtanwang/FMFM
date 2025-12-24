import os
import subprocess
import sys
from dataclasses import dataclass
from threading import Thread
from typing import Dict, List, Optional

from obspy import UTCDateTime

# ==============================================================================
# Data Model
# ==============================================================================
@dataclass
class FocalMechanismSolution:
    """Stores a single, complete focal mechanism solution."""
    event_id: str
    origin_time: UTCDateTime
    longitude: float
    latitude: float
    depth: float
    magnitude: float
    strike: float
    dip: float
    rake: float
    quality: str  # 'A', 'B', 'C', or 'D'
    a_plane_uncertainty: float
    b_plane_uncertainty: float
    probability: float
    p_phase_count: int
    p_misfit: float
    s_phase_count: str # HASH output might be string '**'
    s_misfit: str      # HASH output might be string '**'

# ==============================================================================
# Run HASH
# ==============================================================================
def run_external_program(executable_path: str, input_file_path: str) -> bool:
    """
    Executes an external program with redirected input and real-time output streaming.

    Args:
        executable_path: Path to the executable file (e.g., './hash_driver3').
        input_file_path: Path to the input file to be fed into stdin.

    Returns:
        True if the process completes with a zero exit code, False otherwise.
    """
    def _stream_reader(stream, stream_type):
        """Reads a stream line by line and prints it."""
        for line in iter(stream.readline, ''):
            print(f"[{stream_type}] {line.strip()}")
        stream.close()

    full_exe_path = os.path.abspath(executable_path)
    full_input_path = os.path.abspath(input_file_path)

    if not os.path.exists(full_exe_path):
        print(f"ERROR: Executable not found at {full_exe_path}", file=sys.stderr)
        return False
    if not os.path.exists(full_input_path):
        print(f"ERROR: Input file not found at {full_input_path}", file=sys.stderr)
        return False

    print(f"INFO: Executing '{os.path.basename(full_exe_path)}' with input from '{os.path.basename(full_input_path)}'...")
    
    try:
        with open(full_input_path, 'r') as stdin_file:
            proc = subprocess.Popen(
                [full_exe_path],
                stdin=stdin_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Create and start threads to read stdout and stderr
            stdout_thread = Thread(target=_stream_reader, args=(proc.stdout, "HASH_STDOUT"))
            stderr_thread = Thread(target=_stream_reader, args=(proc.stderr, "HASH_STDERR"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Wait for the process to complete
            proc.wait()
            
            print("\nINFO: HASH program execution finished.")
            print(f"INFO: Exit code: {proc.returncode}")
            return proc.returncode == 0

    except Exception as e:
        print(f"FATAL: An error occurred while running the external program: {e}", file=sys.stderr)
        return False

# ==============================================================================
# HASH Output
# ==============================================================================
class HashOutputParser:
    """A parser for the complex, fixed-width HASH output file."""

    def __init__(self, hash_output_file: str):
        if not os.path.exists(hash_output_file):
            raise FileNotFoundError(f"HASH output file not found: {hash_output_file}")
        self.hash_output_file = hash_output_file
        self.best_solutions: Dict[str, FocalMechanismSolution] = {}

    def _parse_line(self, line: str) -> Optional[FocalMechanismSolution]:
        """Parses a single line of the HASH output file."""
        parts = list(filter(None, line.strip().split(' ')))
        if not parts or len(parts) < 33:
            return None

        try:
            avg_un = (float(parts[24]) + float(parts[25])) / 2
            prob = float(parts[29])
            if parts[27] == '**': return None 
            po_misfit = float(parts[27])
            
            # Quality control
            if avg_un <= 25 and prob >= 80 and po_misfit <= 20: rank = 'A'
            elif avg_un <= 35 and prob >= 60 and po_misfit <= 30: rank = 'B'
            elif avg_un <= 45 and prob >= 50 and po_misfit <= 40: rank = 'C'
            else: rank = 'D'

            return FocalMechanismSolution(
                event_id=parts[0],
                origin_time=UTCDateTime(f"{parts[1]}-{parts[2]}-{parts[3]}T{parts[4]}:{parts[5]}:{parts[6]}"),
                longitude=float(parts[11]),
                latitude=float(parts[10]),
                depth=float(parts[12]),
                magnitude=float(parts[8]),
                strike=float(parts[21]),
                dip=float(parts[22]),
                rake=float(parts[23]),
                quality=rank,
                a_plane_uncertainty=float(parts[24]),
                b_plane_uncertainty=float(parts[25]),
                probability=prob,
                p_phase_count=int(parts[26]),
                p_misfit=po_misfit,
                s_phase_count=parts[31],
                s_misfit=parts[32]
            )
        except (IndexError, ValueError) as e:
            # print(f"Warning: Could not parse line due to format error: {e}. Line: '{line.strip()}'")
            return None

    def parse(self) -> Dict[str, FocalMechanismSolution]:
        """
        Parses the entire file and returns a dictionary with only the best
        solution for each event.
        """
        print(f"INFO: Parsing HASH output from '{self.hash_output_file}'...")
        with open(self.hash_output_file, 'r') as f:
            for line in f:
                solution = self._parse_line(line)
                if solution:
                    # use solution with highest quality
                    if (solution.event_id not in self.best_solutions or 
                            solution.quality < self.best_solutions[solution.event_id].quality):
                        self.best_solutions[solution.event_id] = solution
        
        print(f" -> Found {len(self.best_solutions)} unique best-quality solutions.")
        return self.best_solutions