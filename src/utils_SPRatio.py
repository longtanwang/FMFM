import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Any

import numpy as np
from obspy import UTCDateTime, Stream, read

# ==============================================================================
# Data Classes
# ==============================================================================

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

@dataclass
class AmplitudeMetrics:
    """Stores the calculated amplitude metrics for a station."""
    station_name: str
    network_code: str
    p_wave_snr: float
    sp_amplitude_ratio_log10: float


# ==============================================================================
# Processing Functions
# ==============================================================================

def parse_event_catalog(catalog_path: str) -> List[Event]:
    """
    Parses a seismic event catalog file.
    This version is modified to be 100% logically compatible with the original script,
    meaning it will also include events that have no valid phases.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    events: List[Event] = []
    current_phases: List[StationPhase] = []
    # Use a boolean to track if we are inside a valid event block
    in_event_block = False

    with open(catalog_path) as f:
        for line in f:
            if line.startswith('#'):
                # When a new event header is found, save the previous event
                # FIX: Also save the previous event even if current_phases is empty
                if in_event_block:
                    event_id = len(events) + 1
                    events.append(Event(event_id=event_id, phases=current_phases))
                
                # start a new event
                current_phases = []
                in_event_block = True

            elif in_event_block and len(line.strip()) > 0:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        net, sta = parts[0].split('.')
                        p_arrival = UTCDateTime(parts[1])
                        s_arrival = UTCDateTime(parts[2])
                        current_phases.append(StationPhase(net, sta, p_arrival, s_arrival))
                    except Exception:
                        continue
    
    # add final event in file
    if in_event_block:
        event_id = len(events) + 1
        events.append(Event(event_id=event_id, phases=current_phases))
        
    print(f"Successfully parsed {len(events)} event blocks from catalog '{os.path.basename(catalog_path)}'.")
    return events


def load_and_prepare_stream(
    station_phase: StationPhase, data_root: str, time_padding: float, filter_band: List[float]
) -> Optional[Stream]:
    date_folder = f"{station_phase.p_arrival.year}{station_phase.p_arrival.month:02d}{station_phase.p_arrival.day:02d}"
    file_pattern = os.path.join(data_root,  date_folder, f"*{station_phase.sta}*.mseed")
    sac_files = glob.glob(file_pattern)

    if len(sac_files) != 3:
        # print(f"Warning: Incomplete data for station {station_phase.sta}: found {len(sac_files)} files, expected 3.")
        return None

    try:
        st = Stream()
        start_time = station_phase.p_arrival - time_padding
        end_time = station_phase.s_arrival + time_padding
        for sac_file in sac_files[:3]:
            st += read(sac_file, starttime=start_time, endtime=end_time)
        
        st.detrend('constant')
        st.filter('bandpass', freqmin=filter_band[0], freqmax=filter_band[1])
        # st.merge(method=1, fill_value=0)
        
        return st
    except Exception as e:
        # print(f"Error processing waveform for {station_phase.sta}: {e}")
        return None


def calculate_amplitude_metrics(
    stream: Stream, station_phase: StationPhase, config: Any
) -> Optional[AmplitudeMetrics]:
    def _peak_to_peak(data: np.ndarray, start_idx: int, window: int) -> float:
        segment = data[start_idx : start_idx + window]
        return np.ptp(segment) if len(segment) == window else 0.0

    try:
        sample_rate = stream[0].stats.sampling_rate
        noise_start_idx = int((config.TIME_PADDING_SEC - 2.5) * sample_rate)
        p_start_idx = int(config.TIME_PADDING_SEC * sample_rate - config.OFFSET_NPTS)
        s_start_idx = p_start_idx + int((station_phase.s_arrival - station_phase.p_arrival) * sample_rate - config.OFFSET_NPTS)

        if (station_phase.s_arrival - station_phase.p_arrival) < config.AMP_WINDOW_NPTS / sample_rate:
             return None

        data_e, data_n, data_z = (np.cumsum(tr.data) for tr in stream[:3])
        
        noise_amp = np.sqrt(
            _peak_to_peak(data_e, noise_start_idx, 100)**2 +
            _peak_to_peak(data_n, noise_start_idx, 100)**2 +
            _peak_to_peak(data_z, noise_start_idx, 100)**2
        )
        p_amp = np.sqrt(
            _peak_to_peak(data_e, p_start_idx, config.AMP_WINDOW_NPTS)**2 +
            _peak_to_peak(data_n, p_start_idx, config.AMP_WINDOW_NPTS)**2 +
            _peak_to_peak(data_z, p_start_idx, config.AMP_WINDOW_NPTS)**2
        )
        s_amp = np.sqrt(
            _peak_to_peak(data_e, s_start_idx, config.AMP_WINDOW_NPTS)**2 +
            _peak_to_peak(data_n, s_start_idx, config.AMP_WINDOW_NPTS)**2 +
            _peak_to_peak(data_z, s_start_idx, config.AMP_WINDOW_NPTS)**2
        )
        
        p_snr = p_amp / noise_amp if noise_amp > 1e-9 else 0.0
        sp_ratio_log10 = np.log10(s_amp / p_amp) if p_amp > 1e-9 else 0.0
        
        if p_snr >= config.SNR_THRESHOLD:
            return AmplitudeMetrics(
                station_name=station_phase.sta,
                network_code=station_phase.net,
                p_wave_snr=p_snr,
                sp_amplitude_ratio_log10=sp_ratio_log10
            )
        return None
    except Exception as e:
        return None