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

def datestr(date):
    return '%4s%02d%02d'%(date.year,date.month,date.day)


def read_event_catalog(catalog_path: str) -> List[Event]:
    """
    read .pha file.
    """
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    events: List[Event] = []
    current_phases: List[StationPhase] = []
    current_event_id: int = None
    # Use a boolean to track if we are inside a valid event block
    in_event_block = False

    with open(catalog_path) as f:
        for line in f:
            line = line.strip() # 预先处理
            if line.startswith('#'):
                if current_event_id is not None:
                    events.append(Event(event_id=current_event_id, phases=current_phases))
                try:
                    parts = line.split(',')
                    current_event_id = int(parts[0][1:]) 
                    current_phases = []
                    
                except (IndexError, ValueError) as e:
                    print(f"警告: 无法解析事件头: '{line}'. 错误: {e}")
                    current_event_id = None 
                    current_phases = []

            elif current_event_id is not None and len(line) > 0:
                try:
                    parts = line.split(',')
                    if '.' not in parts[0]:
                        # print(f"警告: 跳过格式错误的台站: {parts[0]}")
                        continue
                    net, sta = parts[0].split('.')
                    if parts[1] == '-1' or parts[2] == '-1':
                        continue
                    
                    p_arrival = UTCDateTime(parts[1])
                    s_arrival = UTCDateTime(parts[2])
                    current_phases.append(StationPhase(net, sta, p_arrival, s_arrival))
                
                except Exception as e:
                    print(f"警告: 无法处理相位行: '{line}'. 错误: {e}")
                    continue
    if current_event_id is not None and current_phases:
        events.append(Event(event_id=current_event_id, phases=current_phases))
        
    print(f"成功: 从 '{os.path.basename(catalog_path)}' 解析了 {len(events)} 个事件。")
    return events


def load_and_prepare_stream(
    station_phase: StationPhase, data_root: str, time_padding: float, filter_band: List[float]
) -> Optional[Stream]:
    date_folder = datestr(station_phase.p_arrival)
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
        noise_start_idx = int((config.read_window - config.amp_window_npts/sample_rate - 0.5) * sample_rate)
        p_start_idx = int(config.read_window * sample_rate - config.offset_npts)
        s_start_idx = p_start_idx + int((station_phase.s_arrival - station_phase.p_arrival) * sample_rate)

        if (station_phase.s_arrival - station_phase.p_arrival) < config.amp_window_npts / sample_rate:
             return None

        data_e, data_n, data_z = (np.cumsum(tr.data) for tr in stream[:3])
        
        noise_amp = np.sqrt(
            _peak_to_peak(data_e, noise_start_idx, config.amp_window_npts)**2 +
            _peak_to_peak(data_n, noise_start_idx, config.amp_window_npts)**2 +
            _peak_to_peak(data_z, noise_start_idx, config.amp_window_npts)**2
        )
        p_amp = np.sqrt(
            _peak_to_peak(data_e, p_start_idx, config.amp_window_npts)**2 +
            _peak_to_peak(data_n, p_start_idx, config.amp_window_npts)**2 +
            _peak_to_peak(data_z, p_start_idx, config.amp_window_npts)**2
        )
        s_amp = np.sqrt(
            _peak_to_peak(data_e, s_start_idx, config.amp_window_npts)**2 +
            _peak_to_peak(data_n, s_start_idx, config.amp_window_npts)**2 +
            _peak_to_peak(data_z, s_start_idx, config.amp_window_npts)**2
        )
        
        p_snr = p_amp / noise_amp if noise_amp > 1e-9 else 0.0
        sp_ratio_log10 = np.log10(s_amp / p_amp) if p_amp > 1e-9 else 0.0
        
        if p_snr >= config.snr_threshold:
            return AmplitudeMetrics(
                station_name=station_phase.sta,
                network_code=station_phase.net,
                p_wave_snr=p_snr,
                sp_amplitude_ratio_log10=sp_ratio_log10
            )
        return None
    except Exception as e:
        return None