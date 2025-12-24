# ==============================================================================
#  Copyright (c) 2025. Longtan Wang and Weilai Pei.
#
#  This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#  International License. To view a copy of this license, visit at
#  http://creativecommons.org/licenses/by-nc-sa/4.0/
# ==============================================================================
#
#  Authors:
#  - Longtan Wang (wanglt@sustech.edu.cn)
#    Department of Earth and Space Sciences,
#    Southern University of Science and Technology, Shenzhen, China
#
#  - Weilai Pei
#    Sinopec Petroleum Exploration and Production Research Institute,
#    Beijing, China
#
#  Contact: wanglt@sustech.edu.cn
#
#  This script is part of the FMFM program.
#  This script formats station, catalog, and polarity data into the specific
#  input paths required by the HASH focal mechanism determination program.
#
# ==============================================================================
import os, config
from typing import List, Dict
from datetime import datetime
from collections import defaultdict
import numpy as np
from obspy import UTCDateTime

# ==============================================================================
# 3. Main Processor Class
# ==============================================================================
class HASHInputFormatter:
    """A class to load raw data and format it into HASH input paths."""

    def __init__(self, config):
        self.config = config
        self.stations: Dict[str, Dict] = {}
        self.catalog: Dict[str, Dict] = {}
        self.polarities: Dict[str, List[Dict]] = {}
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"INFO: Output will be written to '{self.config.output_dir}'")

    def _load_stations(self):
        """Loads station data from the .sta path."""
        print(f"INFO: Loading stations from '{self.config.station_path}'...")
        with open(self.config.station_path) as f:
            for line in f:
                parts = line.strip().split(',')
                net, sta = parts[0].split('.')
                if sta not in self.stations:
                    self.stations[sta] = {
                        "net": net,
                        "sta": sta,
                        "lat": float(parts[1]),
                        "lon": float(parts[2]),
                        "ele": int(float(parts[3]))
                    }
        print(f" -> Loaded {len(self.stations)} unique stations.")

    def _load_catalog(self):
        """Loads event catalog data from the .pha path."""
        print(f"INFO: Loading event catalog from '{self.config.catalog_path}'...")
        with open(self.config.catalog_path) as f:
            for line in f:
                if not line.startswith('#'): continue
                parts = line.strip().split(',')
                event_id = parts[0][1:9]
                if event_id not in self.catalog:
                    self.catalog[event_id] = {
                        "event_id": event_id,
                        "source_time_str": parts[1],
                        "lat": float(parts[2]),
                        "lon": float(parts[3]),
                        "dep": float(parts[4]),
                        "mag": float(parts[5])
                    }
        print(f" -> Loaded {len(self.catalog)} events.")

    def _load_polarities(self):
        """Loading polarity data"""
        print(f"INFO: Loading polarity picks from '{self.config.polarity_path}'...")
        with open(self.config.polarity_path) as f:
            for line in f:
                parts = line.strip().split(',')
                event_id = parts[0].split('_')[0].zfill(8)
                net, sta = parts[0].split('_')[1].split('.')
                polarity_char = ''
                if parts[2] == 'up': polarity_char = 'U'
                elif parts[2] == 'down': polarity_char = 'D'
                else: continue  # Skip unknown polarities
                quality = 'I' if float(parts[3]) > self.config.polarity_probability_threshold else 'E'
                if event_id not in self.polarities: self.polarities[event_id] = []
                self.polarities[event_id].append({
                    "net": net, 
                    "sta": sta, 
                    "polarity": polarity_char, 
                    "quality": quality
                })
        print(f" -> Loaded polarity picks for {len(self.polarities)} events.")
    
    def process_amp_path(self):
        """
        read .amp file and calculate average log10(S/P) for station corr.
        """
        input_pathname = self.config.SPR_path   
        output_pathname = self.config.hash_statcor_path
        spr_all = []
        if self.config.is_calc_sta_corr:
            station_ratios = defaultdict(list)
            with open(input_pathname, 'r') as f:
                amp_data_raw = f.readlines()
            for line in amp_data_raw:
                parts = line.split()
                if len(parts) == 5:
                    station = parts[0]
                    amplitude_ratio = float(parts[3])
                    station_ratios[station].append(amplitude_ratio)
                    spr_all.append(amplitude_ratio)
                if len(parts) == 4:
                    station = parts[0][:5]
                    amplitude_ratio = float(parts[2])
                    station_ratios[station].append(amplitude_ratio)
                    spr_all.append(amplitude_ratio)
            average_ratios = {
                station: sum(ratios) / len(ratios)
                for station, ratios in station_ratios.items()
            }
            mean_spr_all = np.mean(spr_all)
            sorted_stations = sorted(average_ratios.keys())
            path = os.path.join(self.config.output_dir, self.config.hash_statcor_path)
            with open(path, 'w') as f:
                for station in sorted_stations:
                    average_write = average_ratios[station] - mean_spr_all
                    #average_write = mean_spr_all - average_ratios[station]
                    f.write(f"{station:<5s} HHZ XX{average_write:>8.4f}\n")
            return True
        else:
            return True


    def write_station_paths(self):
        """Writes the station, reverse files required by HASH."""
        print("INFO: Writing HASH station-related paths...")
        sorted_station_keys = sorted(self.stations.keys())

        # --- Write example.station path ---
        path = os.path.join(self.config.output_dir, self.config.hash_station_path)
        with open(path, 'w') as f:
            for sta_key in sorted_station_keys:
                st = self.stations[sta_key]
                for comp in ['HHE', 'HHN', 'HHZ']:
                    line = f"{st['net']:3} {st['sta']:<5} {comp:3}                                    {st['lat']:10.5f} {st['lon']:10.5f} {st['ele']:5d}\n"
                    f.write(line)

        # --- Write example.reverse path ---
        path = os.path.join(self.config.output_dir, self.config.hash_reverse_path)
        with open(path, 'w') as f:
            for sta_key in sorted_station_keys:
                f.write(f"{sta_key:<5}19900101 19900102\n")
        
                # --- Write example.statcor file ---
        path = os.path.join(self.config.output_dir, self.config.hash_statcor_path)
        comp = 'HHZ'
        with open(path, 'w') as f:
            for sta_key in sorted_station_keys:
                f.write(f"{sta_key:<5} {comp} XX       0\n")
        print(" -> Station files written successfully.")

    def write_phase_path(self):
        """Merges catalog and polarity data and writes the main HASH phase path."""
        print("INFO: Writing main HASH phase path...")
        path = os.path.join(self.config.output_dir, self.config.hash_phase_path)
        
        sorted_event_ids = sorted(self.polarities.keys())
        
        with open(path, 'w') as f:
            for event_id in sorted_event_ids:
                if event_id not in self.catalog:
                    print(f"WARNING: Event ID '{event_id}' found in polarity path but not in catalog. Skipping.")
                    continue
                
                event = self.catalog[event_id]
                picks = self.polarities[event_id]
                ot_utc = UTCDateTime(event['source_time_str'])
                ot_str = ot_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:]
                ot = datetime.strptime(ot_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                sec_str = f"{ot.second:02d}.{ot.microsecond // 10000:02d}"
                lat_deg, lat_min = int(event['lat']), 60 * (event['lat'] - int(event['lat']))
                lon_deg, lon_min = int(event['lon']), 60 * (event['lon'] - int(event['lon']))

                event_line = (
                    f"{ot.strftime('%Y%m%d%H%M')}{sec_str:>5s}"
                    f"{lat_deg:2d}N{lat_min:5.2f}"
                    f"{abs(lon_deg):3d}E{abs(lon_min):5.2f}"
                    f"{event['dep']:5.2f}{self.config.H_UNCERTAINTY:54.2f}"
                    f"{self.config.V_UNCERTAINTY:6.2f}{event['mag']:44.2f}"
                    f"{event_id:>22s}\n"
                )
                f.write(event_line)

                for pick in picks:
                    phase_line = f"{pick['sta']:<5}{pick['net']:2}  HHZ {pick['quality']} {pick['polarity']}\n"
                    f.write(phase_line)
                
                f.write(f"{'':64}{event_id:8s}\n")
        print(" -> HASH phase path written successfully.")

    def run(self):
        """Executes the complete formatting workflow."""
        self._load_stations()
        self._load_catalog()
        self._load_polarities()
        self.write_station_paths()
        self.write_phase_path()
        self.process_amp_path()
        print("\nSUCCESS: All HASH input paths have been generated.")

if __name__ == "__main__":
    config = config.Config()
    formatter = HASHInputFormatter(config)
    formatter.run()