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
#  This script is part of the POSE-FMS program.
#  This script formats station, catalog, and polarity data into the specific
#  input files required by the HASH focal mechanism determination program.
#
# ==============================================================================
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# ==============================================================================
# 1. Configuration Class
# ==============================================================================
@dataclass(frozen=True)
class Config:
    """Stores all file paths and constant parameters for the formatting process."""
    station_file: str = './input/example_station.sta'
    catalog_file: str = './input/example_catalog.pha'
    polarity_file: str = './output/polarity/example_polarity.dat'
    output_dir: str = './output/focal_mechanisms/HASH_io'

    # output filenames
    hash_station_file: str = 'example.station'
    hash_reverse_file: str = 'example.reverse'
    hash_statcor_file: str = 'example.statcor'
    hash_phase_file: str = 'example.phase'

    # Constants
    DEFAULT_NETWORK: str = 'TU'
    H_UNCERTAINTY: float = 0.07
    V_UNCERTAINTY: float = 0.10


# ==============================================================================
# 2. Data Model Classes
# ==============================================================================
@dataclass
class Station:
    """Represents a single seismic station."""
    net: str
    sta: str
    lat: float
    lon: float
    ele: int

@dataclass
class CatalogEvent:
    """Represents a single event from the catalog."""
    event_id: str
    origin_time_str: str
    lat: float
    lon: float
    dep: float
    mag: float

@dataclass
class PolarityPick:
    """Represents a single polarity pick for a station."""
    net: str
    sta: str
    polarity: str  # 'U' for Up, 'D' for Down
    quality: str   # 'I' for impulsive, 'E' for emergent


# ==============================================================================
# 3. Main Processor Class
# ==============================================================================
class HASHInputFormatter:
    """A class to load raw data and format it into HASH input files."""

    def __init__(self, config: Config):
        self.config = config
        self.stations: Dict[str, Station] = {}
        self.catalog: Dict[str, CatalogEvent] = {}
        self.polarities: Dict[str, List[PolarityPick]] = {}
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"INFO: Output will be written to '{self.config.output_dir}'")

    def _load_stations(self):
        """Loads station data from the .sta file."""
        print(f"INFO: Loading stations from '{self.config.station_file}'...")
        with open(self.config.station_file) as f:
            for line in f:
                parts = line.strip().split(',')
                net, sta = parts[0].split('.')
                if sta not in self.stations:
                    self.stations[sta] = Station(
                        net=net,
                        sta=sta,
                        lat=float(parts[1]),
                        lon=float(parts[2]),
                        ele=int(float(parts[3]))
                    )
        print(f" -> Loaded {len(self.stations)} unique stations.")

    def _load_catalog(self):
        """Loads event catalog data from the .pha file."""
        print(f"INFO: Loading event catalog from '{self.config.catalog_file}'...")
        with open(self.config.catalog_file) as f:
            for line in f:
                if not line.startswith('#'): continue
                parts = line.strip().split(',')
                event_id = parts[0][1:9]
                if event_id not in self.catalog:
                    self.catalog[event_id] = CatalogEvent(
                        event_id=event_id,
                        origin_time_str=parts[1],
                        lat=float(parts[2]),
                        lon=float(parts[3]),
                        dep=float(parts[4]),
                        mag=float(parts[5])
                    )
        print(f" -> Loaded {len(self.catalog)} events.")

    def _load_polarities(self):
        """Loads polarity data from the polarity.dat file."""
        print(f"INFO: Loading polarity picks from '{self.config.polarity_file}'...")
        with open(self.config.polarity_file) as f:
            for line in f:
                parts = line.strip().split(',')
                event_id = parts[0].split('_')[0].zfill(8)
                net, sta = parts[0].split('_')[1].split('.')
                
                polarity_char = ''
                if parts[2] == 'up':
                    polarity_char = 'U'
                elif parts[2] == 'down':
                    polarity_char = 'D'
                else:
                    continue  # Skip unknown polarities
                
                quality = 'I' if float(parts[3]) > 0.95 else 'E'

                if event_id not in self.polarities:
                    self.polarities[event_id] = []
                
                self.polarities[event_id].append(PolarityPick(net, sta, polarity_char, quality))
        print(f" -> Loaded polarity picks for {len(self.polarities)} events.")
    
    def write_station_files(self):
        """Writes the station, reverse, and statcor files required by HASH."""
        print("INFO: Writing HASH station-related files...")
        sorted_station_keys = sorted(self.stations.keys())

        # --- Write example.station file ---
        path = os.path.join(self.config.output_dir, self.config.hash_station_file)
        with open(path, 'w') as f:
            for sta_key in sorted_station_keys:
                st = self.stations[sta_key]
                for comp in ['HHE', 'HHN', 'HHZ']:
                    line = f"{self.config.DEFAULT_NETWORK:3} {st.sta:<5} {comp:3}                                     {st.lat:9.5f} {st.lon:10.5f} {st.ele:5d}\n"
                    f.write(line)

        # --- Write example.reverse file ---
        path = os.path.join(self.config.output_dir, self.config.hash_reverse_file)
        with open(path, 'w') as f:
            for sta_key in sorted_station_keys:
                f.write(f"{sta_key:<5}19900101 19900102\n")

        # --- Write example.statcor file ---
        path = os.path.join(self.config.output_dir, self.config.hash_statcor_file)
        with open(path, 'w') as f:
            for sta_key in sorted_station_keys:
                for comp in ['HHE', 'HHN', 'HHZ']:
                    f.write(f"{sta_key:<5} {comp} XX       0\n")
        print(" -> Station files written successfully.")

    def write_phase_file(self):
        """Merges catalog and polarity data and writes the main HASH phase file."""
        print("INFO: Writing main HASH phase file...")
        path = os.path.join(self.config.output_dir, self.config.hash_phase_file)
        
        sorted_event_ids = sorted(self.polarities.keys())
        
        with open(path, 'w') as f:
            for event_id in sorted_event_ids:
                if event_id not in self.catalog:
                    print(f"WARNING: Event ID '{event_id}' found in polarity file but not in catalog. Skipping.")
                    continue
                
                event = self.catalog[event_id]
                picks = self.polarities[event_id]
                
                ot = datetime.strptime(event.origin_time_str, "%Y%m%d%H%M%S.%f")
                sec_str = f"{ot.second:02d}.{ot.microsecond // 10000:02d}"

                lat_deg, lat_min = int(event.lat), 60 * (event.lat - int(event.lat))
                lon_deg, lon_min = int(event.lon), 60 * (event.lon - int(event.lon))

                event_line = (
                    f"{ot.strftime('%Y%m%d%H%M')}{sec_str:>5s}"
                    f"{lat_deg:2d}N{lat_min:5.2f}"
                    f"{abs(lon_deg):3d}E{abs(lon_min):5.2f}"
                    f"{event.dep:5.2f}{self.config.H_UNCERTAINTY:54.2f}"
                    f"{self.config.V_UNCERTAINTY:6.2f}{event.mag:44.2f}"
                    f"{event_id:>22s}\n"
                )
                f.write(event_line)

                for pick in picks:
                    phase_line = f"{pick.sta:<5}{pick.net:2}  HHZ {pick.quality} {pick.polarity}\n"
                    f.write(phase_line)
                
                f.write(f"{'':64}{event_id:8s}\n")
        print(" -> HASH phase file written successfully.")

    def run(self):
        """Executes the complete formatting workflow."""
        self._load_stations()
        self._load_catalog()
        self._load_polarities()
        self.write_station_files()
        self.write_phase_file()
        print("\nSUCCESS: All HASH input files have been generated.")


# ==============================================================================
# 4. Execution Entry Point
# ==============================================================================
if __name__ == "__main__":
    config = Config()
    formatter = HASHInputFormatter(config)
    formatter.run()