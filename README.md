# FMFM: A Python Package for Automatically Computing First Motion-based Focal Mechanism Solutions

![Version](https://img.shields.io/badge/version-v1.1-blue)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)

## 1. Introduction
The focal mechanisms of small-to-moderate earthquakes provide vital constraints for the analysis of complex fault structures and regional stress fields. Due to their short rupture durations, these events allow for a robust double-couple point source approximation (Lay and Wallace, 1995). Traditional determination methods, such as the HASH algorithm, rely on P-wave first-motion polarities and S/P amplitude ratios (Hardebeck and Shearer, 2002, 2003). However, in the context of high-density seismic arrays, traditional manual analysis is not only prohibitively inefficient but also susceptible to subjective bias. While deep-learning approaches have recently introduced automation, they frequently struggle with generalization issues when applied to tectonic regions that differ significantly from their original training datasets.

To overcome these challenges, **FMFM** (**F**irst **M**otion-based **F**ocal **M**echanism) offers an automatic, rule-based framework designed for the end-to-end processing of large-scale seismic data. By transitioning directly from raw waveforms to focal mechanism solutions, FMFM integrates the POSE algorithm (Wang et al., 2025) for probabilistic polarity identification, automated S/P ratio calculation and correction, and the HASH inversion module into a single streaming architecture. This integrated workflow eliminates the need for complex, multi-step data exchanges between disparate tools, providing a robust and efficient solution for generating high-resolution focal mechanism catalogs. 

The source code can be obtained at: https://github.com/longtanwang/FMFM. The methodology is discussed in more detail in our publications.

## 2. Installation
**FMFM** is a software suite designed for focal mechanism determination.

•	**Python Environment**: Users must first set up a suitable Python environment. This can be easily accomplished by installing Anaconda and the required dependencies.

•	**HASH Integration**: Additionally, users need to download the **HASH_v1.2** source code (https://www.usgs.gov/node/279393), extract it into the ./src directory, and compile it using the **Makefile** provided in our software package.

## 3. Getting Started
### 3.1 Data Preparation
FMFM requires continuous waveform data, earthquake catalogs, and phase arrival times as input.

•	Test Examples: The ```./example_projects``` directory already contains pre-configured catalogs and phase data. However, to run the demo scripts, you must download the sample waveforms (~600MB) separately.
* **Download Wavefroms:** Download waveform from https://github.com/longtanwang/FMFM/releases/latest
* **Data Setup:** After downloading and unzipping, place the waveform files into the following directory: ```./example_projects/input/example_data```

•	New Projects: When initiating a new project, please ensure the data is formatted according to these examples.

•	Parameter Configuration: 

General parameters are set in ```./example_projects/config.py```.
Parameters specifically for the HASH focal mechanism inversion are defined in ```./example_projects/input/example.inp```.

### 3.2 Step-by-Step Execution

Follow these steps in order to process your data:
* **1.	Determine P-wave first-motion polarity**: ```python 1.run_POSE.py```
* **2.	Calculate S/P amplitude ratios**: ```python 2.calc_SP_Ratio.py```
* **3.	Prepare for focal mechanism calculation**: ```python 3.prep_HASH.py```
* **4.	Execute focal mechanism calculation**: ```python 4.run_HASH.py```

### 3.3 Output Files

The output files are stored in the following locations:
* **Polarity Results**: ```./example_projects/output/polarity/example_polarity.dat```
* **Amplitude Ratios**: ```./example_projects/output/focal_mechanisms/HASH_io/example.amp```
* **Amplitude Ratio Corrections**: ```./example_projects/output/focal_mechanisms/HASH_io/example.statcor```
* **Final Focal Mechanism Solutions**: ```./example_projects/output/focal_mechanisms/example_fms.csv``` (Summary table)
```./example_projects/output/focal_mechanisms/example_raw.out``` (Raw HASH output)

# 4 Citation
### When using this software, please cite:

* **Wang, L.**, Meng, H., Zhou, Y., Hou, Y., Pei, W., and Zhou, S. (2025). **FMFM: A Python Package for Automatically Computing First Motion-based Focal Mechanism Solutions**, Seismol. Res. Lett. (under review).
* **Hardebeck, J. L.**, and Shearer, P. M. (2002). A new method for determining first-motion focal mechanisms. Bulletin of the Seismological Society of America, 92(6), 2264-2276.
* **Hardebeck, J. L.**, and Shearer, P. M. (2003). Using S/P Amplitude Ratios to Constrain the Focal Mechanisms of Small Earthquakes. Bulletin of the Seismological Society of America, 93(6), 2434–2444.

### Related Publications
* **Wang, L.**, Zhou, Y., Meng, H., Pei, W., and Zhou, S. (2025). P-wave First-motion Polarity Determination Using Order Statistics and Entropy Theory (POSE) with Applications to Southeastern Tibetan Plateau. Journal of Geophysical Research: Solid Earth, 131, e2025JB032118. https://doi.org/10.1029/2025JB032118.
* **Pei, W.**, Zhuang, J., and Zhou, S. (2025). Stochastic determination of arrival time and initial polarity of seismic waveform. Earth, Planets and Space, 77(1), 36.
