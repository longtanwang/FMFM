# Waveform.py
import numpy as np
import obspy
from scipy.signal import argrelextrema
import pmi
import State
from matplotlib import pyplot as plt

class Waveform():
    """
    Represents and processes a single seismic waveform.

    This class encapsulates a waveform's data and provides a series of methods
    in a pipeline to import, resample, analyze, and prepare the data for
    state-based analysis.
    """
    def __init__(self,name):
        self.name=name
        self.num = 1

    def importdata(self, data, delta):
        """
        Loads the raw waveform data and metadata into the object.

        This method initializes the core data attributes, including the data array,
        the time step (delta), and calculates the initial timestamp array.

        Args:
            data (np.ndarray): The raw waveform amplitude data.
            delta (float): The time interval between samples in seconds.
        """
        self.data = data
        self.delta = delta
        self.length = len(self.data)
        self.timestamp=np.linspace(0,self.delta*(self.length-1),self.length)
        self.dataindexindense=np.arange(0,self.length,1)
        self.densetimestamp = self.timestamp

    def denseunique(self):
        """
        Calculates the unique absolute amplitude values from the waveform data.

        These unique values are later used as a series of thresholds for the
        binarization process in the `densebin` method.
        """
        self.dense_abs = abs(self.data)
        self.dense_abs_unique = np.unique(self.dense_abs)
        dense_abs_unique_index = []
        for i in range(0, len(self.dense_abs_unique)):
            dense_abs_unique_index.append(np.where(self.dense_abs == self.dense_abs_unique[i])[0])
        self.dense_abs_unique_index = dense_abs_unique_index
        self.threshold = self.dense_abs_unique

    def denselong(self,hvcoefficient,mininsertco):
        """
        Resamples the waveform non-uniformly to create a "densified" version.

        This method adds more points to segments where the waveform's gradient
        is steep, effectively increasing the temporal resolution in areas of
        rapid change. This new, longer array is used for detailed analysis.

        Args:
            hvcoefficient (float): A coefficient controlling the ratio of
                                   horizontal (time) to vertical (amplitude) change.
            mininsertco (float): A coefficient controlling the minimum number of
                                 points to insert between original samples.
        """
        hunit=hvcoefficient*(max(self.data)-min(self.data))/(self.length-1)
        vchange = abs(self.data[1:] - self.data[0:-1]) #
        self.longtimestamp=np.array([])
        self.dataindexinlong=np.array([])
        for i in range(0,self.length-1):
            insertpointnumber=int(np.round(np.sqrt(vchange[i]**2+hunit**2)*mininsertco/hunit))
            insertvector = np.linspace(self.timestamp[i], self.timestamp[i + 1], insertpointnumber + 1)
            if(i<self.length-2):
                self.dataindexinlong=np.append(self.dataindexinlong,len(self.longtimestamp))
                self.longtimestamp=np.append(self.longtimestamp,insertvector[0:-1])
            else:
                self.dataindexinlong = np.append(self.dataindexinlong, len(self.longtimestamp))
                self.longtimestamp=np.append(self.longtimestamp, insertvector)
                self.dataindexinlong = np.append(self.dataindexinlong, len(self.longtimestamp)-1)
        self.denselongdata=np.interp(self.longtimestamp, self.timestamp, self.data)
        self.denselongabs=abs(self.denselongdata)
        self.longlength=len(self.denselongdata)
        self.longextrememax = argrelextrema(self.denselongdata, np.greater_equal)[0]
        #self.testmin = argrelextrema(self.denselongdata, np.less_equal)[0]
        self.longextrememin = argrelextrema(-1 * self.denselongdata, np.greater_equal)[0]
        self.longextremeall = np.unique(np.append(self.longextrememax, self.longextrememin))


    def extremearr(self):
        """
        Creates an array based on the influence of local extrema.
        Use like sign maximum near extreme point

        This method processes the "long" waveform to generate an array (`longextremearr`)
        where the value at each point is determined by the amplitude of the most
        significant surrounding extremum of the same sign. It's a form of
        envelope detection or feature extraction.
        """
        self.longextremearr=np.zeros(self.longlength)
        for i in range(1, len(self.longextremeall)):
            if (self.denselongdata[self.longextremeall[i]] * self.denselongdata[self.longextremeall[i - 1]] > 0):
                if (self.denselongabs[self.longextremeall[i]] >= self.denselongabs[self.longextremeall[i - 1]]):
                    self.longextremearr[self.longextremeall[i - 1]:self.longextremeall[i] + 1] = self.denselongabs[self.longextremeall[i]]
                if (self.denselongabs[self.longextremeall[i]] < self.denselongabs[self.longextremeall[i - 1]]):
                    self.longextremearr[self.longextremeall[i - 1]:self.longextremeall[i] + 1] = self.denselongabs[self.longextremeall[i - 1]]
            else:
                signarray=self.denselongdata[self.longextremeall[i - 1]:self.longextremeall[i]]*self.denselongdata[self.longextremeall[i - 1]+1:self.longextremeall[i]+1]
                turnpoint=np.where(signarray<=0)[0][0]+1
                self.longextremearr[self.longextremeall[i - 1]:self.longextremeall[i - 1]+turnpoint]=self.denselongabs[self.longextremeall[i - 1]]
                self.longextremearr[self.longextremeall[i - 1] + turnpoint:self.longextremeall[i]+1] = self.denselongabs[self.longextremeall[i]]
        self.longextremearrmin=np.min(self.longextremearr)


    def densebin(self):
        """
        Performs the main analysis by iterating through amplitude thresholds.

        For each threshold, this method binarizes the waveform, calls an external
        PMI optimizer to find optimal cuts, and processes these cuts to calculate
        various features like peak amplitudes and noise characteristics. This is the
        primary calculation loop that prepares data for the `constructstate` method.
        """
        self.mivalue, self.pmivalue, self.cut, self.peak, self.noiselength, self.noiseindex, self.chances = [], [], [], [], [], [], []
        self.longextremeall.sort()
        for i in range(len(self.threshold)):
            current_threshold = self.threshold[i]
            zeroindex_dense = np.where(abs(self.denselongdata) <= current_threshold)[0]
            if current_threshold > self.longextremearrmin:
                zeroindex = np.where(abs(self.longextremearr) <= current_threshold)[0]
            else:
                zeroindex = zeroindex_dense
            longarray = np.ones(self.longlength + 1)
            if len(zeroindex) > 0: longarray[zeroindex] = 0
            mivalue, pmivalue, longcutsolution = pmi.maxpmi_optimized(longarray)
            if len(zeroindex_dense) > 0.95 * self.longlength:
                mivalue, pmivalue, longcutsolution = 0, 0, np.array([self.longlength])
            if current_threshold > self.longextremearrmin:
                for j in range(len(longcutsolution)):
                    if longcutsolution[j] < self.longlength:
                        cut_val = longcutsolution[j] + 1
                        search_slice = self.denselongabs[cut_val:]
                        if search_slice.size > 0:
                            gt_threshold = search_slice > current_threshold
                            if np.any(gt_threshold): longcutsolution[j] = cut_val + np.argmax(gt_threshold)
                            else: longcutsolution[j] = self.longlength
                        else: longcutsolution[j] = self.longlength
            longcutsolution = np.sort(longcutsolution)
            self.mivalue.append(mivalue); self.pmivalue.append(pmivalue); self.cut.append(longcutsolution - 1)
            peakcollect, noiselength_inner, noiseindex_inner, chances_inner = [], [], [], []
            for j in range(len(longcutsolution)):
                cut_val = int(longcutsolution[j])
                oriindex = zeroindex_dense[zeroindex_dense < cut_val]
                nindex, chance = findnoise(oriindex, self.dataindexinlong, self.dataindexindense, cut_val - 1)
                noiseindex_inner.append(nindex); chances_inner.append(chance); noiselength_inner.append(len(nindex) if hasattr(nindex, '__len__') else 1)
                if cut_val == self.longlength: peakcollect.append(0)
                else:
                    try:
                        next_extreme_pos_idx = np.searchsorted(self.longextremeall, cut_val - 1, side='right')
                        actual_extreme_pos = self.longextremeall[next_extreme_pos_idx]
                        peakcollect.append(self.denselongdata[actual_extreme_pos])
                    except IndexError:
                        peakcollect.append(0)
            self.peak.append(peakcollect); self.noiselength.append(noiselength_inner); self.noiseindex.append(noiseindex_inner); self.chances.append(chances_inner)


    def constructstate(self):
        """
        Constructs and populates a State object from the analysis results.

        This method transforms the results calculated in `densebin` (which are
        organized by threshold) into a state-based representation suitable for
        Markov matrix analysis. It differentiates between single and combined
        ("combo") states.

        Returns:
            State.State: An initialized State object containing all processed data.
        """
        state = State.State(self.name)
        for i in range(0,len(self.threshold)):
                if(i==len(self.threshold)-1):
                    downthreshold = self.threshold[i]
                    upthreshold=np.inf
                else:
                    downthreshold = self.threshold[i]
                    upthreshold = self.threshold[i + 1]
                if(len(self.noiselength[i])>1):
                    noisecombo=[]
                    for j in range(0,len(self.noiselength[i])):
                        noisecombo.append(self.data[self.noiseindex[i][j].astype('int')])
                    state.addcombo(len(self.noiselength[i]), downthreshold,upthreshold, noisecombo,self.chances[i], self.pmivalue[i], self.peak[i],self.longtimestamp[self.cut[i]])
                else:
                    state.addstate(downthreshold,upthreshold, self.data[self.noiseindex[i][0].astype('int')], self.chances[i],self.pmivalue[i], self.peak[i],self.longtimestamp[self.cut[i]])
        return state

def findnoise(oriindex,dataindexinlong,dataindexindense,cutsolution):
    """
    Identifies noise segments based on original and dense index mappings.

    This function takes a set of 'original' indices (oriindex) and determines
    which bins, defined by 'dataindexinlong', they fall into. It then returns
    the corresponding indices from 'dataindexindense' for those bins.

    Args:
        oriindex: An array of indices in the original data scale.
        dataindexinlong: An array mapping original indices to the 'long' data scale.
        dataindexindense: An array mapping original indices to the 'dense' data scale.
        cutsolution: The cut point solution.

    Returns:
        A tuple containing the noise indices and the number of unique bins (chance).
    """
    if cutsolution == dataindexinlong[-1]:
        return np.arange(0, dataindexindense[-1] + 1), len(dataindexindense) - 1
    if len(oriindex) == 0: return np.array([]), 0
    bin_indices = np.searchsorted(dataindexinlong[:-1], oriindex, side='right') - 1
    last_bin_mask = (oriindex >= dataindexinlong[-2]) & (oriindex <= dataindexinlong[-1])
    bin_indices[last_bin_mask] = len(dataindexinlong) - 2
    distributionrange = np.unique(bin_indices)
    distributionrange = distributionrange[distributionrange >= 0].astype(int)
    chance = len(distributionrange)
    if chance == 0: return np.array([]), 0
    if chance == 1:
        idx = distributionrange[0]
        if idx + 1 < len(dataindexindense): return np.arange(dataindexindense[idx], dataindexindense[idx + 1] + 1), 1
        else: return np.array([]), 1
    else:
        ranges_to_concat = [np.arange(dataindexindense[i], dataindexindense[i+1] + 1) for i in distributionrange[:-1] if i + 1 < len(dataindexindense)]
        if not ranges_to_concat: return np.array([]), chance
        noiseindex = np.unique(np.concatenate(ranges_to_concat))
        return noiseindex, chance
