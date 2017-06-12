import numpy as np
import scipy.interpolate as ipl
import scipy.signal as sp
from .common import *

def transformPitch(f0, hFreq, sinusoidEnv, sr):
    nBin = len(sinusoidEnv)
    nFFT = (nBin - 1) * 2
    nyq = sr / 2
    nHarmonic = len(hFreq)

    sinusoidEnv = np.exp(sinusoidEnv)
    sinusoidEnv[0] = 0.0
    harmonic = hFreq * f0 * np.arange(1, nHarmonic + 1)
    harmonic[harmonic >= nyq] = 0.0

    return ipl.interp1d(np.linspace(0.0, nyq, nBin), sinusoidEnv, kind = 'linear')(harmonic)

class Processor:
    def __init__(self, sr):
        self.samprate = float(sr)

    def __call__(self, f0List, hFreqList, sinusoidEnvList):
        hAmpList = np.zeros(hFreqList.shape)
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            hAmpList[iHop] = transformPitch(f0, hFreqList[iHop], sinusoidEnvList[iHop], self.samprate)
        return hAmpList
