import numpy as np
import scipy.signal as sp

from .common import *
from . import adaptivestft

def findPeak(magn, lowerIdx, upperIdx):
    nBin = len(magn)
    assert(lowerIdx >= 0 and lowerIdx < nBin)
    assert(upperIdx >= 0 and upperIdx < nBin)

    if(lowerIdx >= upperIdx):
        return lowerIdx
    rcmp = magn[lowerIdx + 1:upperIdx - 1]
    iPeaks = np.arange(lowerIdx + 1, upperIdx - 1)[np.logical_and(np.greater(rcmp, magn[lowerIdx:upperIdx - 2]), np.greater(rcmp, magn[lowerIdx + 2:upperIdx]))]
    if(len(iPeaks) == 0):
        return lowerIdx + np.argmax(magn[lowerIdx:upperIdx])
    else:
        return iPeaks[np.argmax(magn[iPeaks])]

class Processor:
    def __init__(self, mvf, sr, **kwargs):
        self.samprate = float(sr)
        self.window = kwargs.get("window", "blackman")
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))

        self.mvf = float(mvf)
        self.maxAvgHar = kwargs.get("maxAvgHar", 8)
        self.maxHarmonicOffset = kwargs.get("maxHarmonicOffset", 0.125)

        assert(self.mvf <= self.samprate / 2)

    def __call__(self, x, f0List, maxHar):
        # constant
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)

        # check input
        assert(nHop == len(f0List))

        # do stft
        stftProc = adaptivestft.Processor(self.samprate, window = self.window, hopSize = self.hopSize, fftSize = self.fftSize)
        magnList, phaseList, f0List = stftProc(x, f0List, refineF0 = True)
        magnList.T[0] = 0.0
        magnList = np.log(np.clip(magnList, 1e-8, np.inf))

        # find quasi-harmonic
        hFreqList = np.zeros((nHop, maxHar), dtype = np.float64)
        hAmpList = np.zeros((nHop, maxHar), dtype = np.float64)
        hPhaseList = np.zeros((nHop, maxHar), dtype = np.float64)
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            f0List[iHop], hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop] = self._findHarmonic(f0, magnList[iHop], phaseList[iHop], maxHar)
        hAmpList = np.exp(hAmpList)

        return f0List, hFreqList, hAmpList, hPhaseList

    def _findHarmonic(self, f0, magn, phase, maxHar):
        nBin = len(magn)

        assert((nBin - 1) * 2 == self.fftSize)
        assert(nBin == len(phase))
        assert(f0 > 0.0)

        outFreq = np.zeros(maxHar, dtype = np.float64)
        outAmp = np.full(maxHar, -np.inf, dtype = np.float64)
        outPhase = np.zeros(maxHar, dtype = np.float64)

        oldF0 = 0.0
        i = -1
        while(oldF0 != f0 and i < 16):
            i += 1
            nHar = min(maxHar, int(self.mvf / f0))
            nAvgHar = min(self.maxAvgHar, nHar)
            offset = f0 * self.maxHarmonicOffset
            for iHar in range(1, nAvgHar + 1):
                freq = iHar * f0
                if(freq >= self.mvf):
                    break
                lowerIdx = max(0, int(np.floor((freq - offset) / self.samprate * self.fftSize)))
                upperIdx = min(nBin - 1, int(np.ceil((freq + offset) / self.samprate * self.fftSize)))
                peakBin = findPeak(magn, lowerIdx, upperIdx)
                ipledPeakBin, ipledPeakAmp = parabolicInterpolation(magn, peakBin)
                outFreq[iHar - 1] = ipledPeakBin * self.samprate / self.fftSize
                outAmp[iHar - 1] = ipledPeakAmp
                outPhase[iHar - 1] = lerp(phase[peakBin], phase[peakBin + 1], np.mod(ipledPeakBin, 1.0))
            oldF0, f0 = f0, np.mean(outFreq[:nAvgHar] / np.arange(1, nAvgHar + 1))
        for iHar in range(nAvgHar + 1, nHar):
            freq = iHar * f0
            if(freq >= self.mvf):
                break
            lowerIdx = max(0, int(np.floor((freq - offset) / self.samprate * self.fftSize)))
            upperIdx = min(nBin - 1, int(np.ceil((freq + offset) / self.samprate * self.fftSize)))
            peakBin = findPeak(magn, lowerIdx, upperIdx)
            ipledPeakBin, ipledPeakAmp = parabolicInterpolation(magn, peakBin)
            outFreq[iHar - 1] = ipledPeakBin * self.samprate / self.fftSize
            outAmp[iHar - 1] = ipledPeakAmp
            outPhase[iHar - 1] = lerp(phase[peakBin], phase[peakBin + 1], np.mod(ipledPeakBin, 1.0))
        return outFreq[0], outFreq, outAmp, outPhase
