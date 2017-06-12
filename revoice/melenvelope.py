import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl

from .common import *
from . import adaptivestft

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.window = getWindow(kwargs.get("window", "blackman"))
        self.firstFilter = kwargs.get("firstFilter", 0.0) # in mel
        self.filterDistance = kwargs.get("filterDistance", 100.0) # in mel

    def __call__(self, x, f0List):
        # constant
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        nBin = self.fftSize // 2 + 1
        nyq = self.samprate / 2
        maxMel = freqToMel(nyq)
        nFilter = int(np.ceil((maxMel - self.firstFilter) / self.filterDistance)) + 1
        fRange = np.fft.rfftfreq(self.fftSize, 1.0 / self.samprate)

        # check input
        assert(nHop == len(f0List))
        assert(self.fftSize % 2 == 0)
        assert(x.ndim == 1)

        # build filter bank
        filterBank = np.zeros((nFilter, nBin), dtype = np.float64)
        for iFilter in range(nFilter):
            peakMel = self.firstFilter + iFilter * self.filterDistance
            peakFreq = melToFreq(peakMel)
            leftFreq = melToFreq(peakMel - self.filterDistance)
            rightFreq = melToFreq(peakMel + self.filterDistance)
            iplX = (leftFreq, peakFreq, rightFreq)
            iplY = (0.0, 1.0, 0.0)
            filterBank[iFilter] = ipl.interp1d(iplX, iplY, kind = 'linear', bounds_error = False, fill_value = 0.0)(fRange)
            bankSum = np.sum(filterBank[iFilter])
            if(bankSum > 0.0):
                filterBank[iFilter] /= bankSum

        # analyze magnitude
        stftProc = adaptivestft.Processor(self.samprate, window = self.window, fftSize = self.fftSize, hopSize = self.hopSize)
        magnList, _ = stftProc(x, f0List)

        # gen frequency table
        melX = np.zeros(nFilter, dtype = np.float64)
        for iFilter in range(nFilter):
            melX[iFilter] = iFilter * self.filterDistance + self.firstFilter
        linearX = melToFreq(melX)

        # apply filter and sum, then do sinc interpolation
        melSpectrum = np.zeros(nFilter, dtype = np.float64)
        for iHop in range(nHop):
            for iFilter in range(nFilter):
                melSpectrum[iFilter] = np.sum(magnList[iHop] * filterBank[iFilter])
            melSpectrum = np.log(np.clip(melSpectrum, 1e-10, np.inf))
            magnList[iHop] = ipl.interp1d(linearX, melSpectrum, kind = 'cubic')(fRange)
        return magnList
