import numpy as np
import scipy.signal as sp
import pylab as pl

from .common import *
from . import adaptivestft

def calcCheapTrick(x, f0, sr, order):
    nX = len(x)
    nFFT = (nX - 1) * 2

    x = x.copy()
    iF0 = int(round(f0 / sr * nFFT))
    x[:iF0 // 2] = x[iF0 - iF0 // 2:iF0][::-1]
    smoothed = np.log(np.clip(mavg(x, order), 1e-6, np.inf))

    c = np.fft.irfft(smoothed)
    a = np.arange(1, nX) * (f0 / sr * np.pi)
    c[1:nX] *= np.sin(a) / a * (1.18 - 2.0 * 0.09 * np.cos(np.arange(1, nX) * (2.0 * np.pi * f0 / sr)))
    c[nX:] = c[1:nX - 1][::-1]
    smoothed = np.fft.rfft(c).real

    return smoothed

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.window = kwargs.get("window", "blackman")

        self.fixedF0 = 240.0
        self.orderFac = kwargs.get("orderFac", 1.0)

    def __call__(self, x, f0List):
        # constant
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        nBin = self.fftSize // 2 + 1
        B = getWindow(self.window)[1]
        avgF0 = np.mean(f0List[f0List > 0.0])

        # check input
        assert(nHop == len(f0List))
        assert(self.fftSize % 2 == 0)
        assert(x.ndim == 1)

        # analyze magnitude
        stftProc = adaptivestft.Processor(self.samprate, window = self.window, fftSize = self.fftSize, hopSize = self.hopSize)
        magnList, _ = stftProc(x, f0List)

        # apply true envelope
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                f0 = self.fixedF0
            order = int(np.ceil(f0 / self.samprate * self.fftSize / 3.0) * self.orderFac)
            magnList[iHop] = calcCheapTrick(magnList[iHop], f0, self.samprate, order)

        return magnList
