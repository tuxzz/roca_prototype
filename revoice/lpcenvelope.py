import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl

from .common import *
from . import lpc

class Processor:
    def __init__(self, sr, **kwargs):
        defaultOrder = int(np.ceil(sr / 16000 * 13))
        if(defaultOrder % 2 == 0):
            defaultOrder += 1
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.order = kwargs.get("order", defaultOrder)
        self.window = getWindow(kwargs.get("window", "blackman"))
        self.method = kwargs.get("method", "ac")
        assert(self.method in ("burg", "ac"))

    def __call__(self, x, f0List):
        if(self.method == "burg"):
            lpcProc = lpc.Burg(self.samprate)
        elif(self.method == "ac"):
            lpcProc = lpc.Autocorrelation(self.samprate)
        else:
            assert(False)
        coeff, xms = lpcProc(x, f0List, self.order)
        lpcSpectrum = lpc.toSpectrum(coeff, xms, lpcProc.preEmphasisFreq, self.fftSize, self.samprate)
        return np.log(np.clip(lpcSpectrum, 1e-8, np.inf))
