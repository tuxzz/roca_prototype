import numpy as np
import scipy.signal as sp

from .common import *
from . import adaptivestft

def calcTrueEnvelope(x, order, nIter = 24, maxStep = 1.5):
    nX = len(x)

    assert(nX > order * 2)

    # initialize the iteration using A0(k) = log(|X(k)|)
    a = x.copy().astype(np.complex128)

    # prepare iter
    lastC = np.fft.irfft(a)
    lastC[order:-order] = 0.0
    v = np.fft.rfft(lastC)

    less = a.real < v.real
    a.real[less] = v.real[less]
    lastC = np.fft.irfft(a)
    lastC[order:-order] = 0.0
    v = np.fft.rfft(lastC)

    for iIter in range(nIter):
        step = np.power(maxStep, (nIter - iIter) / nIter)
        less = a.real < v.real
        a.real[less] = v.real[less]
        c = np.fft.irfft(a)
        lastC[:order] = c[:order] + (c[:order] - lastC[:order]) * step
        lastC[-order:] = c[-order:] + (c[-order:] - lastC[-order:]) * step
        lastC[order:-order] = 0.0
        v = np.fft.rfft(lastC)
    return v.real

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.window = kwargs.get("window", "blackman")

        self.orderFac = kwargs.get("orderFac", 1.0)
        self.nIter = kwargs.get("nIter", 24)
        self.maxStep = kwargs.get("maxStep", 1.5)

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
            if(f0 > 0.0):
                order = int(self.samprate / (2 * B * f0) * self.orderFac)
            else:
                order = int(self.samprate / (2 * B * avgF0) * self.orderFac)
            magnList[iHop] = calcTrueEnvelope(np.log(np.clip(magnList[iHop], 1e-6, np.inf)), order, nIter = self.nIter, maxStep = self.maxStep)

        return magnList
