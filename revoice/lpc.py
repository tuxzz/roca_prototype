import numpy as np
import scipy.signal as sp
import scipy.linalg as sla

from .common import *

def toFormant(coeff, sr, minFreq = 100.0, maxFreq = 7500.0):
    nHop, order = coeff.shape
    nMaxFormant = int(np.ceil(order / 2))
    nyq = sr / 2
    maxFreq = min(maxFreq, nyq - 50.0)

    assert(maxFreq > minFreq)

    FList = np.zeros((nHop, nMaxFormant))
    bwList = np.zeros((nHop, nMaxFormant))

    for iHop in range(nHop):
        polyCoeff = np.zeros(order + 1)
        polyCoeff[:order] = coeff[iHop][::-1]
        polyCoeff[-1] = 1.0
        roots = np.roots(polyCoeff)
        roots = roots[roots.imag >= 0.0] # remove conjugate roots
        roots = fixIntoUnit(roots)

        hopF = []
        hopBw = []
        for iRoot in range(len(roots)):
            F = np.abs(np.arctan2(roots.imag[iRoot], roots.real[iRoot])) * nyq / np.pi
            if(F >= minFreq and F < maxFreq):
                bw = -np.log(np.abs(roots[iRoot]) ** 2) * nyq / np.pi
                hopF.append(F)
                hopBw.append(bw)

        hopF = np.asarray(hopF)
        hopBw = np.asarray(hopBw)
        sortOrder = np.argsort(hopF)
        nFormant = min(nMaxFormant, len(hopF))
        FList[iHop][:nFormant] = hopF[sortOrder][:nFormant]
        bwList[iHop][:nFormant] = hopBw[sortOrder][:nFormant]

    return FList, bwList

def toSpectrum(coeff, xms, deEmphasisFreq, fftSize, sr, bandwidthReduction = None):
    nHop = len(coeff)
    assert(nHop == len(xms))

    out = np.zeros((nHop, fftSize // 2 + 1))
    for iHop in range(nHop):
        out[iHop] = toSpectrumSingleFrame(coeff[iHop], xms[iHop], fftSize, sr, deEmphasisFreq, bandwidthReduction)
    return out

def toSpectrumSingleFrame(coeff, xms, fftSize, sr, deEmphasisFreq = None, bandwidthReduction = None):
    nyq = sr / 2
    nData = len(coeff) + 1
    scale = 1.0 / np.sqrt(2.0 * nyq * nyq / (fftSize / 2.0))

    # copy to buffer
    fftBuffer = np.zeros(fftSize)
    fftBuffer[0] = 1.0
    fftBuffer[1:nData] = coeff

    # deemphasis
    if(deEmphasisFreq is not None):
        fac = np.exp(-2 * np.pi * deEmphasisFreq / nyq)
        nData += 1
        for i in reversed(range(1, nData)):
            fftBuffer[i] -= fac * fftBuffer[i - 1]

    # reduce bandwidth
    if(bandwidthReduction is not None):
        fac = np.exp(np.pi * bandwidthReduction / sr)
        fftBuffer[1:nData] *= np.power(fac, np.arange(2, nData + 1))

    # do fft
    if(xms > 0.0):
        scale *= np.sqrt(xms)
    o = np.fft.rfft(fftBuffer)
    o.real[0] = scale / o.real[0]
    o.imag[0] = 0.0
    o[1:fftSize // 2] = np.conj(o[1:fftSize // 2] * scale / (np.abs(o[1:fftSize // 2]) ** 2))
    o.real[-1] = scale / o.real[-1]
    o.imag[-1] = 0.0

    return np.abs(o)

class LPC:
    def __init__(self, sr, **kwargs):
        self.preEmphasisFreq = kwargs.get("preEmphasisFreq", 50.0)
        self.samprate = float(sr)
        self.resampleRatio = kwargs.get("resampleRatio", 1.0)
        self.window = kwargs.get("window", "blackman")
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))

    def procSingleFrame(self, x):
        raise NotImplementedError("This LPC Object is not implemented.")

    def __call__(self, x, f0List, order, unvoiced = True):
        nX = len(x)
        nHop = len(f0List)

        resampledNX = int(round(nX * self.resampleRatio))
        if(resampledNX != nX):
            resampleRatio = resampledNX / nX
            sr = self.samprate * resampleRatio
            x = sp.resample(x, resampledNX)
        else:
            sr = self.samprate
            resampleRatio = 1.0
        del resampledNX
        if(self.preEmphasisFreq is not None):
            x = preEmphasis(x, self.preEmphasisFreq, sr)

        coeff = np.zeros((nHop, order))
        xms = np.zeros(nHop)
        windowFunc, B, _ = getWindow(self.window)
        for iHop, f0 in enumerate(f0List):
            if(not unvoiced and f0 <= 0.0):
                continue
            windowSize = int(np.ceil(sr / f0 * B * 2.0)) if(f0 > 0.0) else self.hopSize * 2
            if(windowSize % 2 == 0):
                windowSize += 1
            window = windowFunc(windowSize)
            iCenter = int(round(self.hopSize * iHop * resampleRatio))
            frame = getFrame(x, iCenter, windowSize) * window
            if(np.sum(np.abs(frame)) < 1e-10):
                continue
            coeff[iHop], xms[iHop] = self.procSingleFrame(frame, order)

        return coeff, xms

class Burg(LPC):
    def __init__(self, sr, **kwargs):
        super().__init__(sr, **kwargs)

    def procSingleFrame(self, x, order):
        n = len(x)
        m = order

        a = np.ones(m)
        aa = np.ones(m)
        b1 = np.ones(n)
        b2 = np.ones(n)

        # (3)
        xms = np.sum(x * x) / n

        if(xms <= 0.0):
            raise ValueError("Empty/Zero input.")
            return np.zeros(m), 0.0

        # (9)
        b1[0] = x[0]
        b2[n - 2] = x[n - 1]
        b1[1:n - 1] = b2[:n - 2] = x[1:n - 1]

        for i in range(m):
            # (7)
            numer = np.sum(b1[:n - i - 1] * b2[:n - i - 1])
            deno = np.sum((b1[:n - i - 1] ** 2) + (b2[:n - i - 1] ** 2))
            if(deno <= 0):
                raise ValueError("Bad denominator(order is too large/x is too short?).")

            a[i] = 2.0 * numer / deno

            # (10)
            xms *= 1.0 - a[i] * a[i]

            # (5)
            a[:i] = aa[:i] - a[i] * aa[:i][::-1]

            if(i < m - 1):
                # (8)
                # NOTE: i -> i + 1
                aa[:i + 1] = a[:i + 1]

                for j in range(n - i - 2):
                    b1[j] -= aa[i] * b2[j]
                    b2[j] = b2[j + 1] - aa[i] * b1[j + 1]

        return -a, np.sqrt(xms * n)

class Autocorrelation(LPC):
    def __init__(self, sr, **kwargs):
        super().__init__(sr, **kwargs)
        self.useFastSolver = kwargs.get("useFastSolver", True)

    def procSingleFrame(self, x, order):
        if(self.useFastSolver):
            return self.__fastSolve__(x, order)
        else:
            return self.__slowSolve__(x, order)

    @staticmethod
    def __slowSolve__(x, m):
        n = len(x)
        p = m + 1
        r = np.zeros(p)
        nx = np.min((p, n))
        x = np.correlate(x, x, 'full')
        r[:nx] = x[n - 1:n+m]
        a = np.dot(sla.pinv2(sla.toeplitz(r[:-1])), -r[1:])
        gain = np.sqrt(r[0] + np.sum(a * r[1:]))
        return a, gain

    @staticmethod
    def __fastSolve__(x, m):
        n = len(x)
        # do autocorrelate via FFT
        nFFT = roundUpToPowerOf2(2 * n - 1)
        nx = np.min((m + 1, n))
        r = np.fft.irfft(np.abs(np.fft.rfft(x, n = nFFT) ** 2))
        r = r[:nx] / n
        a, e, k = Autocorrelation.levinson_1d(r, m)
        gain = np.sqrt(np.sum(a * r * n))

        return a[1:], gain

    @staticmethod
    def levinson_1d(r, order):
        r = np.atleast_1d(r)
        if r.ndim > 1:
            raise ValueError("Only rank 1 are supported for now.")
        n = r.size
        if n < 1:
            raise ValueError("Cannot operate on empty array !")
        elif order > n - 1:
            raise ValueError("Order should be <= size-1")
        if not np.isreal(r[0]):
            raise ValueError("First item of input must be real.")
        elif not np.isfinite(1/r[0]):
            raise ValueError("First item should be != 0")
        # Estimated coefficients
        a = np.empty(order + 1, r.dtype)
        # temporary array
        t = np.empty(order + 1, r.dtype)
        # Reflection coefficients
        k = np.empty(order, r.dtype)

        a[0] = 1.0
        e = r[0]

        for i in range(1, order+1):
            acc = r[i]
            for j in range(1, i):
                acc += a[j] * r[i-j]
            k[i-1] = -acc / e
            a[i] = k[i-1]
            for j in range(order):
                t[j] = a[j]
            for j in range(1, i):
                a[j] += k[i-1] * np.conj(t[i-j])
            e *= 1 - k[i-1] * np.conj(k[i-1])

        return a, e, k
