import numpy as np
import scipy.signal as sp
import scipy.optimize as so
from .common import *

def spectrum(x, F, bw, amp, sr):
    C = -np.exp(2.0 * np.pi * bw / sr)
    B = -2 * np.exp(np.pi * bw / sr)
    A = 1 - B - C
    z = np.exp(2.0j * np.pi * (0.5 + (x - F) / sr))
    ampFac = (1.0 + B - C) / A * amp
    return np.abs(A / (1.0 - B / z - C / z / z)) * ampFac

def spectrumFromFilterList(x, FList, bwList, ampList, sr):
    nFilter = len(FList)
    assert(nFilter == len(bwList))
    assert(nFilter == len(ampList))
    o = np.zeros(x.shape)
    for iFilter in range(nFilter):
        o += spectrum(x, FList[iFilter], bwList[iFilter], ampList[iFilter], sr)
    return o

class ParameterOptimizer:
    def __init__(self, sr):
        self.samprate = float(sr)
        self.nPreOptimizeIter = 4
        self.nOptimizeIter = 16
        self.responseRadius = 1000.0
        self.minF = 100.0
        self.maxF = 6000.0
        self.minBw = 80.0
        self.maxBw = 800.0
        self.maxDeltaF = 36.0
        self.integralSample = 128

    def optimizeF(self, x, y, FList, ampList, bwList, rate = 1.0):
        nFilter = len(FList)
        nyq = self.samprate / 2

        newFList = FList.copy()
        estY = spectrumFromFilterList(x, FList, bwList, ampList, self.samprate)
        cost = np.log(np.clip(estY, 1e-6, np.inf)) - np.log(np.clip(y, 1e-6, np.inf))

        integralX = np.linspace(0.0, 1.0, self.integralSample)
        iplX = np.concatenate(((0.0, ), x, (nyq, )))
        iplY = np.concatenate(((cost[0], ), cost, (cost[-1], )))
        integralSampler = ipl.interp1d(iplX, iplY, kind = 'linear')
        del iplX, iplY
        for iFilter in range(nFilter):
            F = FList[iFilter]
            bw = bwList[iFilter]
            amp = ampList[iFilter]

            leftBound = max(0.0, F - self.responseRadius)
            if(iFilter == nFilter - 1):
                rightBound = min(nyq, F + min(self.responseRadius, bw))
            else:
                rightBound = min(nyq, F + self.responseRadius)
            leftMean = np.mean(integralSampler(F + (leftBound - F) * integralX))
            rightMean = np.mean(integralSampler(F + (rightBound - F) * integralX))
            sym = -1 if(leftMean < rightMean) else 1
            deltaF = min(self.maxDeltaF, np.abs(leftMean - rightMean) * (16.0 * rate))
            newFList[iFilter] = min(self.maxF, max(self.minF, F + sym * deltaF))

        return newFList

    def optimizeAmp(self, x, y, FList, ampList, bwList, rate = 1.0):
        nFilter = len(FList)
        nyq = self.samprate / 2

        newAmpList = ampList.copy()
        estY = spectrumFromFilterList(x, FList, bwList, ampList, self.samprate)
        cost = estY - y

        integralX = np.linspace(0.0, 1.0, self.integralSample)
        iplX = np.concatenate(((0.0, ), x, (nyq, )))
        iplY = np.concatenate(((cost[0], ), cost, (cost[-1], )))
        integralSampler = ipl.interp1d(iplX, iplY, kind = 'linear')
        del iplX, iplY
        for iFilter in range(nFilter):
            F = FList[iFilter]
            bw = bwList[iFilter]
            amp = ampList[iFilter]

            leftBound = max(0.0, F - bw * 2)
            if(iFilter == nFilter - 1):
                rightBound = min(nyq, F + bw)
            else:
                rightBound = min(nyq, F + bw * 2)
            mean = np.mean(integralSampler(leftBound + (rightBound - leftBound) * integralX))
            sym = 1 if(mean < 0) else -1
            newAmpList[iFilter] = min(1.0, max(1e-4, amp + sym * np.abs(mean) * rate * 2))

        return newAmpList

    def optimizeBw(self, x, y, FList, ampList, bwList, rate = 1.0):
        nFilter = len(FList)
        nyq = self.samprate / 2

        newBwList = bwList.copy()
        estY = spectrumFromFilterList(x, FList, bwList, ampList, self.samprate)
        cost = estY - y

        integralX = np.linspace(0.0, 1.0, self.integralSample)
        iplX = np.concatenate(((0.0, ), x, (nyq, )))
        iplY = np.concatenate(((cost[0], ), cost, (cost[-1], )))
        integralSampler = ipl.interp1d(iplX, iplY, kind = 'linear')
        del iplX, iplY
        for iFilter in range(nFilter):
            F = FList[iFilter]
            bw = bwList[iFilter]
            amp = ampList[iFilter]

            leftBound = max(0.0, F - bw * 4)
            if(iFilter == nFilter - 1):
                rightBound = min(nyq, F + bw * 0.05)
            else:
                rightBound = min(nyq, F + bw * 4)
            mean = np.mean(integralSampler(leftBound + (rightBound - leftBound) * integralX))
            sym = 1 if(mean < 0) else -1
            newBwList[iFilter] = min(self.maxBw, max(self.minBw, bw + sym * np.sqrt(np.abs(mean)) * 1000.0 * rate))

        return newBwList

    @staticmethod
    def defaultPreOptimizeReferenceGetter(envelope, fftSize, sr, fac = 1.25):
        iplX = np.arange(len(envelope)) / fftSize * sr
        return ipl.interp1d(iplX, envelope * fac, kind = 'linear')

    def optimizeSingleFrame(self, x, y, initFList, initBwList, initAmpList, rate = 1.0, preOptimizeReferenceGetter = None):
        origOrder = np.argsort(initFList)
        FList = initFList[origOrder]
        bwList = initBwList[origOrder]
        ampList = initAmpList[origOrder]

        # preOptimize F
        for iIter in range(self.nPreOptimizeIter):
            FList = self.optimizeF(x, y, FList, ampList, bwList, rate = rate * 2.0)
            if(preOptimizeReferenceGetter is not None):
                ampList = preOptimizeReferenceGetter(FList)

        if(self.nOptimizeIter is not None):
            for iIter in range(self.nOptimizeIter):
                ampList = self.optimizeAmp(x, y, FList, ampList, bwList, rate = rate)
                bwList = self.optimizeBw(x, y, FList, ampList, bwList, rate = rate)
                FList = self.optimizeF(x, y, FList, ampList, bwList, rate = rate)
        else:
            for iIter in range(1024):
                newAmpList = self.optimizeAmp(x, y, FList, ampList, bwList, rate = rate)
                newBwList = self.optimizeBw(x, y, FList, ampList, bwList, rate = rate)
                newFList = self.optimizeF(x, y, FList, ampList, bwList, rate = rate)

                maxDeltaAmp = np.max(np.abs(newAmpList - ampList))
                maxDeltaBw = np.max(np.abs(newBwList - bwList))
                maxDeltaF = np.max(np.abs(newFList - FList))

                FList, bwList, ampList = newFList, newBwList, newAmpList
                if(maxDeltaAmp < 2e-3 and maxDeltaBw < 1.0 and maxDeltaF < 3.6):
                    break

        order = np.argsort(FList)
        FList = FList[order]
        bwList = bwList[order]
        ampList = ampList[order]

        return FList[origOrder], bwList[origOrder], ampList[origOrder]

    def optimizeSingleFrameAutoJitter(self, x, y, initFList, initBwList, preOptimizeReferenceGetter, rate = 1.0, nJitter = 32):
        nFilter = len(initFList)

        cost = []
        param = []

        ampList = preOptimizeReferenceGetter(initFList)
        FList, bwList, ampList = self.optimizeSingleFrame(x, y, initFList, initBwList, ampList, rate = rate, preOptimizeReferenceGetter = preOptimizeReferenceGetter)
        param.append((FList, bwList, ampList))
        estY = spectrumFromFilterList(x, FList, bwList, ampList, self.samprate)
        delta = np.log(np.clip(estY, 1e-6, np.inf)) - np.log(np.clip(y, 1e-6, np.inf))
        cost.append(np.sum(np.abs(delta)))

        for iJitter in range(nJitter):
            FList = np.clip(initFList * np.random.uniform(0.8, 1.2, nFilter), self.minF, self.maxF)
            ampList = preOptimizeReferenceGetter(initFList)
            FList, bwList, ampList = self.optimizeSingleFrame(x, y, FList, initBwList, ampList, rate = rate, preOptimizeReferenceGetter = preOptimizeReferenceGetter)
            param.append((FList, bwList, ampList))
            estY = spectrumFromFilterList(x, FList, bwList, ampList, self.samprate)
            delta = np.log(np.clip(estY, 1e-6, np.inf)) - np.log(np.clip(y, 1e-6, np.inf))
            cost.append(np.sum(np.abs(delta)))

        return param[np.argmin(np.abs(cost))]

    def __call__(self, harmonicList, hAmpList, envList, FList = None, bwList = None, ampList = None, nFormant = None, nJitter = 0, rate = 1.0, preEmphasisFreq = 50.0):
        nHop = envList.shape[0]
        fftSize = (envList.shape[1] - 1) * 2
        assert(nJitter >= 0)

        envList = np.exp(envList)

        if(nFormant is not None):
            assert(FList is None)
        if(FList is None):
            assert(nFormant is not None)
            FList = np.full((nHop, nFormant), formantFreq(np.arange(1, nFormant + 1), L = 0.16))
        else:
            FList = FList.copy()
        nFormant = FList.shape[1]
        if(bwList is None):
            bwList = np.full((nHop, nFormant), 300.0)
        else:
            bwList = bwList.copy()
        if(ampList is None):
            ampList = np.zeros((nHop, nFormant))
            if(nJitter == 0):
                for iHop in range(nHop):
                    porg = self.defaultPreOptimizeReferenceGetter(envList[iHop], fftSize, self.samprate)
                    ampList[iHop] = porg(FList[iHop])
        else:
            ampList = ampList.copy()

        peEnv = preEmphasisResponse(np.arange(envList.shape[1]) / fftSize * self.samprate, 50.0, self.samprate)
        for iHop in range(nHop):
            print(iHop)
            if(harmonicList[iHop][0] == 0.0):
                continue
            porg = self.defaultPreOptimizeReferenceGetter(envList[iHop] * peEnv, fftSize, self.samprate)
            need = harmonicList[iHop] > 0.0
            harmonics = harmonicList[iHop][need]
            hAmps = hAmpList[iHop][need]
            hAmps *= preEmphasisResponse(harmonics, 50.0, self.samprate)
            if(nJitter > 0):
                FList[iHop], bwList[iHop], ampList[iHop] = self.optimizeSingleFrameAutoJitter(harmonics, hAmps, FList[iHop], bwList[iHop], porg, nJitter = nJitter, rate = rate)
            else:
                FList[iHop], bwList[iHop], ampList[iHop] = self.optimizeSingleFrame(harmonics, hAmps, FList[iHop], bwList[iHop], ampList[iHop], rate = rate, preOptimizeReferenceGetter = porg)
            ampList[iHop] /= preEmphasisResponse(FList[iHop], 50.0, self.samprate)
        return FList, bwList, ampList
