import numpy as np

from .common import *
from . import yin

def normalized_pdf(a, b, begin, end, number):
    x = np.arange(0, number, dtype = np.float64) * ((end - begin) / number)
    v = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    for i in range(2, len(v) + 1):
        i = len(v) - i
        if(v[i] < v[i + 1]):
            v[i] = v[i + 1]
    return v / np.sum(v)

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))

        self.minFreq = kwargs.get("minFreq", 80.0)
        self.maxFreq = kwargs.get("maxFreq", 1000.0)
        self.maxIter = 4
        self.prefilter = kwargs.get("prefilter", True)

        self.valleyThreshold = kwargs.get("valleyThreshold", 1.0)
        self.valleyStep = kwargs.get("valleyStep", 0.01)

        self.probThreshold = kwargs.get("probThreshold", 0.02)
        self.weightPrior = kwargs.get("weightPrior", 5.0)
        self.bias = kwargs.get("bias", 1.0)

        self.pdf = kwargs.get("pdf", normalized_pdf(1.7, 6.8, 0.0, 1.0, 128))

    def extractF0(self, obsProbList):
        nHop = len(obsProbList)

        out = np.zeros(nHop, dtype = np.float64)
        for iHop, (freqProb) in enumerate(obsProbList):
            if(len(freqProb) > 0):
                out[iHop] = freqProb.T[0][np.argmax(freqProb.T[1])]

        return out

    def __call__(self, x, removeDC = True):
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        pdfSize = len(self.pdf)

        if(removeDC):
            x = simpleDCRemove(x)
        if(self.prefilter):
            x = yin.doPrefilter(x, self.maxFreq, self.samprate)

        out = []
        for iHop in range(nHop):
            windowSize = 0
            minFreq = self.minFreq
            newWindowSize = max(roundUpToPowerOf2(self.samprate / minFreq * 2), self.hopSize * 2)
            iIter = 0
            while(newWindowSize != windowSize and iIter < self.maxIter):
                windowSize = newWindowSize
                frame = getFrame(x, iHop * self.hopSize, windowSize)
                if(removeDC):
                    frame = simpleDCRemove(frame)

                buff = yin.difference(frame)
                buff = yin.cumulativeDifference(buff)
                valleyIndexList = yin.findValleys(buff, minFreq, self.maxFreq, self.samprate, threshold = self.valleyThreshold, step = self.valleyStep)
                nValley = len(valleyIndexList)
                if(valleyIndexList):
                    possibleFreq = max(self.samprate / valleyIndexList[-1] - 20.0, self.minFreq)
                    newWindowSize = max(int(np.ceil(self.samprate / possibleFreq * 2)), self.hopSize * 4)
                    if(newWindowSize % 2 == 1):
                        newWindowSize += 1
                    iIter += 1

            freqProb = np.zeros((nValley, 2), dtype = np.float64)
            probTotal = 0.0
            weightedProbTotal = 0.0
            for iValley, valley in enumerate(valleyIndexList):
                ipledIdx, ipledVal = parabolicInterpolation(buff, valley)
                freq = self.samprate / ipledIdx
                v0 = 1 if(iValley == 0) else min(1.0, buff[valleyIndexList[iValley - 1]] + 1e-10)
                v1 = 0 if(iValley == nValley - 1) else max(0.0, buff[valleyIndexList[iValley + 1]]) + 1e-10
                prob = 0.0
                for i in range(int(v1 * pdfSize), int(v0 * pdfSize)):
                    prob += self.pdf[i] * (1.0 if(ipledVal < i / pdfSize) else 0.01)
                prob = min(prob, 0.99)
                prob *= self.bias
                probTotal += prob
                if(ipledVal < self.probThreshold):
                    prob *= self.weightPrior
                weightedProbTotal += prob
                freqProb[iValley] = freq, prob

            # renormalize
            if(nValley > 0 and weightedProbTotal != 0.0):
                freqProb.T[1] *= probTotal / weightedProbTotal

            out.append(freqProb)

        return out
