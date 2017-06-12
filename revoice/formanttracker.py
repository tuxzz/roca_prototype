import numpy as np
import scipy.stats as ss
import pylab as pl

from .common import *
from . import hmm

def parameterGenerate(nFormant, hopSize, sr):
    maxTransHerz = (hopSize / sr) / 0.0025 * 50.0
    return nFormant, maxTransHerz

class Processor:
    def __init__(self, nFormant, maxTransHerz, **kwargs):
        self.nFormant = nFormant
        self.minFreq = kwargs.get("minFreq", 100.0)
        self.maxFreq = kwargs.get("maxFreq", 7500.0)
        self.preEmphasisFreq = kwargs.get("preEmphasisFreq", 100.0)

        self.inputTrust = kwargs.get("inputTrust", 0.75)
        self.herzPerBin = kwargs.get("herzPerBin", 12.5)
        self.transSelf = kwargs.get("transSelf", 0.625)
        self.maxTransHerz = maxTransHerz
        self.L = kwargs.get("L", 0.168)
        self.transProbRadius = kwargs.get("radius", 300.0)
        self.preProbRadius = kwargs.get("radius", 600.0)

        self.model = self.createModel()

    def createModel(self):
        freqRange = self.maxFreq - self.minFreq
        nBin = int(np.ceil(freqRange / self.herzPerBin))
        halfMaxTransBin = int(round((self.maxTransHerz / self.herzPerBin) / 2))
        transSigma = self.transProbRadius / 3.0

        nTrans = 0
        for iFromState in range(nBin):
            begin = max(0, iFromState - halfMaxTransBin)
            end = min(nBin, iFromState + halfMaxTransBin + 1)
            for iToState in range(begin, end):
                nTrans += 1

        init = np.ndarray(nBin, dtype = np.float64)
        frm = np.zeros(nTrans, dtype = np.int)
        to = np.zeros(nTrans, dtype = np.int)
        transProb = np.zeros(nTrans, dtype = np.float64)

        init.fill(1.0 / nBin)
        iTrans = 0
        for iFromState in range(nBin):
            begin = max(0, iFromState - halfMaxTransBin)
            end = min(nBin, iFromState + halfMaxTransBin + 1)
            probSum = 0.0
            beginITrans = iTrans
            for iToState in range(begin, end):
                distanceHerz = (iToState - iFromState) * self.herzPerBin
                if(iToState == iFromState):
                    prob = ss.norm.pdf(distanceHerz, loc = 0.0, scale = transSigma) * self.transSelf
                else:
                    prob = ss.norm.pdf(distanceHerz, loc = 0.0, scale = transSigma) * (1.0 - self.transSelf)
                probSum += prob

                frm[iTrans] = iFromState
                to[iTrans] = iToState
                transProb[iTrans] = prob
                iTrans += 1
            transProb[beginITrans:iTrans] /= probSum

        return hmm.SparseHMM(init, frm, to, transProb)

    def calcStateProb(self, iFormant, hopF, theoreticalF, envGetter):
        hopF = np.asarray(hopF)
        nBin = len(self.model.init)
        preSigma = self.preProbRadius / 3

        binFreq = np.arange(nBin) * self.herzPerBin + self.minFreq
        binAmp = envGetter(binFreq)
        binAmp -= np.min(binAmp)

        if(iFormant < len(hopF)):
            offset = np.abs(theoreticalF - hopF[iFormant]) / 3
            probPdf = ss.norm.pdf(binFreq, loc = hopF[iFormant], scale = preSigma + offset)
            probPdf += (1.0 - self.inputTrust) * np.max(probPdf)
            binAmp *= probPdf
        binAmp /= np.sum(binAmp)

        return binAmp

    def __call__(self, FList, envList, sr):
        nHop = len(FList)
        nState = len(self.model.init)
        fftSize = (envList.shape[1] - 1) * 2

        out = np.zeros((nHop, self.nFormant))
        iplX = np.arange(envList.shape[1]) / fftSize * sr
        envList = np.exp(envList)
        if(self.preEmphasisFreq is None):
            pe = np.full(envList.shape[1], 1.0)
        else:
            pe = preEmphasisResponse(iplX, self.preEmphasisFreq, sr)
        for iFormant in range(self.nFormant):
            obsSeq = np.zeros((nHop, nState))
            for iHop in range(nHop):
                need = np.logical_and(FList[iHop] >= self.minFreq, FList[iHop] < self.maxFreq)
                hopF = FList[iHop][need]
                envGetter = ipl.interp1d(iplX, np.log(envList[iHop] * pe), kind = 'linear')
                obsSeq[iHop] = self.calcStateProb(iFormant, hopF, formantFreq(iFormant + 1, L = self.L), envGetter)
            path = self.model.viterbiDecode(obsSeq)
            out.T[iFormant] = path * self.herzPerBin + self.minFreq
            if(iFormant < FList.shape[1]):
                need = np.abs(out.T[iFormant] - FList.T[iFormant]) < 50.0
                out.T[iFormant][need] = FList.T[iFormant][need]

        return out
