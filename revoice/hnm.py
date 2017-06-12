import numpy as np
import scipy.signal as sp

from .common import *
from . import adaptivestft
from . import cheaptrick
from . import hnm_qfft
from . import hnm_qhm

def synthSinusoid(hFreq, hAmp, hPhase, r, sr):
    # constant
    nOut = len(r)
    nHar = len(hFreq)
    nyq = sr / 2

    # check input
    assert(nHar == len(hAmp))
    assert(nHar == len(hPhase))
    assert(hFreq.ndim == 1)
    assert(hAmp.ndim == 1)
    assert(hPhase.ndim == 1)
    assert(sr > 0.0)

    # compute
    out = np.zeros(nOut)
    for iHar in range(nHar):
        freq = hFreq[iHar]
        amp = hAmp[iHar]
        phase = hPhase[iHar] if(not hPhase is None) else 0.0
        if(freq <= 0.0 or freq >= nyq):
            break
        if(amp <= 0.0):
            continue
        out[:] += np.cos(2.0 * np.pi / sr * freq * r + phase) * amp
    return out

def filterNoise(x, noiseEnvList, hopSize):
    olaFac = 2
    windowFac = 4
    nHop, noiseEnvBin = noiseEnvList.shape
    windowSize = hopSize * windowFac
    nBin = windowSize // 2 + 1
    nX = len(x)

    assert(getNFrame(nX, hopSize) == nHop)
    assert(hopSize % olaFac == 0)

    window, windowMean = np.hanning(windowSize), 0.5
    analyzeNormFac = 0.5 * windowMean * windowSize
    synthNormScale = windowFac // 2 * olaFac

    window = np.sqrt(window)
    buff = np.zeros(nBin, dtype = np.complex128)
    out = np.zeros(nX)
    for iFrame in range(nHop * olaFac):
        iHop = iFrame // olaFac
        iCenter = iFrame * hopSize // olaFac
        frame = getFrame(x, iCenter, windowSize)
        if(np.max(frame) == np.min(frame)):
            continue

        ffted = np.fft.rfft(frame * window)
        phase = np.angle(ffted)

        env = ipl.interp1d(np.linspace(0, nBin, noiseEnvBin), noiseEnvList[iHop], kind = "linear")(np.arange(nBin))
        magn = np.exp(env) * analyzeNormFac

        buff.real = magn * np.cos(phase)
        buff.imag = magn * np.sin(phase)

        synthed = np.fft.irfft(buff) * window
        ob, oe, ib, ie = getFrameRange(nX, iCenter, windowSize)
        out[ib:ie] += synthed[ob:oe] / synthNormScale
    return out

class Analyzer:
    supoortedHarmonicAnalysisMethod = {
        "qfft": hnm_qfft.Processor,
        "qhmair": hnm_qhm.Processor,
    }

    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))

        self.mvf = kwargs.get("mvf", min(sr / 2 - 1e3, 20e3))
        self.harmonicAnalysisMethod = kwargs.get("harmonicAnalysisMethod", 'qfft')
        self.harmonicAnalysisParameter = kwargs.get("harmonicAnalysisParameter", {})
        self.noiseEnergyThreshold = kwargs.get("noiseEnergyThreshold", 1e-8)

        assert(self.mvf <= self.samprate / 2)

    def __call__(self, x, f0List):
        # constant
        nX = len(x)
        nBin = self.fftSize // 2 + 1
        nHop = getNFrame(nX, self.hopSize)
        minF0 = np.min(f0List[f0List > 0.0])
        maxHar = max(0, int(self.mvf / minF0))

        # check input
        assert(nHop == len(f0List))
        assert(f0List.ndim == 1)

        # dc adjustment
        x = simpleDCRemove(x)

        # (quasi)harmonic analysis
        harProc = self.supoortedHarmonicAnalysisMethod[self.harmonicAnalysisMethod](self.mvf, self.samprate, **self.harmonicAnalysisParameter, hopSize = self.hopSize)
        f0List, hFreqList, hAmpList, hPhaseList = harProc(x, f0List, maxHar)

        # resynth & ola & record sinusoid energy
        sinusoid = np.zeros(nX, dtype = np.float64)
        sinusoidEnergyList = np.zeros(nHop, dtype = np.float64)
        olaWindow = np.hanning(2 * self.hopSize)
        energyAnalysisWindowNormFac = 1.0 / np.sqrt(0.375)
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            energyAnalysisRadius = int(round(self.samprate / f0)) * 2
            synthLeft = max(energyAnalysisRadius, self.hopSize)
            synthRight = max(energyAnalysisRadius + 1, self.hopSize)
            synthRange = np.arange(-synthLeft, synthRight)

            ob, oe, ib, ie = getFrameRange(nX, iHop * self.hopSize, 2 * self.hopSize)
            synthed = synthSinusoid(hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop], synthRange, self.samprate)

            # integrate energy
            energyBegin = synthLeft - energyAnalysisRadius
            energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1)
            sinusoidEnergyList[iHop] = np.mean((synthed[energyBegin:energyBegin + energyAnalysisRadius * 2 + 1] * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)

            # ola
            olaBegin = synthLeft - self.hopSize
            sinusoid[ib:ie] += synthed[olaBegin + ob:olaBegin + oe] * olaWindow[ob:oe]

        noise = x - sinusoid # extract noise

        # build noise envelope
        envProc = cheaptrick.Processor(self.samprate, hopSize = self.hopSize, fftSize = self.fftSize)
        noiseEnvList = envProc(noise, f0List)

        # record noise energy
        energyAnalysisWindowNormFac = 1.0 / np.sqrt(0.375)
        noiseEnergyList = np.zeros(nHop, dtype = np.float64)
        for iHop, f0 in enumerate(f0List):
            if(f0 > 0.0):
                frame = getFrame(noise, iHop * self.hopSize, 4 * int(round(self.samprate / f0)) + 1)
            else:
                frame = getFrame(noise, iHop * self.hopSize, 2 * self.hopSize)
            energyAnalysisWindow = np.hanning(len(frame))
            noiseEnergyList[iHop] = np.mean((frame * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)
        noiseEnergyList[noiseEnergyList < self.noiseEnergyThreshold] = 0.0

        # relative phase shift
        need = f0List > 0.0
        f0Need = f0List[need]
        nNeed = len(f0Need)
        f0Need = f0Need.reshape(nNeed, 1)
        base = hPhaseList[need].T[0].reshape(nNeed, 1)
        hPhaseList[need] = wrap(hPhaseList[need] - (hFreqList[need] / f0Need) * base)

        # relative harmonic shift
        voiced = f0List > 0.0
        nVoiced = np.sum(voiced)
        hFreqList[voiced] /= f0List[voiced].reshape(nVoiced, 1) * np.arange(1, maxHar + 1)
        hFreqList[hFreqList <= 0.0] = 1.0

        # debug
        saveWav("sin.wav", sinusoid, self.samprate)
        saveWav("noise.wav", noise, self.samprate)

        return f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList

class Synther:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.olaFac = kwargs.get("olaFac", 2)

        self.mvf = kwargs.get("mvf", min(sr / 2 - 1e3, 20e3))
        self.maxNoiseEnvHarmonic = kwargs.get("maxNoiseEnvHarmonic", 4)
        self.maxNoiseEnvDCAdjustment = kwargs.get("maxNoiseEnvDCAdjustment", 10.0)

        assert(self.mvf <= self.samprate / 2)
        assert(self.hopSize % self.olaFac == 0)

    def __call__(self, f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList, enableSinusoid = True, enableNoise = True):
        # constant
        nHop = len(f0List)
        nOut = nHop * self.hopSize
        nHar = hFreqList.shape[1]
        nBin = self.fftSize // 2 + 1
        energyAnalysisWindowNormFac = 1.0 / np.sqrt(0.375)
        hFreqList = hFreqList.copy()

        # check input
        assert(f0List.ndim == 1)
        assert(hFreqList.ndim == 2)
        assert(hAmpList.ndim == 2)
        assert(hPhaseList.ndim == 2)
        assert(sinusoidEnergyList.ndim == 1)
        assert(len(hFreqList) == nHop)
        assert(len(hAmpList) == nHop)
        assert(len(hPhaseList) == nHop)
        assert(len(sinusoidEnergyList) == nHop)
        assert(noiseEnvList.ndim == 2)
        assert(noiseEnergyList.ndim == 1)
        assert(len(noiseEnvList) == nHop)
        assert(len(noiseEnergyList) == nHop)
        assert(noiseEnvList.shape[1] == nBin)

        # relative harmonic shift
        voiced = f0List > 0.0
        nVoiced = np.sum(voiced)
        hFreqList[voiced] *= f0List[voiced].reshape(nVoiced, 1) * np.arange(1, nHar + 1)
        hFreqList[np.logical_or(hFreqList <= 0.0, hFreqList > self.mvf)] = 0.0

        # relative phase shift & olaFac
        syncedHPhaseList = np.zeros((nHop * self.olaFac, nHar))
        basePhase = 0.0
        syncedHPhaseList[0] = hPhaseList[0]
        for iFrame in range(1, nHop * self.olaFac):
            iHop = iFrame // self.olaFac
            f0 = f0List[iHop]
            if(f0 <= 0.0):
                continue
            basePhase += f0 * 2 * np.pi * (self.hopSize / self.olaFac / self.samprate)
            syncedHPhaseList[iFrame] = wrap(hPhaseList[iHop] + (hFreqList[iHop] / f0) * wrap(basePhase))

        if(enableSinusoid):
            sinusoid = np.zeros(nOut)
            synthWindow = np.hanning(2 * self.hopSize)
            for iFrame in range(nHop * self.olaFac):
                iHop = iFrame // self.olaFac
                f0 = f0List[iHop]
                if(f0 <= 0.0 or sinusoidEnergyList[iHop] <= 0.0):
                    continue
                energyAnalysisRadius = int(round(self.samprate / f0)) * 2
                synthLeft = max(energyAnalysisRadius, self.hopSize)
                synthRight = max(energyAnalysisRadius + 1, self.hopSize)
                synthRange = np.arange(-synthLeft, synthRight)
                energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1)

                need = hFreqList[iHop] > 0.0
                synthed = synthSinusoid(hFreqList[iHop][need], hAmpList[iHop][need], syncedHPhaseList[iFrame][need], synthRange, self.samprate)

                # integrate energy
                energyBegin = synthLeft - energyAnalysisRadius
                energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1)
                energy = np.mean((synthed[energyBegin:energyBegin + energyAnalysisRadius * 2 + 1] * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)

                assert(energy > 0.0)

                # ola
                olaBegin = synthLeft - self.hopSize
                iCenter = iFrame * self.hopSize // self.olaFac
                ob, oe, ib, ie = getFrameRange(nOut, iCenter, 2 * self.hopSize)

                if(iHop < nHop - 1):
                    currEnergy = lerp(sinusoidEnergyList[iHop], sinusoidEnergyList[iHop + 1], (iFrame / self.olaFac) % 1)
                else:
                    currEnergy = sinusoidEnergyList[iHop]
                sinusoid[ib:ie] += synthed[olaBegin + ob:olaBegin + oe] * np.sqrt(currEnergy / energy) * synthWindow[ob:oe]
            sinusoid /= self.olaFac

        if(enableNoise):
            noise = np.zeros(nOut)
            noiseTemplate = np.random.uniform(-1.0, 1.0, nOut)
            noiseTemplate = filterNoise(noiseTemplate, noiseEnvList, self.hopSize)

            # energy normalize & apply noise energy envelope
            for iFrame in range(nHop * self.olaFac):
                if(noiseEnergyList[iHop] <= 0.0):
                    continue
                iCenter = iFrame * self.hopSize // self.olaFac
                iHop = iFrame // self.olaFac
                f0 = f0List[iHop]
                if(f0 > 0.0):
                    frame = getFrame(noiseTemplate, iCenter, 4 * int(round(self.samprate / f0)) + 1)
                else:
                    frame = getFrame(noiseTemplate, iCenter, 2 * self.hopSize)
                energyAnalysisWindow = np.hanning(len(frame))
                noiseEnergy = np.mean((frame * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)

                ob, oe, ib, ie = getFrameRange(nOut, iCenter, 2 * self.hopSize)
                if(iHop < nHop - 1):
                    currEnergy = lerp(noiseEnergyList[iHop], noiseEnergyList[iHop + 1], (iFrame / self.olaFac) % 1)
                    sinusoidCurrEnergy = lerp(sinusoidEnergyList[iHop], sinusoidEnergyList[iHop + 1], (iFrame / self.olaFac) % 1)
                else:
                    currEnergy = noiseEnergyList[iHop]
                    sinusoidCurrEnergy = sinusoidEnergyList[iHop]
                if(f0 > 0.0):
                    energyAnalysisRadius = int(round(self.samprate / f0)) * 2
                    synthLeft = max(energyAnalysisRadius, self.hopSize)
                    synthRight = max(energyAnalysisRadius + 1, self.hopSize)
                    synthRange = np.arange(-synthLeft, synthRight)
                    energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1)

                    need = hFreqList[iHop] > 0.0
                    nHar = min(np.sum(need), self.maxNoiseEnvHarmonic)
                    synthed = synthSinusoid(hFreqList[iHop][need][:nHar], hAmpList[iHop][need][:nHar], syncedHPhaseList[iFrame][need][:nHar], synthRange, self.samprate)
                    synthed -= np.min(synthed)
                    # dc adjustment
                    if(currEnergy > 0.0 and sinusoidCurrEnergy > currEnergy):
                        energyRatio = (sinusoidCurrEnergy - currEnergy) / sinusoidCurrEnergy
                        synthed += self.maxNoiseEnvDCAdjustment * (1.0 - np.sqrt(energyRatio))

                    # integrate energy
                    energyBegin = synthLeft - energyAnalysisRadius
                    energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1)
                    envEnergy = np.mean((synthed[energyBegin:energyBegin + energyAnalysisRadius * 2 + 1] * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)

                    assert(envEnergy > 0.0)

                    # ola
                    olaBegin = synthLeft - self.hopSize
                    normalizedEnv = synthed[olaBegin:olaBegin + 2 * self.hopSize] / np.sqrt(envEnergy)
                    frame = getFrame(noiseTemplate, iCenter, 2 * self.hopSize)
                    noise[ib:ie] += (frame * np.sqrt(currEnergy / noiseEnergy) * normalizedEnv * synthWindow)[ob:oe]
                else:
                    noise[ib:ie] += (frame * np.sqrt(currEnergy / noiseEnergy) * synthWindow)[ob:oe]
            noise /= self.olaFac
        out = np.zeros(nOut)
        if(enableSinusoid):
            saveWav("sinrs.wav", sinusoid, self.samprate)
            out += sinusoid
        if(enableNoise):
            saveWav("noisers.wav", noise, self.samprate)
            out += noise

        saveWav("synthed.wav", out, self.samprate)

        return out
