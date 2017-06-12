from revoice import *
from .common import *
import pylab as pl

class Synther:
    def __init__(self, voicedb, sr, **kwargs):
        self.samprate = sr
        self.hopSize = kwargs.get("hopSize", roundUpToPowerOf2(self.samprate * 0.0025))
        self.maxNHar = kwargs.get("maxNHar", 128)
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.voicedb = voicedb
        self.phoneConf = voicedb.loadObject("config.phone")

    def preprocess(self, phoneList, energyMatchPair, f0MatchPair):
        nPhone = len(phoneList)
        newPhoneList = []

        prevPhone = None
        for iPhone, (name, begin, end, votRatio) in enumerate(phoneList):
            assert(begin >= 0.0 and end > begin)
            assert(votRatio >= 0.0)
            meanF0 = matchLEQ(f0MatchPair[0], (begin + end) / 2)
            phone = self.phoneConf.query(name, meanF0, (), ())
            modifiedVot = (phone.vot - phone.left) * votRatio
            if(begin < modifiedVot):
                modifiedVot = modifiedVot - begin
                begin = 0.0
                votRatio = modifiedVot / (phone.vot - phone.left)
            else:
                begin -= modifiedVot
            if(iPhone != 0):
                newPhoneList[iPhone - 1][2] -= modifiedVot
            newPhoneList.append([name, begin, end, votRatio])

        newPhoneList = [tuple(x) for x in newPhoneList]
        return newPhoneList, energyMatchPair, f0MatchPair

    def synth(self, phoneList, energyMatchPair, f0MatchPair):
        nHop = int(np.ceil(phoneList[-1][2] * self.samprate / self.hopSize))
        nBin = self.fftSize // 2 + 1
        prevEnd = -1
        iPrevEnd = -1

        f0List, sinusoidEnergyList, noiseEnergyList = np.zeros(nHop), np.zeros(nHop), np.zeros(nHop)
        hFreqList, hAmpList, hPhaseList = np.full((nHop, self.maxNHar), 1.0), np.zeros((nHop, self.maxNHar)), np.zeros((nHop, self.maxNHar))
        noiseEnvList = np.zeros((nHop, nBin))
        sinEnvList = np.zeros((nHop, nBin))

        # splice
        timeProc = timetransform.Processor(self.samprate)
        for name, begin, end, votRatio in phoneList:
            assert(begin >= prevEnd)
            meanF0 = matchLEQ(f0MatchPair[0], (begin + end) / 2)
            phone = self.phoneConf.query(name, meanF0, (), ())
            pF0Object = self.voicedb.loadObject(phone.f0)
            pHNMObject = self.voicedb.loadObject(phone.hnm)
            pSinEnvObject = self.voicedb.loadObject(phone.sinEnv)
            assert(pF0Object.samprate == self.samprate)
            assert(pHNMObject.samprate == self.samprate)
            assert(pSinEnvObject.samprate == self.samprate)
            assert(pF0Object.hopSize == self.hopSize)
            assert(pHNMObject.hopSize == self.hopSize)
            assert(pSinEnvObject.hopSize == self.hopSize)
            assert(phone.overlap <= phone.left)
            assert(votRatio >= 0.0)

            # constant
            iLeft = int(round(phone.left * self.samprate / self.hopSize))
            iRight = int(np.ceil(phone.right * self.samprate / self.hopSize))
            iOverlapedLeft = int(round((phone.left - phone.overlap) * self.samprate / self.hopSize))
            nOverlapHop = iLeft - iOverlapedLeft

            iBegin = int(round(begin * self.samprate / self.hopSize))
            iEnd = int(round(end * self.samprate / self.hopSize))
            # get sample
            crossFadeWindowX = np.hanning(nOverlapHop * 2)[nOverlapHop:]
            crossFadeWindowY = np.hanning(nOverlapHop * 2)[:nOverlapHop]
            pF0List = pF0Object.f0List[iOverlapedLeft:iRight]
            pHFreqList, pHAmpList, pHPhaseList, pSinusoidEnergyList, pNoiseEnvList, pNoiseEnergyList = pHNMObject.hFreqList[iOverlapedLeft:iRight], pHNMObject.hAmpList[iOverlapedLeft:iRight], pHNMObject.hPhaseList[iOverlapedLeft:iRight], pHNMObject.sinusoidEnergyList[iOverlapedLeft:iRight], pHNMObject.noiseEnvList[iOverlapedLeft:iRight], pHNMObject.noiseEnergyList[iOverlapedLeft:iRight]
            pSinEnvList = pSinEnvObject.envList[iOverlapedLeft:iRight]
            pNHar = pHFreqList.shape[1]
            del pF0Object, pHNMObject, pSinEnvObject

            # calc timeList
            iVOT = int(round((phone.vot - phone.left) * self.samprate / self.hopSize))
            iAttack = int(round((phone.attack - phone.left) * self.samprate / self.hopSize))
            iRelease = int(round((phone.release - phone.left) * self.samprate / self.hopSize))
            iModifiedVOT = int(round(min((phone.vot - phone.left) * votRatio * self.samprate / self.hopSize, iEnd - iBegin)))
            iModifiedAttack = int(round((phone.attack - phone.left) * self.samprate / self.hopSize))
            iModifiedRelease = int(round((phone.release - phone.left) * self.samprate / self.hopSize))
            keepAttack = iModifiedAttack > iModifiedVOT and iAttack < (iRight - iLeft - 1) and iModifiedAttack < (iEnd - iBegin - 1)
            keepRelease = (iModifiedRelease > iModifiedVOT or iModifiedRelease > iModifiedAttack) and iRelease < (iRight - iLeft - 1) and iModifiedRelease < (iEnd - iBegin - 1)
            iplX = []
            iplY = []
            if(nOverlapHop > 0.0):
                iplX.append(-nOverlapHop)
                iplY.append(-nOverlapHop)
            iplX.append(0.0)
            iplY.append(0.0)
            if(iModifiedVOT > 0.0 and iVOT > 0.0):
                iplX.append(iModifiedVOT)
                iplY.append(iVOT)
            if(keepAttack):
                iplX.append(iModifiedAttack)
                iplY.append(iAttack)
            if(keepRelease):
                iplX.append(iModifiedRelease)
                iplY.append(iRelease)
            iplX.append(iEnd - iBegin - 1)
            iplY.append(iRight - iLeft - 1)
            print(iplX, iplY, keepAttack, keepRelease)
            timeList = ipl.PchipInterpolator(iplX, iplY)(np.linspace(-nOverlapHop, iEnd - iBegin - 1, iEnd - iBegin + nOverlapHop))
            timeList += nOverlapHop
            del iplX, iplY

            rNHar = min(pNHar, self.maxNHar)
            pF0List, pHPhaseList, (pHFreqList, pHAmpList, pSinusoidEnergyList, pSinEnvList), (pNoiseEnvList, pNoiseEnergyList) = timeProc.simple(timeList, pF0List, pHPhaseList, (pHFreqList, pHAmpList, pSinusoidEnergyList, pSinEnvList), (pNoiseEnvList, pNoiseEnergyList))
            if(nOverlapHop != 0):
                ncBegin = iBegin - nOverlapHop
                lx = [f0List[ncBegin:iBegin], sinusoidEnergyList[ncBegin:iBegin], noiseEnergyList[ncBegin:iBegin], hFreqList[ncBegin:iBegin,:rNHar], hAmpList[ncBegin:iBegin,:rNHar], hPhaseList[ncBegin:iBegin,:rNHar], noiseEnvList[ncBegin:iBegin], sinEnvList[ncBegin:iBegin]]
                ly = [pF0List[:nOverlapHop], pSinusoidEnergyList[:nOverlapHop], pNoiseEnergyList[:nOverlapHop], pHFreqList[:nOverlapHop,:rNHar], pHAmpList[:nOverlapHop,:rNHar], pHPhaseList[:nOverlapHop,:rNHar], pNoiseEnvList[:nOverlapHop], pSinEnvList[:nOverlapHop]]
                for i, x in enumerate(lx):
                    shape = (nOverlapHop,) + (1,) * (x.ndim - 1)
                    x *= crossFadeWindowX.reshape(shape)
                    x += ly[i] * crossFadeWindowY.reshape(shape)
                del lx, ly
            f0List[iBegin:iEnd], sinusoidEnergyList[iBegin:iEnd], noiseEnergyList[iBegin:iEnd], hFreqList[iBegin:iEnd,:rNHar], hAmpList[iBegin:iEnd,:rNHar], hPhaseList[iBegin:iEnd,:rNHar], noiseEnvList[iBegin:iEnd], sinEnvList[iBegin:iEnd] = pF0List[nOverlapHop:], pSinusoidEnergyList[nOverlapHop:], pNoiseEnergyList[nOverlapHop:], pHFreqList[nOverlapHop:,:rNHar], pHAmpList[nOverlapHop:,:rNHar], pHPhaseList[nOverlapHop:,:rNHar], pNoiseEnvList[nOverlapHop:], pSinEnvList[nOverlapHop:]
            prevEnd = end
            iPrevEnd = iEnd
        del timeList, pF0List, pHPhaseList, pHFreqList, pHAmpList, pSinusoidEnergyList, pNoiseEnvList, pNoiseEnergyList, pSinEnvList
        del timeProc

        uvList = f0List <= 0.0
        # pitch shift
        pitchProc = pitchtransform.Processor(self.samprate)
        for iHop in range(nHop):
            f0List[iHop] = linearLEQ(f0MatchPair[0], f0MatchPair[1], iHop * self.hopSize / self.samprate)
            sinusoidEnergyList[iHop] *= linearLEQ(energyMatchPair[0], energyMatchPair[1], iHop * self.hopSize / self.samprate)
            noiseEnergyList[iHop] *= linearLEQ(energyMatchPair[0], energyMatchPair[1], iHop * self.hopSize / self.samprate)
        f0List[uvList] = 0.0
        hAmpList = pitchProc(f0List, hFreqList, sinEnvList)

        # synth
        synProc = hnm.Synther(self.samprate)
        synthed = synProc(f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList)
        return synthed
