from .common import *

class F0Object:
    classVersion = 0

    def __init__(self, f0List, hopSize, samprate):
        self.version = self.classVersion
        self.f0List = f0List
        self.hopSize = hopSize
        self.samprate = samprate

    @classmethod
    def convertFromOldVersion(cls, oldObject):
        raise TypeError("Unsupported object verison")

class HNMObject:
    classVersion = 0

    def __init__(self, samprate, hopSize, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList):
        self.version = self.classVersion
        self.samprate = samprate
        self.hopSize = hopSize
        self.hFreqList, self.hAmpList, self.hPhaseList, self.sinusoidEnergyList, self.noiseEnvList, self.noiseEnergyList = hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList

    @classmethod
    def convertFromOldVersion(cls, oldObject):
        raise TypeError("Unsupported object verison")

class EnvObject:
    classVersion = 0

    def __init__(self, envList, hopSize, samprate):
        self.version = self.classVersion
        self.envList = envList
        self.hopSize = hopSize
        self.samprate = samprate

    @classmethod
    def convertFromOldVersion(cls, oldObject):
        raise TypeError("Unsupported object verison")

class PhoneConfigurationObject:
    classVersion = 0

    def __init__(self):
        self.version = self.classVersion
        self.phoneDict = {}

    def insert(self, name, meanF0, pre, post, left, right, vot, attack, release, overlap, f0, hnm, sinEnv):
        l = self.phoneDict.get(name, None)
        if(l is None):
            l = []
            self.phoneDict[name] = l
        l.append(Phone(name, meanF0, pre, post, left, right, vot, attack, release, overlap, f0, hnm, sinEnv))

    def remove(self, phone):
        l = self.phoneDict[phone.name]
        l.remove(phone)
        if(not l):
            del self.phoneDict[phone.name]

    def query(self, name, meanF0, pre, post, strict = False):
        l = self.phoneDict.get(name, None)
        if(l is None):
            raise KeyError("Query failed")
        def score(x):
            score = -np.abs(x.meanF0 - meanF0)
            if(x.pre == pre):
                score += 10000
            elif(x.pre == tuple()):
                score += 1000
            if(x.post == post):
                score += 10000
            elif(x.post == tuple()):
                score += 1000
            return score
        l.sort(key = score)
        return l[-1]

class Phone:
    def __init__(self, name, meanF0, pre, post, left, right, vot, attack, release, overlap, f0, hnm, sinEnv):
        self.name = name
        self.meanF0 = meanF0
        self.pre = pre
        self.post = post
        self.left, self.right, self.vot, self.attack, self.release, self.overlap = left, right, vot, attack, release, overlap
        self.f0 = f0
        self.hnm = hnm
        self.sinEnv = sinEnv
