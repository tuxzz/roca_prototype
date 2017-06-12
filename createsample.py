assert __name__ == "__main__"
from roca import *
from revoice import *
from roca.common import *

import sys, os, getopt

dbPath = os.environ.get("ROCA_VOICEDB_PATH", None)
wavPath = None
outName = None
analyzeF0, analyzeHNM, analyzeSinEnv = False, False, False

optList, args = getopt.getopt(sys.argv[1:], [], [
    "db=",
    "wav=",
    "f0",
    "hnm",
    "sinenv",
    "all",
])
for opt, arg in optList:
    if(opt == "--db"):
        dbPath = arg
    elif(opt == "--wav"):
        wavPath = arg
        if(outName is None):
            outName = os.path.splitext(os.path.basename(wavPath))[0]
    elif(opt == "--out"):
        outName = out
    elif(opt == "--f0"):
        analyzeF0 = True
    elif(opt == "--hnm"):
        analyzeHNM = True
    elif(opt == "--sinenv"):
        analyzeSinEnv = True
    elif(opt == "--all"):
        analyzeF0, analyzeHNM, analyzeSinEnv = True, True, True
    else:
        assert False, "Invalid option %s" % (opt)

assert dbPath is not None, "Database path is not specified."
assert wavPath is not None, "Wave file path is not specified."

voiceDB = voicedb.DevelopmentVoiceDB(dbPath)
w, sr = loadWav(wavPath)
f0Object, hnmObject, envObject = None, None, None
f0ObjectPath, hnmObjectPath, sinEnvObjectPath = "%s.f0Obj" % (outName,), "%s.hnmObj" % (outName,), "%s.sinEnvObj" % (outName,)
if(w.ndim != 1):
    print("Warning: Multichannel audio is not supported, use left channel only", file = sys.stderr)
    w = w.T[0]
if(not sr in (44100, 48000)):
    print("Warning: Samprate of 44100 or 48000 is recommended", file = sys.stderr)

if(analyzeF0):
    print("F0...")
    pyinProc = pyin.Processor(sr)
    obsProbList = pyinProc(w)
    monopitchProc = monopitch.Processor(*monopitch.parameterFromPYin(pyinProc))
    f0List = monopitchProc(w, obsProbList)
    f0Object = objects.F0Object(f0List, monopitchProc.hopSize, sr)
    del pyinProc, obsProbList, monopitchProc, f0List
    voiceDB.saveObject(f0ObjectPath, f0Object)

if((analyzeHNM or analyzeSinEnv) and f0Object is None):
    assert voiceDB.hasObject(f0ObjectPath), "No f0 object available."
    f0Object = voiceDB.loadObject(f0ObjectPath)

if(analyzeHNM):
    print("HNM...")
    hnmProc = hnm.Analyzer(sr, harmonicAnalysisMethod = "qfft")
    assert f0Object.hopSize == hnmProc.hopSize
    f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmProc(w, f0Object.f0List)

    # update f0
    f0Object.f0List = f0List
    voiceDB.saveObject(f0ObjectPath, f0Object)

    hnmObject = objects.HNMObject(sr, hnmProc.hopSize, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList)
    voiceDB.saveObject(hnmObjectPath, hnmObject)
    del hnmProc, f0List, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList

if(analyzeSinEnv):
    print("SinEnv...")
    if(hnmObject is None):
        assert voiceDB.hasObject(hnmObjectPath), "No hnm object available."
        hnmObject = voiceDB.loadObject(hnmObjectPath)
    synProc = hnm.Synther(sr)
    envProc = mfienvelope.Processor(sr)
    assert hnmObject.hopSize == envProc.hopSize
    assert hnmObject.hopSize == f0Object.hopSize
    assert synProc.hopSize == envProc.hopSize
    synthed = synProc(f0Object.f0List, hnmObject.hFreqList, hnmObject.hAmpList, hnmObject.hPhaseList, hnmObject.sinusoidEnergyList, hnmObject.noiseEnvList, hnmObject.noiseEnergyList)
    sinEnv = envProc(w, f0Object.f0List)
    sinEnvObject = objects.EnvObject(sinEnv, envProc.hopSize, sr)
    voiceDB.saveObject(sinEnvObjectPath, sinEnvObject)
