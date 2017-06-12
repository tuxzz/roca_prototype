import numpy as np
import scipy.interpolate as ipl
import scipy.signal as sp
from .common import *

class Processor:
    def __init__(self, sr):
        self.samprate = float(sr)

    def simple(self, timeList, f0List, hPhaseList = None, vAdditional = (), vuvAdditional = ()):
        nHop = len(f0List)
        nNewHop = len(timeList)

        newF0List = np.zeros(nNewHop)
        if(hPhaseList is not None):
            newHPhaseList = np.zeros((nNewHop, hPhaseList.shape[1]))
        newVAdditional, newVUVAdditional = [], []
        vAdditional, vuvAdditional = list(vAdditional), list(vuvAdditional)
        for iItem, item in enumerate(vAdditional):
            if(isinstance(item, tuple)):
                assert(len(item) == 2)
                item, default = item
                vAdditional[iItem] = item
            else:
                default = 0.0
            newItem = np.full((nNewHop, *item.shape[1:]), default)
            newVAdditional.append(newItem)

        for iItem, item in enumerate(vuvAdditional):
            if(isinstance(item, tuple)):
                assert(len(item) == 2)
                item, default = item
                vuvAdditional[iItem] = item
            else:
                default = 0.0
            newItem = np.full((nNewHop, *item.shape[1:]), default)
            newVUVAdditional.append(newItem)

        # v
        segments = splitArray(f0List)
        segBeginHop = 0
        newHopIndexList = np.arange(nNewHop)
        for iSegment, segment in enumerate(segments):
            if(segment[0] > 0.0):
                segEndHop = segBeginHop + len(segment)
                newHopIndexSlice = newHopIndexList[np.logical_and(timeList >= segBeginHop, timeList <= segEndHop)]
                newHopIndexBegin, newHopIndexEnd = newHopIndexSlice[0], newHopIndexSlice[-1] + 1
                if(len(segment) == 1):
                    iHop = segment[0]
                    newF0List[newHopIndexBegin:newHopIndexEnd] = f0List[iHop]
                    if(hPhaseList is not None):
                        newHPhaseList[newHopIndexBegin:newHopIndexEnd] = hPhaseList[iHop]
                    for iItem, item in enumerate(vAdditional):
                        newVAdditional[iItem][newHopIndexBegin:newHopIndexEnd] = item[iHop]
                else:
                    iplX = np.arange(segBeginHop, segEndHop)
                    iplY = timeList[newHopIndexBegin:newHopIndexEnd]
                    newF0List[newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, f0List[segBeginHop:segEndHop], kind = 'linear')(iplY)
                    if(hPhaseList is not None):
                        newHPhaseList[newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, np.unwrap(hPhaseList[segBeginHop:segEndHop], axis = 0), axis = 0, kind = 'linear')(iplY)
                    for iItem, item in enumerate(vAdditional):
                        newVAdditional[iItem][newHopIndexBegin:newHopIndexEnd] = ipl.interp1d(iplX, item[segBeginHop:segEndHop], axis = 0, kind = 'linear')(iplY)
            segBeginHop += len(segment)
        del iplX, iplY

        # vuv
        for iItem, item in enumerate(vuvAdditional):
            newVUVAdditional[iItem][:] = ipl.interp1d(np.arange(nHop), item, axis = 0, kind = 'linear')(timeList)
        return newF0List, newHPhaseList, newVAdditional, newVUVAdditional
