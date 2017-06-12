import sys
import numpy as np

class SparseHMM:
    def __init__(self, init, frm, to, transProb):
        self.init = init
        self.frm = frm
        self.to = to
        self.transProb = transProb

        assert(self.init.ndim == 1)
        assert(self.frm.ndim == 1)
        assert(self.to.ndim == 1)
        assert(self.transProb.ndim == 1)
        assert(len(self.frm) == len(self.to))
        assert(len(self.transProb) == len(self.to))

    def viterbiDecode(self, obsSeq):
        nState = len(self.init)
        nFrame = len(obsSeq)
        nTrans = len(self.transProb)

        assert(obsSeq.ndim == 2)
        assert(obsSeq.shape[1] == nState)

        #scale = np.zeros((nFrame), dtype = np.float64)

        if(len(obsSeq) < 1): # too short
            return np.zeros(0, dtype = np.int)

        delta = np.zeros((nState), dtype = np.float64)
        oldDelta = np.zeros((nState), dtype = np.float64)
        psi = np.zeros((nFrame, nState), dtype = np.int) # matrix of remembered indices of the best transitions

        # init first frame
        oldDelta = self.init * obsSeq[0]
        deltaSum = np.sum(oldDelta)

        #scale[0] = 1.0 / deltaSum

        # rest of forward step
        fromState = self.frm
        toState = self.to
        currTransProb = self.transProb
        for iFrame in range(1, nFrame):
            currValue = oldDelta[fromState] * currTransProb

            for iTrans in range(nTrans):
                ts = toState[iTrans]
                if(currValue[iTrans] > delta[ts]):
                    delta[ts] = currValue[iTrans] # will be multiplied by the right obs later
                    psi[iFrame][ts] = fromState[iTrans]

            delta *= obsSeq[iFrame]
            deltaSum = np.sum(delta)

            if(deltaSum > 0.0):
                oldDelta = delta / deltaSum
                #scale[iFrame] = 1.0 / deltaSum
            else:
                print("WARNING: Viterbi decoder has been fed some zero probabilities at frame %d." % (iFrame), file = sys.stderr)
                oldDelta.fill(1.0 / nState)
                #scale[iFrame] = 1.0
            delta.fill(0.0)

        # init backward step
        bestStateIdx = np.argmax(oldDelta)
        #bestValue = oldDelta[bestStateIdx]

        path = np.ndarray((nFrame), dtype = np.int) # the final output path
        path[-1] = bestStateIdx

        # rest of backward step
        for iFrame in reversed(range(nFrame - 1)):
            path[iFrame] = psi[iFrame + 1][path[iFrame + 1]]
        return path
