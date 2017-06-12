from revoice.common import *
import bisect

def testBit(int_type, offset):
    mask = 1 << offset
    return(int_type & mask)

def setBit(int_type, offset):
    mask = 1 << offset
    return(int_type | mask)

def clearBit(int_type, offset):
    mask = ~(1 << offset)
    return(int_type & mask)

def toggleBit(int_type, offset):
    mask = 1 << offset
    return(int_type ^ mask)

def setBitValue(int_type, offset, value):
    if(bool(value)):
        setBit(int_type, offset)
    else:
        clearBit(int_type, offset)

def findL(l, v):
    i = bisect.bisect_left(l, v)
    return len(v) if(i == 0) else i - 1

def findG(l, v):
    return bisect.bisect_right(l, v)

def findLEQ(l, v):
    i = bisect.bisect_right(l, v)
    return len(v) if(i == 0) else i - 1

def findGEQ(l, v):
    return bisect.bisect_left(l, v)

def find(l, v):
    n = len(v)
    i = bisect.bisect_left(l, v)
    return i if(i != n and l[i] == v) else n

def matchLEQ(l, v):
    i = bisect.bisect_right(l, v)
    return i if(i == 0) else i - 1

def matchGEQ(l, v):
    n = len(v)
    i = bisect.bisect_left(l, v)
    return n - 1 if(i == n) else i

def linearLEQ(lx, ly, x):
    if(x <= lx[0]):
        return ly[0]
    elif(x >= lx[-1]):
        return ly[-1]
    i = matchLEQ(lx, x)
    return lerp(ly[i], ly[i + 1], (x - lx[i]) / (lx[i + 1] - lx[i]))
