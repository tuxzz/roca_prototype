import numpy as np
import scipy.io.wavfile as wavfile
import scipy.interpolate as ipl
import scipy.signal as sp
import scipy.special as spec
import numba as nb
import numbers

windowDict = {
    #           func(N), main-lobe-width, mean
    'hanning': (sp.hanning, 1.5, 0.5),
    'blackman': (sp.blackman, 1.73, 0.42),
    'blackmanharris': (sp.blackmanharris, 2.0044, (35875 - 3504 * np.pi) / 100000),
}

def loadWav(filename): # -> samprate, wave in float64
    samprate, w = wavfile.read(filename)
    if(w.dtype == np.int8):
        w = w.astype(np.float64) / 127.0
    elif(w.dtype == np.short):
        w = w.astype(np.float64) / 32767.0
    elif(w.dtype == np.int32):
        w = w.astype(np.float64) / 2147483647.0
    elif(w.dtype == np.float32):
        w = w.astype(np.float64)
    elif(w.dtype == np.float64):
        pass
    else:
        raise ValueError("Unsupported sample format: %s" % (str(w.dtype)))
    return w, samprate

def saveWav(filename, data, samprate):
    wavfile.write(filename, int(samprate), data)

def splitArray(seq, cond = lambda v:v <= 0.0 or np.isnan(v)):
    if(len(seq) == 0):
        return []
    o = []
    n = len(seq)
    last = 0
    i = 0
    while(i < n):
        if(cond(seq[i])):
            if(last != i):
                o.append(seq[last:i])
            last = i
            while(i < n and cond(seq[i])): i += 1
            o.append(seq[last:i])
            last = i
        i += 1
    if(last != n):
        o.append(seq[last:])
    return o

def simpleDCRemove(x):
    return x - np.mean(x)

@nb.jit(nb.types.Tuple((nb.int64, nb.int64, nb.int64, nb.int64))(nb.int64, nb.int64, nb.int64), nopython = True, cache = True)
def getFrameRange(inputLen, center, size):
    leftSize = size // 2
    rightSize = size - leftSize # for odd size

    inputBegin = min(inputLen, max(center - leftSize, 0))
    inputEnd = max(0, min(center + rightSize, inputLen))

    outBegin = max(leftSize - center, 0)
    outEnd = outBegin + (inputEnd - inputBegin)

    return outBegin, outEnd, inputBegin, inputEnd

@nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.int64), nopython = True, cache = True)
def getFrame(input, center, size):
    out = np.zeros(size, input.dtype)

    outBegin, outEnd, inputBegin, inputEnd = getFrameRange(len(input), center, size)

    out[outBegin:outEnd] = input[inputBegin:inputEnd]
    return out

@nb.jit(nb.int64(nb.int64, nb.int64), nopython = True, cache = True)
def getNFrame(inputSize, hopSize):
    return inputSize // hopSize + 1 if(inputSize % hopSize != 0) else inputSize // hopSize

def getWindow(window):
    if(type(window) is str):
        return windowDict[window]
    elif(type(window) is tuple):
        assert(len(window) == 3)
        return window
    else:
        raise TypeError("Invalid window.")

def mavg(x, order):
    return sp.fftconvolve(x, np.full(order, 1.0 / order))[order // 2:order // 2 + len(x)]

def roundUpToPowerOf2(v):
    return int(2 ** np.ceil(np.log2(v)))

def parabolicInterpolation(input, i, val = True, overAdjust = False):
    lin = len(input)

    ret = 0.0
    if(i > 0 and i < lin - 1):
        s0 = float(input[i - 1])
        s1 = float(input[i])
        s2 = float(input[i + 1])
        a = (s0 + s2) / 2.0 - s1
        if(a == 0):
            return (i, input[i])
        b = s2 - s1 - a
        adjustment = -(b / a * 0.5)
        if(not overAdjust and abs(adjustment) > 1.0):
            adjustment = 0.0
        x = i + adjustment
        if(val):
            y = a * adjustment * adjustment + b * adjustment + s1
            return (x, y)
        else:
            return x
    else:
        x = i
        if(val):
            y = input[x]
            return (x, y)
        else:
            return x

def fixIntoUnit(x):
    if(isinstance(x, complex)):
        return (1 + 0j) / np.conj(x) if np.abs(x) > 1.0 else x
    else:
        need = np.abs(x) > 1.0
        x[need] = (1 + 0j) / np.conj(x[need])
        return x

def formantFreq(n, L = 0.168, c = 340.29):
    return (2 * n - 1) * c / 4 / L

def countFormant(freq, L = 0.168, c = 340.29):
    return int(round((freq * 4 * L / c + 1) / 2))

def preEmphasisResponse(x, freq, sr):
    x = np.asarray(x)
    a = np.exp(-2.0 * np.pi * freq / sr)
    z = np.exp(2j * np.pi * x / sr)
    return np.abs(1 - a / z)

def preEmphasis(x, freq, sr):
    o = np.zeros(len(x))
    fac = np.exp(-2.0 * np.pi * freq / sr)
    o[0] = x[0]
    o[1:] = x[1:] - x[:-1] * fac
    return o

def deEmphasis(x, freq, sr):
    o = x.copy()
    fac = np.exp(-2.0 * np.pi * freq / sr)

    for i in range(1, len(x)):
        o[i] += o[i - 1] * fac
    return o

def lerp(a, b, ratio):
    return a + (b - a) * ratio

def formantFreq(n, L = 0.168, c = 340.29):
    return (2 * n - 1) * c / 4 / L

def formantNumber(freq, L = 0.168, c = 340.29):
    return int(round((freq * 4 * L / c + 1) / 2))

def freqToMel(x, a = 2595.0, b = 700.0):
    return a * np.log10(1.0 + x / b)

def melToFreq(x, a = 2595.0, b = 700.0):
    return (np.power(10, x / a) - 1.0) * b

def freqToSemitone(freq):
    return np.log2(freq / 440.0) * 12.0 + 69.0

def semitoneToFreq(semi):
    return np.power(2, (semi - 69.0) / 12.0) * 440.0

def calcSRER(x, y):
    return np.log10(np.std(x) / np.std(x - y)) * 20.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lambertW(x):
    A = 2.344
    B = 0.8842
    C = 0.9294
    D = 0.5106
    E = -1.213
    y = (2.0 * np.e * x + 2.0) ** 0.5
    w = (2.0 * np.log(1.0 + B * y) - np.log(1.0 + C * np.log(1.0 + D * y)) + E) / (1.0 + 1.0 / (2.0 * np.log(1.0 + B * y) + 2.0 * A))
    for i in range(24):
        u = np.exp(w)
        v = w * u - x
        w -= v / ((1.0 + w) * u - ((w + 2.0) * v) / (2.0 * w + 2.0))
    return w

def wrap(phase):
    out = phase - np.round(phase / (2.0 * np.pi)) * 2.0 * np.pi
    if(isinstance(phase, numbers.Number)):
        if(out > np.pi):
            out -= 2 * np.pi
        elif(out < np.pi):
            out += 2 * np.pi
    else:
        out[out > np.pi] -= 2 * np.pi
        out[out < -np.pi] += 2 * np.pi

    return out
