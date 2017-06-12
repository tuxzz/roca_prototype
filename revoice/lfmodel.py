import numpy as np
import scipy.optimize as so
from .common import *
import pylab as pl

# tp, te, ta are T-normalized

# [1] Huber, Stefan, and Axel Roebel. "On the use of voice descriptors for glottal source shape parameter estimation." Computer Speech & Language 28.5 (2014): 1170-1194.
# [2] Fant, Gunnar. "The LF-model revisited. Transformations and frequency domain analysis." Speech Trans. Lab. Q. Rep., Royal Inst. of Tech. Stockholm 2.3 (1995): 40.
def fromRd(Rd):
    if(Rd < 0.21): Rap = 1e-6
    elif(Rd < 2.7): Rap = (4.8 * Rd - 1.0) * 0.01
    else: Rap = 0.323 / Rd
    if(Rd < 2.7):
        Rkp = (22.4 + 11.8 * Rd) * 0.01
        Rgp = 0.25 * Rkp / ((0.11 * Rd) / (0.5 + 1.2 * Rkp) - Rap)
    else:
        OQupp = 1.0 - 1.0 / (2.17 * Rd)
        Rgp = 9.3552e-3 + 5.96 / (7.96 - 2.0 * OQupp)
        Rkp = 2.0 * Rgp * OQupp - 1.0428

    tp = 1.0 / (2.0 * Rgp)
    te = tp * (Rkp + 1.0)
    ta = Rap
    return tp, te, ta

# [1] Fant, Gunnar, Johan Liljencrants, and Qi-guang Lin. "A four-parameter model of glottal flow." STL-QPSR 4.1985 (1985): 1-13.
# [2] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
def calcParameter(T0, Ee, tp, te, ta):
    assert(T0 > 0.0 and Ee > 0.0 and tp > 0.0 and te > 0.0 and ta > 0.0)
    wg = np.pi / tp #[1](2)
    sinWgTe = np.sin(wg * te)
    cosWgTe = np.cos(wg * te)

    e = (-ta * lambertW((te - T0) * np.exp((te - T0) / ta) / ta) + te - T0) / (ta * te - ta * T0) #[1](12)
    A = e * ta / (e * e * ta) + (te - T0) * (1.0 - e * ta) / (e * ta) # [3]p.18, integral{0, T0} Ug(t) dt
    afunc = lambda x : (x * x + wg * wg) * sinWgTe * A + wg * (np.exp(-x * te) - cosWgTe) + x * sinWgTe
    dafunc = lambda x : sinWgTe * (2 * A * x + 1) - wg * te * np.exp(-te * x)
    a = so.fsolve(afunc, 0.0, fprime = dafunc)[0]
    assert(a < 1e9)

    E0 = -Ee / (np.exp(a * te) * sinWgTe) #(5)
    return wg, sinWgTe, cosWgTe, e, A, a, E0

# [1] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
def calcSpectrum(f, T0, Ee, tp, te, ta):
    assert((f > 0.0).all)
    tp *= T0
    te *= T0
    ta *= T0
    wg, sinWgTe, cosWgTe, e, A, a, E0 = calcParameter(T0, Ee, tp, te, ta)

    r = a - 2.0j * np.pi * f
    P1 = E0 / (r * r + wg * wg)
    P2 = wg + np.exp(r * te) * (r * sinWgTe - wg * cosWgTe)
    P3 = Ee * np.exp((-2.0j * np.pi * te) * f) / ((2.0j * e * ta * np.pi * f) * (e + 2.0j * np.pi * f))
    P4 = e * (1.0 - e * ta) * (1.0 - np.exp((-2.0j * np.pi * (T0 - te)) * f)) - (2.0j * e * ta * np.pi) * f

    pole = P1 * P2 + P3 * P4
    magn = np.abs(pole)
    phase = np.angle(pole)

    return magn, phase

# [1] Fant, Gunnar, Johan Liljencrants, and Qi-guang Lin. "A four-parameter model of glottal flow." STL-QPSR 4.1985 (1985): 1-13.
# [2] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
def calcFlowDerivative(t, T0, Ee, tp, te, ta):
    tp *= T0
    te *= T0
    ta *= T0
    wg, sinWgTe, cosWgTe, e, A, a, E0 = calcParameter(T0, Ee, tp, te, ta)

    out = np.zeros(t.shape)
    o = t <= te
    c = np.logical_and(t <= T0, t > te)
    to = t[o]
    tc = t[c]
    out[o] = E0 * np.exp(a * to) * np.sin(wg * to)
    out[c] = -Ee * (np.exp((te - tc) * e) - np.exp((te - T0) * e)) / (e * ta) # [2] p.18. formula in [1] p.8 is wrong.
    return out
