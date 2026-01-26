"""
Lumped parameter models describing the -- typically nonlinear but static -- behaviour of TAHs.
"""

import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod

from utils import TDP


class TAH(ABC):
    def pressure(self, volume: float, t: float) -> float:
        pass

    def pressure_diff(self, volume: float, flow: float, t: float) -> float:
        pass

class TimeVaryingElastance(TAH):
    def __init__(self, E: TDP = TDP(min=0.06, max=2.31), V0: float = 20):
        self.E = E
        self.V0 = V0

    def pressure(self, volume: float, t: float) -> float:
        """
        Ventricular pressure via time-varying elastance model:
        Pv = E(t) * (V - V0)
        """
        return self.E(t) * (volume - self.V0)

    def pressure_diff(self, volume: float, flow: float, t: float) -> float:
        """
        dPv/dt = dE/dt * (V - V0) + E(t) * Q
        Q = dV/dt (ventricular flow rate)
        """
        return self.E.diff(t) * (volume - self.V0) + self.E(t) * flow

class PressureActuatedLinearMembrane(TAH):
    def __init__(self, Pact: TDP = TDP(min=-4, max=120), E: float = 0.1, V0: float = 20):
        self.Pact = Pact
        self.E = E
        self.V0 = V0

    def fcn(self, dv):
        return self.E * dv

    def pressure(self, volume: float, t: float) -> float:
        """
        Ventricular pressure via time-varying actuation pressure:
        Pv = E * (V - V0) + Pact(t)
        """
        dv = self.V0 - volume
        return  self.Pact(t) - self.fcn(dv)

    def pressure_diff(self, volume: float, flow: float, t: float) -> float:
        """
        dPv/dt = E * Q + dPact/dt
        Q = dV/dt (ventricular flow rate)
        """
        return self.E * flow + self.Pact.diff(t)

class PressureActuatedNonlinearMembrane(TAH):
    def __init__(self, Pact: TDP = TDP(min=0, max=120), V0: float = 80,
                 a: float = 10,
                 b: float = 0.01,
                 c: float = -20,
                 d : float = 13.85):
        self.Pact = Pact
        self.V0 = V0
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    """"
    Assume logit function for Pv(dV) = a * ln(z / (1 - z)) + d, with z = b * (dV - c)

    Default values:
    a = 10
    b = 0.01
    c = -20
    d = 20
    """

    def fcn(self, dv):
        """
        dp = fcn(dv)
        """
        tmp = self.b * (dv - self.c)
        return self.a * np.log(tmp / (1 - tmp)) + self.d

    def pressure(self, volume: float, t: float) -> float:
        dV = self.V0 - volume
        return self.Pact(t) - self.fcn(dV)

    def pressure_diff(self, volume: float, flow: float, t: float) -> float:
        """
        dPv/dt = dPv/dV * Q + dPact/dt
        Q = dV/dt (ventricular flow rate)

        pv = a * ln(z/(1-z)) + d, z = b(dV - c)
        dpv/dz = a / z(1-z), dz/dV = b
        dpv/dV = ab / z(1-z)
        """
        #dpv/df = -1
        #d dv / dv = -1
        # dpv/ dv = dpv / df * df / d dv * d dv / dv = df/ d dv
        dV = self.V0 - volume
        z = self.b * (dV - self.c)

        dPvdV =  self.a * self.b / (z * (1 - z))

        return self.Pact.diff(t) + dPvdV * flow

class LIMO(TAH):
    def __init__(self, Pact: TDP = TDP(min=0, max=120), L: float = 0.017, N: int = 8, D: float = 0.05, H: float = 0.001, mu: float = 3e5):
        self.Pact = Pact
        # self.c = np.array([-2.26531554e+00,  1.61644772e+01,  1.01374824e-02, -3.77366149e+01,
        #                    -3.00151200e+00,  1.06145026e+01,  2.87000461e+01,  9.13326808e-01])
        self.c = np.array([-3.73674074, 26.34867036, 1.09021501, -59.92398559,
               -5.03912252, 11.03755932, 42.90372121, 1.77124204])
        # self.c = np.array([ -1.80115542,  12.81912422,  -0.39968068, -29.69849822,
        # -2.1104906 ,  10.1370886 ,  23.2695619 ,   0.5423379 ])
        self.mmhg2pa = 133
        self.ml2m3 = 1e-6
        self.mu = mu / self.mmhg2pa
        self.L = L
        self.H = H
        self.N = N
        self.D = D


    def fcn(self, vc, pact):
        # normalized function
        # Pch = Pc N L / mu H = f[vcn, pan]
        return self.c[0] + self.c[1] * vc + self.c[2] * pact + self.c[3] * vc**2 + self.c[4] * pact**2 + self.c[5] * vc * pact + self.c[6] * vc**3 + self.c[7] * pact**3

    def pressure(self, vc: float, t: float) -> float:
        # Pc = muH/NL * f[vc/NL2D, L/muH Pact]
        vcn = vc * self.ml2m3 / self.N / self.L**2 / self.D
        pactn = self.Pact(t) * self.L / self.mu / self.H
        return self.mu * self.H * self.fcn(vcn, pactn) / self.N / self.L

    def dfcndvc(self, vc, pact):
        a, b, c, d, e, f, g, h = self.c
        return b + 2 * d * vc + f * pact + 3 * g * vc**2

    def dfcndpact(self, vc, pact):
        a, b, c, d, e, f, g, h = self.c
        return c + 2 * e * pact + f * vc + 3 * h * pact**2

    def pressure_diff(self, vc: float, flow: float, t: float) -> float:
        vcn = vc * self.ml2m3 / self.N / self.L ** 2 / self.D
        pactn = self.Pact(t) * self.L / self.mu / self.H
        return self.mu * self.H * (self.dfcndvc(vcn, pactn) * flow / self.N / self.L**2 / self.D * self.ml2m3 + self.dfcndpact(vcn, pactn) * self.Pact.diff(t) * self.L / self.mu / self.H) / self.N / self.L

if __name__ == '__main__':

    limo = LIMO(TDP(min=10, max=120))

    plt.figure()
    vc = np.linspace(0.1, 1.0, 100)
    t = np.linspace(0.001, 2.5, 100)
    for time in t:
        plt.plot(vc, limo.fcn(vc, time))

    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 2.5])

    plt.figure()
    vc = np.linspace(10, 100, 100)
    t = np.linspace(1, 140, 10)
    for time in t:
        plt.plot(vc, limo.pressure(vc, time))

    plt.xlim([0, 100])
    plt.ylim([-1, 100])

    plt.xlabel(r'$v_\text{c}$')
    plt.ylabel(r'$P_\text{c}$')

    plt.show()