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

    def pressure(self, volume: float, t: float) -> float:
        """
        Ventricular pressure via time-varying actuation pressure:
        Pv = E * (V - V0) + Pact(t)
        """
        return self.E * (volume - self.V0) + self.Pact(t)

    def pressure_diff(self, volume: float, flow: float, t: float) -> float:
        """
        dPv/dt = E * Q + dPact/dt
        Q = dV/dt (ventricular flow rate)
        """
        return self.E * flow + self.Pact.diff(t)

class PressureActuatedNonlinearMembrane(TAH):
    def __init__(self, Pact: TDP = TDP(min=-4, max=120), V0: float = 20,
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

    def pressure(self, volume: float, t: float) -> float:
        """
        Ventricular pressure via time-varying actuation pressure:
        Pv = a * ln(z / (1-z)) + d + Pact(t), z = b * (dV - c)
        """
        dV = volume - self.V0
        tmp = self.b * (dV - self.c)
        Pv = self.a * np.log(tmp / (1 - tmp)) + self.d
        Pv += self.Pact(t)
        return Pv

    def pressure_diff(self, volume: float, flow: float, t: float) -> float:
        """
        dPv/dt = dPv/dV * Q + dPact/dt
        Q = dV/dt (ventricular flow rate)

        dPv/dV = a / (z * (1 - b * z)), z = dV - c
        """

        dPvdV =  self.a / (volume * (1 - self.b * volume))

        return dPvdV * flow + self.Pact.diff(t)

