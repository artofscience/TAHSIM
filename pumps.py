"""
Lumped parameter models describing the analytical -- typically nonlinear but static -- behaviour of (centrifugal) pumps.
"""
import numpy as np
from abc import ABC, abstractmethod
from utils import cubic_fit, quadratic_fit
from math import pi
from matplotlib import pyplot as plt

class Pump(ABC):
    g = 9.81  # gravitational acceleration
    rho = 1000  # fluid density
    rpmtoradps = (2 * pi) / 60 # conversion rate rpm to rad/s
    radpstorpm = 1 / rpmtoradps # conversion rate rad/s to rpm
    cubpstolmin = 60e3 # conversion rate m3/s to l/min
    lmintocubps = 1 / cubpstolmin # conversion rate l/min to m3/s
    gamma = rho * g

class CentrifugalPump(Pump):
    """
    Pressure in heads h[m], can be converted to Pa via P = y * h, with gamma specific weight
    Flow in L/min, can be converted to m3/s by multiplication with 60000
    """

    def __init__(self,
                 hm0: float = 6,
                 qn0: float = 4,
                 hn0: float = 4.5,
                 qm0: float = 8,
                 w0: float = 1770 * (2 * pi / 60),
                 effn: float = 0.35):

        self.qn0 = qn0 # nominal capacity
        self.hn0 = hn0 # nominal head

        self.hm0 = hm0 # maximum head at zero capacity
        self.qm0 = qm0 # maximum capacity at zero head
        self.w0 = w0 # reference speed

        self.q0p = np.array([0.0, self.qn0, self.qm0]) # capacity points for H-Q at w0
        self.h0p = np.array([self.hm0, self.hn0, 0.0]) # head points for H-Q at w0
        # self.p0p = self.gamma * self.h0p

        # self.hq0_coeff = quadratic_fit(self.q0p, self.h0p)
        self.hq0_coeff = cubic_fit(self.q0p, self.h0p, 0, 0)
        self.hq0 = np.poly1d(self.hq0_coeff) # H-Q curve at w0, h0(q0)
        # self.pq0 = self.gamma * self.hq0
        self.hq = lambda q, w: sum([self.hq0[i] * (w / self.w0) ** (2 - i) * q ** (i) for i in range(len(self.hq0)+1)]) # H-Q curve at w, h(q, w)
        # self.pq = lambda q, w: self.gamma * self.hq(q, w)

        self.pn0 = self.hn0 * self.gamma  # nominal pressure
        self.nhp = self.qn0 * self.pn0 / 60000  # nominal horse power
        self.effn0 = effn # nominal efficiency (at reference speed)
        self.nbp = self.nhp / self.effn0

        self.eff0p = np.array([0.0, self.effn0, 0.0]) # efficiency points for eta(q0)
        self.eff0 = np.poly1d(cubic_fit(self.q0p, self.eff0p, 1, 0))

    def solve(self, t, y):
        # note assumes here y = (shaft speed, flow rate)
        h_pump = self.hq(y[1], y[0])
        qop_ref = (self.w0 / y[0]) * y[1]
        hydraulic_power_ref = qop_ref * self.hq0(qop_ref) * self.gamma / 60000
        mechanical_power_ref = hydraulic_power_ref / self.eff0(qop_ref)
        tau = mechanical_power_ref * y[0]**2 / self.w0**3
        return tau, h_pump

    def get_operating_points(self, n: int = 100, tol: float = 1e-6):
        return np.linspace(tol, self.qm0-tol, n), np.linspace(0.0, self.hm0, n)