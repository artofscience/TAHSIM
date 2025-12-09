"""
Lumped-parameter models describing the dynamics of DC motors.
"""
from abc import ABC, abstractmethod
import numpy as np

from scipy.integrate import solve_ivp
from math import sqrt, pi
from utils import Sigmoid

class Motor(ABC):
    rpmtoradps = (2 * pi) / 60  # conversion rate rpm to rad/s
    radpstorpm = 1 / rpmtoradps  # conversion rate rad/s to rpm

    def __init__(self, voltage=lambda t: Sigmoid()(t), load_torque=lambda w: 1e-9 * np.power(w, 2)):
        self.applied_voltage = voltage
        self.load_torque = load_torque

    def __call__(self, y0 = (0.0, 0.0), t: float = 1.0, voltage=None,
                 load_torque=None,
                 t_begin: float = 0.0,
                 atol: float = 1e-3, rtol: float = 1e-3):
        if voltage is not None: self.applied_voltage = voltage
        if load_torque is not None: self.load_torque = load_torque
        return solve_ivp(self.solve, [t_begin, t], y0, atol=atol, rtol=rtol)

    def solve(self, t, y):
        pass

    def set_voltage(self, voltage):
        self.applied_voltage = voltage

    def set_torque(self, torque):
        self.load_torque = torque

class DCMotor(Motor):

    """
    EOM for DC motors
    V = L d{I} + R I + kb d{theta}
    M dd{theta} = kt I - tau - mu d{theta}

    Say w = d{theta}

    V = L d{I} + R I + kb w
    M d{v} = kt I - mu w - tau

    I.e. two coupled first-order ode.

    From "DC Motors" (2010), by Javier R. Movellan

    Equilibrium analysis:

    Apply V = V0 and tau = tau0, we find equilibrium

    V0 = R I + kb w
    tau0 = kt I - mu w

    """
    def __init__(self, applied_voltage=None,
                 load_torque=None,
                 L: float = 0.11 / 1000,
                 R: float = 1.71,
                 M: float = 3.88 / 1e7,
                 kt: float = 5.9 / 1000,
                 mu: float = 12 / 1e7):

        # Default values for Maxxon Amax 22, 5 Watt motor
        super().__init__(applied_voltage, load_torque)
        self.L = L # motor inductance in Henrys
        self.R = R # motor winding resistance in Ohms
        self.M = M # rotor moment of inertia in Kilogram / meter squared
        self.kt = kt # motor torque constant in Newton Meters / Amp
        self.mu = mu # motor viscous friction constant in Newton Meters / (radians per second)
        self.kb = kt # motor back electro magnetic force constant
        self.ktR = self.kt / self.R

    def solve(self, t, y):
        # y = [I, v]

        dI = (self.applied_voltage(t) - self.R * y[0] - self.kb * y[1]) / self.L
        dv = (self.kt * y[0] - self.mu * y[1] - self.load_torque(t, y[1])) / self.M

        return [dI, dv]

    def solve_tau(self, t, i, w, tau):
        dI = (self.applied_voltage(t) - self.R * i - self.kb * w) / self.L
        dv = (self.kt * i - tau - self.mu * w) / self.M
        return [dI, dv]

    def speed(self, V: float = 1.0, tau: float = 1e-3):

        # can be rewritten as
        # v = vn - tau / (mu + kb kt / R)

        return (V * self.ktR - tau) / (self.mu + self.kb * self.ktR)

    def torque(self, V: float = 1.0, v: float = 1.0):

        # can be rewritten as
        # tau = tau_s - v * (mu + kb kt / R)
        return V * self.ktR - v * (self.mu + self.kb * self.ktR)

    def stall_torque(self, V: float = 1.0):
        # torque at no velocity
        # tau_s = V kt / R
        return self.torque(V, 0.0)

    def stall_current(self, V: float = 1.0):
        return V / self.R
        # or return self.stall_torque(V) / self.kt

    def max_speed(self, V: float = 1.0):
        # no load velocity
        # vn = tau_stall / (mu + kb kt / R)
        return self.speed(V, 0.0)

    def current(self, V: float = 1.0, v: float = 1.0):
        return self.stall_current(V) - v * (self.mu / self.kt + self.kb / self.R)

    def max_power_speed(self, V: float = 1.0):
        return 0.5 * self.max_speed(V)

    def max_power_torque(self, V: float = 1.0):
        return 0.5 * self.stall_torque(V)

    def efficiency(self, V: float = 1.0, v: float = 1.0):
        return (self.stall_torque(V) * v - (self.mu + self.kb * self.ktR) * v**2) / (V**2 / self.R - self.stall_torque(V) * v)

    def efficient_speed(self, V: float = 1.0):
        a = self.stall_torque(V)
        b = self.mu + self.kb * self.ktR
        c = V**2 / self.R

        return (b * c - sqrt(b**2 * c**2 - a**2 * b * c)) / (a * b)