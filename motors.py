"""
Lumped-parameter models describing the dynamics of DC motors.
"""
from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

from scipy.integrate import solve_ivp
from math import sqrt, pi
from utils import Sigmoid

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large'}
pylab.rcParams.update(params)

class Motor(ABC):
    rpmtoradps = (2 * pi) / 60  # conversion rate rpm to rad/s
    radpstorpm = 1 / rpmtoradps  # conversion rate rad/s to rpm

    def __init__(self, voltage=lambda t: Sigmoid()(t), load_torque=lambda w: 1e-9 * np.power(w, 2)):
        self.applied_voltage = voltage
        self.load_torque = load_torque

    def __call__(self, y0 = (0.0, 0.0), t: float = 1.0, voltage=None,
                 load_torque=None,
                 t_begin: float = 0.0,
                 atol: float = 1e-6, rtol: float = 1e-6):
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
        self.kb = self.kt # motor back electro magnetic force constant or voltage constant in Volts / (radians per second)
        self.kv = 1 / self.kb # velocity or speed constant in (radians per second) / Volts
        self.alpha = self.kt / self.R
        self.beta = self.kb * self.alpha
        self.gamma = self.mu + self.beta
        self.cm = self.M / self.gamma # mechanical time constant
        self.ce = self.L / self.R # electrical time constant
        self.stall_torque = lambda v: self.alpha * v
        self.max_speed = lambda v: self.stall_torque(v) / self.gamma
        self.stall_current = lambda v: v / self.R
        self.max_power_speed = lambda v: 0.5 * self.stall_torque(v) / self.gamma
        self.max_power_torque = lambda v: 0.5 * self.stall_torque(v)
        self.max_power_current = lambda v: 0.5 * self.stall_current(v)
        self.max_power = lambda v: self.max_power_torque(v) * self.max_power_speed(v)

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
        # w[v, tau] = wb - tau/gamma
        return self.max_speed(V) - tau / self.gamma

    def torque(self, V: float = 1.0, w: float = 1.0):
        # tau[v, w] = tb - gamma * w
        return self.stall_torque(V) - self.gamma * w

    def current_speed(self, V: float = 1.0, w: float = 1.0):
        # i[v, w] = ib - (gamma/kt) * w
        return self.stall_current(V) - (self.gamma / self.kt) * w

    def current_torque(self, V: float = 1.0, tau: float = 1e-3):
        return tau / self.kt

    def power_electrical_speed(self, V: float = 1.0, w: float = 1.0):
        # Pe[v, w] = v i[v, w] = v**2 / R - (gamma / kt) v w
        return V**2 / self.R - (self.gamma / self.kt) * V * w

    def power_electrical_torque(self, V: float = 1.0, tau: float = 1e-3):
        return V * tau / self.kt

    def power_mechanical_speed(self, V: float = 1.0, w: float = 1.0):
        # Pm[v, w] = w tau[v, w] = a v w - gamma w**2
        return self.alpha * V * w - self.gamma * w**2

    def power_mechanical_torque(self, V: float = 1.0, tau: float = 1e-3):
        return self.max_speed(V) * tau - tau**2 / self.gamma

    def efficiency_speed(self, V: float = 1.0, w: float = 1.0):
        # n = Pm / Pe = (a v w - gamma w**2) / (v**2 / R - a v w)
        return (self.stall_torque(V) * w - self.gamma * w ** 2) / (V ** 2 / self.R - self.stall_torque(V) * w)

    def max_efficiency_speed(self, V: float = 1.0):
        a = self.stall_torque(V)
        b = self.gamma
        c = V**2 / self.R

        return (b * c - sqrt(b**2 * c**2 - a**2 * b * c)) / (a * b)

    def efficiency_torque(self, V: float = 1.0, tau: float = 1e-3):
        return (self.alpha - tau / V) * self.kt / self.gamma

if __name__ == '__main__':
    motor = DCMotor()

    fig, ax = plt.subplots()

    v = 6
    omega_max = motor.max_speed(v)
    stall_torque = motor.stall_torque(v)
    stall_current = motor.stall_current(v)

    omega = np.linspace(1e-9, omega_max, 100)
    torque =  motor.torque(v, omega)
    ax.plot(omega / omega_max, torque / stall_torque, label=r'$\frac{\tau}{\bar{\tau}}$')

    current = motor.current_speed(v, omega)
    ax.plot(omega /omega_max, current / stall_current, label=r'$\frac{i}{\bar{i}}$')

    power_electrical = motor.power_electrical_speed(v, omega)
    ax.plot(omega / omega_max, power_electrical / np.max(power_electrical), label=r'$\frac{\mathcal{P}_\text{e}}{\bar{\mathcal{P}}_\text{e}}$')

    power_mechanical = motor.power_mechanical_speed(v, omega)
    ax.plot(omega / omega_max, power_mechanical / np.max(power_electrical), label=r'$\frac{\mathcal{P}_\text{m}}{\bar{\mathcal{P}}_\text{e}}$')

    power_lost = power_electrical - power_mechanical
    ax.plot(omega / omega_max, power_lost / np.max(power_electrical), label=r'$\frac{\mathcal{P}_\text{e} - \mathcal{P}_\text{m}}{\bar{\mathcal{P}}_\text{e}}$')

    efficiency = motor.efficiency_speed(v, omega)
    ax.plot(omega / omega_max, efficiency, label=r'$\eta_\text{m} = \frac{\mathcal{P}_\text{m}}{\mathcal{P}_\text{e}}$')

    ax.set_xlabel(r'$\frac{\omega}{\bar{\omega}}$')
    ax.legend()

    plt.show()


    fig, ax = plt.subplots()

    torque = np.linspace(1e-9, stall_torque, 100)

    speed = motor.speed(v, torque)
    ax.plot(torque / stall_torque, speed / omega_max, label=r'$\frac{\omega}{\bar{\omega}}$')

    current = motor.current_torque(v, torque)
    ax.plot(torque / stall_torque, current / stall_current, label=r'$\frac{i}{\bar{i}}$')

    power_electrical = motor.power_electrical_torque(v, torque)
    ax.plot(torque / stall_torque, power_electrical / np.max(power_electrical), label=r'$\frac{\mathcal{P}_\text{e}}{\bar{\mathcal{P}}_\text{e}}$')

    power_mechanical = motor.power_mechanical_torque(v, torque)
    ax.plot(torque / stall_torque, power_mechanical / np.max(power_electrical), label=r'$\frac{\mathcal{P}_\text{m}}{\bar{\mathcal{P}}_\text{e}}$')

    power_lost = power_electrical - power_mechanical
    ax.plot(torque / stall_torque, power_lost / np.max(power_electrical), label=r'$\frac{\mathcal{P}_\text{e} - \mathcal{P}_\text{m}}{\bar{\mathcal{P}}_\text{e}}$')

    efficiency = power_mechanical / power_electrical
    ax.plot(torque / stall_torque, efficiency, label=r'$\eta_\text{m} = \frac{\mathcal{P}_\text{m}}{\mathcal{P}_\text{e}}$')

    ax.set_xlabel(r'$\frac{\tau}{\bar{\tau}}$')

    ax.legend()
    plt.show()



