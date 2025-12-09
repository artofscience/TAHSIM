import numpy as np
from scipy.integrate import solve_ivp

from motors import DCMotor
from pumps import CentrifugalPump
from inverters import RLCCircuit

class MotorPumpLoadAssembly:
    def __init__(self,
                 motor: DCMotor = DCMotor(),
                 pump: CentrifugalPump = CentrifugalPump(),
                 inverter: RLCCircuit = RLCCircuit()):
        self.motor = motor
        self.pump = pump
        self.inverter = inverter

    def __call__(self,
                 y0: tuple[float, ...],
                 t: float = 1.0,
                 t_begin: float = 0.0,
                 atol: float = 1e-3, rtol: float = 1e-3):
        sol = solve_ivp(self.solve, [t_begin, t], y0, atol=atol, rtol=rtol)
        z = [self.pump.solve(sol.t[i], y) for i, y in enumerate(sol.y[1:3].T)]
        return sol.t, np.vstack(([sol.y, np.asarray(z).T]))

    def solve(self, t, y):
        tau, h_pump = self.pump.solve(t, y[1:3]) # in kPa
        return self.ode(t, y, tau, h_pump)

    def ode(self, t, y, tau, h_pump):
        dI, dw = self.motor.solve_tau(t, y[0], y[1], tau)
        dq, dh = self.inverter.solve(t, h_pump, y[2], y[3])
        return [dI, dw, dq, dh]