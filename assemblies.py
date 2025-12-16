import numpy as np
from scipy.integrate import solve_ivp

from motors import DCMotor
from pumps import CentrifugalPump
from circuits import Circuit, RLCCircuit

class MotorPumpLoadAssembly:
    def __init__(self,
                 motor: DCMotor = DCMotor(),
                 pump: CentrifugalPump = CentrifugalPump(),
                 circuit: Circuit = RLCCircuit()):
        self.motor = motor
        self.pump = pump
        self.circuit = circuit

    def __call__(self,
                 y0: tuple[float, ...],
                 t: float = 10.0,
                 t_begin: float = 0.0,
                 atol: float = 1e-6, rtol: float = 1e-6):
        sol = solve_ivp(self.solve, [t_begin, t], y0, atol=atol, rtol=rtol)
        z = [self.pump.solve(sol.t[i], y) for i, y in enumerate(sol.y[1:3].T)]
        return sol.t, np.vstack(([sol.y, np.asarray(z).T]))

    def solve(self, t, y):
        tau, h_pump = self.pump.solve(t, y[1:3]) # in kPa
        return self.ode(t, y, tau, h_pump)

    def ode(self, t, y, tau, h_pump):
        dmotor = self.motor.solve_tau(t, y[0], y[1], tau)
        dcircuit = self.circuit.solve(t, h_pump, y[2:])
        return dmotor + dcircuit # combinging two tuples