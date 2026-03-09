from utils import Sigmoid
from math import pi
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np

class PI_RL_Circuit:
    def __init__(self, reference_current= lambda t: 0.0 * t + 1.0):
        """
        RL-circuit: L di/dt = V(t) - R i
        PI-control: V(t) = kp * error + ki * time_integral(error), w/ error = ref(t) - i

        Say state variables:
        x1 = i
        x2 = time_integral(error)

        then

        L dx1/dt = V(t) - R x1 = kp * (ref(t) - x1) + ki * x2 - R x1
        dx2/dt = error = ref(t) - x1

        """
        self.reference_current = reference_current
        self.resistance: float = 1
        self.inductance: float = 1
        self.kp: float = 10
        self.ki: float = 10

    def __call__(self, t, y):
        current, int_error = y
        error = self.reference_current(t) - current
        voltage = self.kp * error + self.ki * int_error
        return (voltage - self.resistance * current) / self.inductance, error

reference_current = lambda t: Sigmoid(1, 1.0, k=5)(t) #+ Sigmoid(0.5, 3, k=10)(t) * np.sin(5*t)
circuit = PI_RL_Circuit(reference_current)
sol = solve_ivp(circuit, t_span=[0, 10], y0=[0, 0], rtol=1e-9, atol=1e-9)
dsol = circuit(sol.t, sol.y)

time = sol.t
current, int_error = sol.y
dcurrent, error = dsol

voltage = circuit.kp * error + circuit.ki * int_error

# voltage
plt.figure()
plt.plot(time, voltage, label='voltage')
plt.plot(time, circuit.resistance * current, label="Resistance")
plt.plot(time, circuit.inductance * dcurrent, label="Inductance")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# current
plt.figure()
plt.plot(time, reference_current(time), label='reference current')
plt.plot(time, current, label="current")
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.legend()
plt.show()
