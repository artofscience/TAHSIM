from utils import Sigmoid
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

class RLCircuitRef:
    def __init__(self, voltage):
        self.voltage = voltage
        self.resistance: float = 1.0
        self.inductance: float = 1.0

    def __call__(self, t, y):
        current = y
        return ( self.voltage(t) - self.resistance * current ) / self.inductance

voltage = lambda t: Sigmoid(1, 1.0, k=5)(t)
circuit = RLCircuitRef(voltage)
sol = solve_ivp(circuit, t_span=[0, 10], y0=[0], rtol=1e-9, atol=1e-9)
dsol = circuit(sol.t, sol.y)

time = sol.t
current = sol.y[0]
dcurrent = dsol[0]

# voltage
plt.figure()
plt.plot(time, voltage(time), label="Voltage")
plt.plot(time, circuit.resistance * current, label="Resistance")
plt.plot(time, circuit.inductance * dcurrent, label="Inductance")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")

# current
plt.figure()
plt.plot(time, current, label="Current")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.show()

