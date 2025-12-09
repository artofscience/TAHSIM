import numpy as np
from matplotlib import pyplot as plt
from math import pi

from utils import colored_line, Sigmoid

from pumps import CentrifugalPump
from motors import DCMotor
from inverters import RLCCircuit
from assemblies import MotorPumpLoadAssembly

def plot_pump_props(pump):
    q0, h0 = pump.get_operating_points(1000)

    z = np.linspace(0.1, 2, 10)
    w = pump.w0 * z

    for i in w:
        zi = i / pump.w0
        q = zi * q0
        hi = pump.hq(q, i)
        plt.plot(q * pump.cubpstolmin, hi, color="black", label=f"w/w0 = {zi:.2f}")

    plt.scatter(pump.q0p * pump.cubpstolmin, pump.h0p, color="blue")
    plt.plot(q0 * pump.cubpstolmin, pump.hq0(q0), color="blue", linewidth=4)

    plt.xlim([0.0, 1.5 * pump.q0p[-1] * pump.cubpstolmin])
    plt.ylim([0.0, 2.5 * pump.h0p[0]])

    plt.legend()


class Oscillator:
    def __init__(self, Ropen: float = 1e4, Rclosed: float = 1e8):
        self.Ropen = Ropen
        self.Rclosed = Rclosed
        self.closed = True
        self.dhopen = 50
        self.dhclose = 10

    def __call__(self, t, dh):

        if dh > self.dhopen:
            self.closed = False
            return self.Ropen
        elif dh < self.dhclose:
            self.closed = True
            return self.Rclosed
        else:
            return self.Rclosed if self.closed else self.Ropen


pump = CentrifugalPump(hm0=24, hn0=18, qn0=16/60000, qm0=32/60000)
motor = DCMotor()

# setup time-dependent voltage
voltage = lambda t: Sigmoid(10, 10, 1.0)(t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = Oscillator(1e3, 1e10)
capacitance = lambda t: 5e-6
inductance = lambda t: 10

circuit = RLCCircuit(resistance, capacitance, inductance)

system = MotorPumpLoadAssembly(motor, pump, circuit)

fig, ax = plt.subplots()
# plot pump characteristics
plot_pump_props(pump)

# solve system
y0 = (0.0, 1.0, 0.0, 0.0)
time, sol = system(y0, t=10.0)

line = colored_line(sol[2]*pump.cubpstolmin, sol[5], time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)

fig, ax = plt.subplots(4,3)

ax[0 , 0].plot(time, voltage(time), label='V')
# ax[0, 1].plot(time, resistance(time, time), label='R')
# ax[0, 2].plot(time, capacitance(time), label='C')
ax[1,0].plot(time, sol[0], label='I')
ax[1,1].plot(time, sol[1] * pump.radpstorpm, label="w")
ax[1,2].plot(time, sol[2] * pump.cubpstolmin, label='Capacity [L/min]')
ax[2,0].plot(time, sol[4], label='Torque')
ax[2,1].plot(time, sol[5], label='Pump head [m]')
ax[2,1].plot(time, sol[3], label='Circuit head [m]')
ax[2,2].plot(time, circuit.C(time) * sol[3], label='Volume')
ax[3,0].plot(time, voltage(time) * sol[0], label='Electrical power')
ax[3,1].plot(time, sol[4] * sol[1], label='Mechanical power')
ax[3,2].plot(time, sol[2] * sol[5] * pump.gamma, label='Hydraulic power')


for i in [0, 1, 2, 3]:
    for j in [0, 1, 2]:
        ax[i,j].legend()
        for t in [1, 2, 3, 4]:
            ax[i, j].axvline(x=t, color="red", linestyle="dotted")

plt.show()

