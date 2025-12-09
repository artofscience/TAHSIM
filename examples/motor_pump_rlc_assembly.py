import numpy as np
from matplotlib import pyplot as plt
from math import pi

from utils import colored_line, Sigmoid

from pumps import CentrifugalPump
from motors import DCMotor
from inverters import RLCCircuit
from assemblies import MotorPumpLoadAssembly

from helper_functions.plot_pump_props import plot_pump_props

pump = CentrifugalPump(hm0=24, hn0=18, qn0=16/60000, qm0=32/60000)
motor = DCMotor()

# setup time-dependent voltage
voltage = lambda t: Sigmoid(10, 50, 1.0)(t) + Sigmoid(2, 50, 3.0)(t) * np.sin(3 * 2*pi * t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = lambda t, h: 2e4 * (1 + Sigmoid(10, 50, 2.0)(t))
capacitance = lambda t: 2e-7 * (1 + Sigmoid(20, 50, 4.0)(t))
inductance = lambda t: 100

circuit = RLCCircuit(resistance, capacitance, inductance)

system = MotorPumpLoadAssembly(motor, pump, circuit)

fig, ax = plt.subplots()
# plot pump characteristics
plot_pump_props(pump)

# solve system
y0 = (0.0, 1.0, 0.0, 0.0)
time, sol = system(y0, t=5.0)

line = colored_line(sol[2]*pump.cubpstolmin, sol[5], time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)

fig, ax = plt.subplots(4,3)

ax[0 , 0].plot(time, voltage(time), label='V')
ax[0, 1].plot(time, resistance(time, 0), label='R')
ax[0, 2].plot(time, capacitance(time), label='C')
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

