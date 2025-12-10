import numpy as np
from matplotlib import pyplot as plt
from math import pi

from utils import colored_line, Sigmoid

from pumps import CentrifugalPump
from motors import DCMotor
from circuits import RLCCircuit, NLRLCircuit
from assemblies import MotorPumpLoadAssembly, MotorPumpLoadAssembly0

from helper_functions.plot_pump_props import plot_pump_props

pump = CentrifugalPump(hm0=2.4, hn0=1.8, qn0=1.6/60000, qm0=3.2/60000)
motor = DCMotor()

# setup time-dependent voltage
voltage = lambda t: Sigmoid(3.5, 10, 1.0)(t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = lambda t: 3e9 * np.ones_like(t)

circuit = NLRLCircuit(resistance, 100)

system = MotorPumpLoadAssembly0(motor, pump, circuit)

fig, ax = plt.subplots()
# plot pump characteristics
plot_pump_props(pump)

# solve system
y0 = (0.0, 1.0, 1e-9) # current, speed, flow
time, sol = system(y0, t=5.0)

line = colored_line(sol[2]*pump.cubpstolmin, sol[4], time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)

fig, ax = plt.subplots(4,3)

ax[0,0].plot(time, voltage(time), label='V')
ax[0,1].plot(time, resistance(time), label='R')
ax[1,0].plot(time, sol[0], label='I')
ax[1,1].plot(time, sol[1] * pump.radpstorpm, label="w")
ax[1,2].plot(time, sol[2] * pump.cubpstolmin, label='Pump flow rate [L/min]')
ax[2,0].plot(time, sol[3], label='Torque')
ax[2,1].plot(time, sol[4], label='Pump head [m]')
ax[3,0].plot(time, voltage(time) * sol[0], label='Electrical power')
ax[3,1].plot(time, sol[3] * sol[1], label='Mechanical power')
ax[3,2].plot(time, sol[2] * sol[4] * pump.gamma, label='Hydraulic power')


for i in [0, 1, 2, 3]:
    for j in [0, 1, 2]:
        ax[i,j].legend()
        for t in [1, 2, 3, 4]:
            ax[i, j].axvline(x=t, color="red", linestyle="dotted")

plt.show()

