import numpy as np
from math import pi

from matplotlib import pyplot as plt

from utils import colored_line, Sigmoid

from pumps import CentrifugalPump
from motors import DCMotor
from circuits import NLRLCircuit
from assemblies import MotorPumpLoadAssembly0

from helper_functions.plot_pump_props import plot_pump_props

pump = CentrifugalPump(hm0=2.4, hn0=1.8, qn0=1.6, qm0=3.2)
motor = DCMotor()

# setup time-dependent voltage
voltage = lambda t: Sigmoid(1.5, 30, 1.0)(t) + Sigmoid(0.1, 5, 4)(t) * np.sin(2 * 2*pi * t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = lambda t: 0.5 + Sigmoid(2, 30, 2.0)(t) + Sigmoid(1, 50, 3.0)(t) * np.sin(10 * 2*pi * t)

circuit = NLRLCircuit(resistance, 0.05)

system = MotorPumpLoadAssembly0(motor, pump, circuit)

# solve system
y0 = (0.0, 1.0, 0.1) # current, speed, flow
time, sol = system(y0, t=7.0, atol=1e-6, rtol=1e-6) # sol =
derivatives = system.solve(time, sol)

fig, ax = plt.subplots()
line = colored_line(sol[0], voltage(time), time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)
plt.xlim([0.0, 3])
plt.ylim([0.0, 3])
ax.set_xlabel('Current I [A]')
ax.set_ylabel('Voltage V [V]')

fig, ax = plt.subplots()
line = colored_line(sol[1], 1e3 * sol[3], time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)
plt.xlim([0.0, 1.5 * pump.w0])
plt.ylim([0.0, 10])
ax.set_xlabel('Shaft speed w [rad/s]')
ax.set_ylabel('Load torque [Nmm]')


fig, ax = plt.subplots()
plot_pump_props(pump)
line = colored_line(sol[2], sol[4], time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)
ax.set_xlabel('Flow rate Q [L/min]')
ax.set_ylabel('Pressure head h [m]')

plt.figure()
plt.title('Voltage')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')

VA = voltage(time)
VR = motor.R * sol[0]
VE = motor.kb * sol[1]
VL = motor.L * derivatives[0]

plt.plot(time, VA, label="Applied")
plt.plot(time, VR, label="Resistance")
plt.plot(time, VE, label="Back-EMF")
plt.plot(time, VL, label="Inductance")
plt.legend()

plt.figure()
plt.title('Torque')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nmm]')

TL = sol[3]
TM = motor.kt * sol[0]
TR = motor.mu * sol[1]
TI = motor.M * derivatives[1]

plt.plot(time, 1e3 * TL, label="Pump")
plt.plot(time, 1e3 * TM, label="Motor")
plt.plot(time, 1e3 * TR, label="Viscous friction")
plt.plot(time, 1e3 * TI, label="Inertia")
plt.legend()

plt.figure()
plt.title('Pressure')
plt.xlabel('Time [s]')
plt.ylabel('Pressure head [m]')

HP = sol[4]
HR = circuit.h(time, sol[2])
HI = circuit.inductance * derivatives[2]

plt.plot(time, HP, label="Pump")
plt.plot(time, HR, label="Resistance")
plt.plot(time, HI, label="Impedance")
plt.legend()

plt.figure()
plt.title('Power')
plt.plot(time, VA * sol[0], label='Electrical input')
plt.plot(time, VR * sol[0], label="Electrical resistance")
plt.plot(time, VL * sol[0], label="Electrical inductance")
plt.plot(time, VE * sol[0], label="Electrical Back-EMF")

plt.plot(time, TL * sol[1], label='Mechanical load power')
plt.plot(time, TR * sol[1], label="Mechanical friction")
plt.plot(time, TI * sol[1], label="Mechanical inertia")
plt.plot(time, TM * sol[1], "--", label="Mechanical motor power")

plt.plot(time, HP * pump.gamma * sol[2] / 60000, label='Pump hydraulic power')
plt.plot(time, HR * pump.gamma * sol[2] / 60000, label='Resistor hydraulic power')
plt.plot(time, HI * pump.gamma * sol[2] / 60000, label='Impedance hydraulic power')

plt.xlabel('t [s]')
plt.ylabel('P [W]')
plt.legend()

plt.show()

