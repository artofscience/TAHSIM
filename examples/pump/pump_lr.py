import numpy as np
from math import pi

from matplotlib import pyplot as plt

from utils import colored_line, Sigmoid

from pumps import CentrifugalPump, MotorPumpLoadAssembly
from motors import DCMotor
from circuits import NLRLCircuit

from examples.helper_functions.plot_pump_props import plot_pump_props

pump = CentrifugalPump()
motor = DCMotor(R=0.2, L=0.11/10)

# setup time-dependent voltage
voltage = lambda t: Sigmoid(1.5, 1.0)(t) #+ Sigmoid(0.1, 7)(t) * np.sin(2 * 2 * pi * t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = lambda t: 0.5 + Sigmoid(2, 3.0)(t) + Sigmoid(1.5, 5.0)(t) * np.sin(2 * 2 * pi * t)

circuit = NLRLCircuit(resistance)

system = MotorPumpLoadAssembly(motor, pump, circuit)

# solve system
y0 = (0.0, 0.5, 0.01) # current, speed, flow
time, sol = system(y0)
derivatives = system.solve(time, sol)

VA = voltage(time)

current = sol[0]
speed = sol[1]
flow_pump = sol[2]
TL = sol[3]
HP = sol[4]

dcurrent = derivatives[0]
dspeed = derivatives[1]
dflow_pump = derivatives[2]


fig, ax = plt.subplots()
line = colored_line(current, voltage(time), time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)
plt.xlim([0.0, 3])
plt.ylim([0.0, 3])
ax.set_xlabel('Current I [A]')
ax.set_ylabel('Voltage V [V]')

fig, ax = plt.subplots()
line = colored_line(speed, 1e3 * TL, time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)
plt.xlim([0.0, 1.5 * pump.w0])
plt.ylim([0.0, 10])
ax.set_xlabel('Shaft speed w [rad/s]')
ax.set_ylabel('Load torque [Nmm]')


fig, ax = plt.subplots()
plot_pump_props(pump)
line = colored_line(flow_pump, HP, time, ax, linewidth=10, cmap="hsv")
fig.colorbar(line)
ax.set_xlabel('Flow rate Q [L/min]')
ax.set_ylabel('Pressure head h [m]')

plt.figure()
plt.title('Voltage')
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')

VR = motor.R * current
VE = motor.kb * speed
VL = motor.L * dcurrent

plt.plot(time, VA, label="Applied")
plt.plot(time, VR, label="Resistance")
plt.plot(time, VE, label="Back-EMF")
plt.plot(time, VL, label="Inductance")
plt.legend()

plt.figure()
plt.title('Torque')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nmm]')

TM = motor.kt * current
TR = motor.mu * speed
TI = motor.M * dspeed

plt.plot(time, 1e3 * TL, label="Pump")
plt.plot(time, 1e3 * TM, label="Motor")
plt.plot(time, 1e3 * TR, label="Viscous friction")
plt.plot(time, 1e3 * TI, label="Inertia")
plt.legend()

plt.figure()
plt.title('Pressure')
plt.xlabel('Time [s]')
plt.ylabel('Pressure head [m]')

HR = circuit.h(time, flow_pump)
HI = circuit.impedance * dflow_pump

plt.plot(time, HP, label="Pump")
plt.plot(time, HR, label="Resistance")
plt.plot(time, HI, label="Impedance")
plt.legend()

plt.figure()
plt.title('Power')
plt.plot(time, VA * current, label='Electrical input')
plt.plot(time, VR * current, label="Electrical resistance")
plt.plot(time, VL * current, label="Electrical inductance")
plt.plot(time, VE * current, label="Electrical Back-EMF")

plt.plot(time, TL * speed, label='Mechanical load power')
plt.plot(time, TR * speed, label="Mechanical friction")
plt.plot(time, TI * speed, label="Mechanical inertia")
plt.plot(time, TM * speed, "--", label="Mechanical motor power")

plt.plot(time, HP * pump.gamma * flow_pump / 60000, label='Pump hydraulic power')
plt.plot(time, HR * pump.gamma * flow_pump / 60000, label='Resistor hydraulic power')
plt.plot(time, HI * pump.gamma * flow_pump / 60000, label='Impedance hydraulic power')

plt.xlabel('t [s]')
plt.ylabel('P [W]')
plt.legend()

plt.show()

