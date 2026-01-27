from matplotlib import pyplot as plt
import numpy as np

from utils import colored_line, Sigmoid

from pumps import CentrifugalPump
from motors import DCMotor
from circuits import RLCCircuit, Oscillator, RLCRCCircuit
from assemblies import MotorPumpLoadAssembly

from helper_functions.plot_pump_props import plot_pump_props

pump = CentrifugalPump()
motor = DCMotor()

# setup time-dependent voltage
voltage = lambda t: Sigmoid(1.5, 1.0)(t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = Oscillator(0.1, 10, dhopen=2, dhclose=0.5)
capacitance = lambda t: 0.1 #+ Sigmoid(1, 4)(t)

circuit = RLCRCCircuit(resistance, capacitance, impedance=0.01, Rout=lambda t: 1, Cac=lambda t: 0.1)

system = MotorPumpLoadAssembly(motor, pump, circuit)

# solve system
y0 = (0.0, 1.0, 0.1, 0.0, 0.0) # current, speed, flow, circuit head
time, sol = system(y0)
# derivatives = system.solve(time, sol)

VA = voltage(time)

current = sol[0]
speed = sol[1]
QP = sol[2]
HC = sol[3]
HR = sol[4]
TL = sol[5]
HP = sol[6]

circuit.resistance.closed = True
QHV = np.asarray([(HC[i] - HR[i])/circuit.resistance(time[i], HC[i] - HR[i]) for i, hr in enumerate(HR)])
QC = QP - QHV

QR = HR / circuit.Rout(time)
QAC = QHV - QR


# dcurrent = derivatives[0]
# dspeed = derivatives[1]
# dflow_pump = derivatives[2]
# dHR = derivatives[3]


# fig, ax = plt.subplots()
# line = colored_line(current, VA, time, ax, linewidth=10, cmap="hsv")
# fig.colorbar(line)
# plt.xlim([0.0, 3])
# plt.ylim([0.0, 3])
# ax.set_xlabel('Current I [A]')
# ax.set_ylabel('Voltage V [V]')
#
# fig, ax = plt.subplots()
# line = colored_line(speed, 1e3 * TL, time, ax, linewidth=10, cmap="hsv")
# fig.colorbar(line)
# plt.xlim([0.0, 1.5 * pump.w0])
# plt.ylim([0.0, 10])
# ax.set_xlabel('Shaft speed w [rad/s]')
# ax.set_ylabel('Load torque [Nmm]')
#
#
# fig, ax = plt.subplots()
# plot_pump_props(pump)
# line = colored_line(QP, HP, time, ax, linewidth=10, cmap="hsv")
# fig.colorbar(line)
# ax.set_xlabel('Flow rate Q [L/min]')
# ax.set_ylabel('Pressure head h [m]')
#
# plt.figure()
# plt.title('Voltage')
# plt.xlabel('Time [s]')
# plt.ylabel('Voltage [V]')
#
# VR = motor.R * current
# VE = motor.kb * speed
# # VL = motor.L * dcurrent
#
# plt.plot(time, VA, label="Applied")
# plt.plot(time, VR, label="Resistance")
# plt.plot(time, VE, label="Back-EMF")
# # plt.plot(time, VL, label="Inductance")
# plt.legend()
#
# plt.figure()
# plt.title('Torque')
# plt.xlabel('Time [s]')
# plt.ylabel('Torque [Nmm]')
#
# TM = motor.kt * current
# TR = motor.mu * speed
# # TI = motor.M * dspeed
#
# plt.plot(time, 1e3 * TL, label="Pump")
# plt.plot(time, 1e3 * TM, label="Motor")
# plt.plot(time, 1e3 * TR, label="Viscous friction")
# # plt.plot(time, 1e3 * TI, label="Inertia")
# plt.legend()
#
plt.figure()
plt.title('Pressure')
plt.xlabel('Time [s]')
plt.ylabel('Pressure head [m]')

# HI = circuit.impedance * dflow_pump

plt.plot(time, HP, label="Pump")
plt.plot(time, HC, label="Capacitor")
plt.plot(time, HR, label="Resistance")
# plt.plot(time, HI, label="Impedance")
plt.legend()

plt.figure()
plt.title('Flow')
plt.xlabel('Time [s]')
plt.ylabel('Flow rate [L/min]')
plt.plot(time, QP, label="Pump")
plt.plot(time, QR, label="Flow resistor")
plt.plot(time, QC, label="Flow capacitor")
plt.plot(time, QHV, label="Oscillator")
plt.plot(time, QAC, label="Actuator")
plt.legend()
#
# plt.figure()
# plt.title('Power')
# plt.plot(time, VA * current, label='Electrical input')
# plt.plot(time, VR * current, label="Electrical resistance")
# # plt.plot(time, VL * current, label="Electrical inductance")
# plt.plot(time, VE * current, label="Electrical Back-EMF")
#
# plt.plot(time, TL * speed, label='Mechanical load power')
# plt.plot(time, TR * speed, label="Mechanical friction")
# # plt.plot(time, TI * speed, label="Mechanical inertia")
# plt.plot(time, TM * speed, "--", label="Mechanical motor power")
#
# plt.plot(time, HP * pump.gamma * QP / 60000, label='Pump hydraulic power')
# plt.plot(time, HR * pump.gamma * QP / 60000, label='Resistor hydraulic power')
# # plt.plot(time, HI * pump.gamma * flow_pump / 60000, label='Impedance hydraulic power')
#
# plt.xlabel('t [s]')
# plt.ylabel('P [W]')
# plt.legend()

plt.show()

