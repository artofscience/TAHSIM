from matplotlib import pyplot as plt
import numpy as np
from utils import colored_line, Sigmoid

from pumps import CentrifugalPump, MotorPumpLoadAssembly
from motors import DCMotor
from circuits import RLCCircuit, Oscillator

from examples.helper_functions.plot_pump_props import plot_pump_props

pump = CentrifugalPump()
motor = DCMotor(R=0.2, L=0.11/10)

# setup time-dependent voltage
voltage = lambda t: Sigmoid(1.5, 0.5)(t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = Oscillator(0.1, 1000)
capacitance = lambda t: 0.4 #+ Sigmoid(1, 4)(t)

circuit = RLCCircuit(resistance, capacitance)

system = MotorPumpLoadAssembly(motor, pump, circuit)

# solve system
y0 = (0.0, 1.0, 1e-6, 0.0) # current, speed, flow, circuit head
time, sol = system(y0, 3)
resistance.dhopen -= 0.01
resistance.dhclose += 0.01

derivatives = np.zeros((4, len(time)))
for i in range(0, len(time)):
    derivatives[:, i] = system.solve(time[i], sol[:, i])

VA = voltage(time)

current = sol[0]
speed = sol[1]
QP = sol[2]
HR = sol[3]
TL = sol[4]
HP = sol[5]


QR = np.asarray([hr/resistance(0, hr) for hr in HR])
QC = QP - QR
#
dcurrent = derivatives[0]
dspeed = derivatives[1]
dflow_pump = derivatives[2]
dHR = derivatives[3]

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
qx = np.linspace(-20, 20, 100)
hy = np.linspace(0,3.5, 100)
QX, HY = np.meshgrid(qx, hy)
plt.contourf(QX, HY, QX * HY *pump.gamma / 60000)
plt.colorbar()
ax.plot(QP, HP, 'k-', linewidth=6)
colored_line(QP, HP, time, ax, linewidth=3, cmap="hsv")
ax.plot(QR, HR, 'r-', linewidth=6)
line = colored_line(QR, HR, time, ax, linewidth=3, cmap="hsv")
ax.plot(QC, HR, 'y-', linewidth=6)
colored_line(QC, HR, time, ax, linewidth=3, cmap="hsv")
fig.colorbar(line)

ax.set_xlim([-20, 20])
ax.set_ylim([0, 3.5])
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

HI = circuit.impedance * dflow_pump

plt.plot(time, HP, label="Pump")
plt.plot(time, HR, label="Resistance")
plt.plot(time, HI, label="Impedance")
plt.legend()

plt.figure()
plt.title('Flow')
plt.xlabel('Time [s]')
plt.ylabel('Flow rate [L/min]')
plt.plot(time, QP, label="Pump")
plt.plot(time, QR, label="Flow resistor")
plt.plot(time, QC, label="Flow capacitor")
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

plt.plot(time, HP * pump.gamma * QP / 60000, label='Pump hydraulic power')
plt.plot(time, HR * pump.gamma * QR / 60000, label='Resistor hydraulic power')
plt.plot(time, HI * pump.gamma * QP / 60000, label='Impedance hydraulic power')
plt.plot(time, HR * pump.gamma * QC / 60000, label='Capacitor hydraulic power')


plt.xlabel('t [s]')
plt.ylabel('P [W]')
plt.legend()

plt.show()

