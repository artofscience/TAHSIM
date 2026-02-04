from matplotlib import pyplot as plt
import numpy as np
from utils import colored_line, Sigmoid

from pumps import CentrifugalPump, MotorPumpLoadAssembly
from motors import DCMotor
from circuits import RLCRCCircuitCL, Oscillator

from examples.helper_functions.plot_pump_props import plot_pump_props

pump = CentrifugalPump()
motor = DCMotor(R=0.2, L=0.11/10)

# setup time-dependent voltage
voltage = lambda t: Sigmoid(1.5, 0.5)(t)
motor.set_voltage(voltage)

# setup time-dependent circuit parameters
resistance = Oscillator(0.1, 1000)
capacitance = lambda t: 1 - Sigmoid(0.5, 2)(t)

circuit = RLCRCCircuitCL(resistance, capacitance, Rout= lambda t: 0.5, Cac=lambda t: 1)

system = MotorPumpLoadAssembly(motor, pump, circuit)


# solve system
y0 = (0.0, 1.0, 1e-6, 2, 2) # current, speed, pump flow, circuit head, circuit head 2
time, sol = system(y0, 3)
resistance.dhopen -= 0.01
resistance.dhclose += 0.01


derivatives = np.zeros((5, len(time)))
for i in range(0, len(time)):
    derivatives[:, i] = system.solve(time[i], sol[:, i])

VA = voltage(time)

current = sol[0]
speed = sol[1]
QP = sol[2] # pump flow
HC1 = sol[3]
HC2 = sol[4]
TL = sol[5]
DHP = sol[6]


circuit.resistance.closed = True
QHV = np.asarray([(HC1[i] - HC2[i])/circuit.resistance(time[i], HC1[i] - HC2[i]) for i, hr in enumerate(HC2)])
QC1 = QP - QHV

# QR = HR / circuit.Rout(time)
QC2 = QHV - QP

dcurrent = derivatives[0]
dspeed = derivatives[1]
dflow_pump = derivatives[2]
dHC1 = derivatives[3]
dHC2 = derivatives[4]

plt.figure()
plt.title('Pressure')
plt.xlabel('Time [s]')
plt.ylabel('Pressure head [m]')

DHI = circuit.impedance * dflow_pump
DHR = circuit.Rout(time) * QP


plt.plot(time, DHP, label="Delta Pump")
plt.plot(time, HC1, label="Capacitor1")
plt.plot(time, HC2, label="Capacitor2")
plt.plot(time, HC1-HC2, label="Delta HV")
plt.plot(time, DHR, label="Delta Resistance")
plt.plot(time, DHI, label="Delta Impedance")
# plt.plot(time, DHP - DHI - DHR, label="Delta HV2")
[plt.axhline(value, color='k', linestyle='--') for value in [resistance.dhopen, resistance.dhclose]]
plt.legend()

plt.figure()
plt.title('Flow')
plt.xlabel('Time [s]')
plt.ylabel('Flow rate [L/min]')
plt.plot(time, QP, label="Flow Pump / Resistor / Impedance")
plt.plot(time, QC1, label="Flow capacitor1")
plt.plot(time, QC2, label="Flow capacitor2")
plt.plot(time, QHV, label="Flow HV")
plt.legend()

plt.figure()
plt.title('Volume')
plt.ylabel('Volume [mL]')
plt.xlabel('Time [s]')
plt.plot(time, circuit.capacitance(time) * HC1 * (100/6), label="Capacitor1")
plt.plot(time, circuit.Cac(time) * HC2 * (100/6), label="Capacitor2")
plt.legend()


plt.show()

