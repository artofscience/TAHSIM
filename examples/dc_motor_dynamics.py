import numpy as np
from matplotlib import pyplot as plt
from math import pi
from utils import Sigmoid, colored_line

from motors import DCMotor

voltage = lambda t: Sigmoid(6.0, 20, 0.3)(t) + Sigmoid(1.0, 20, 1.0)(t) * np.sin(2 * (2*pi) * t)# step in voltage at t = tV
torque = lambda t, w: + Sigmoid(1e-3, 20, 2.0)(t) * np.sin(6 * (2*pi) * t) + 1e-9 * np.power(w, 2)# time-dependent torque

motor = DCMotor(voltage, torque)

sol = motor(t=4.0)

torpm = 60 / (2 * pi)
tomilli = 1e3

plt.figure()

time = sol.t
voltage = voltage(time)
current = sol.y[0]
torque = torque(time, sol.y[1] * tomilli)
speed = sol.y[1] * torpm / tomilli

plt.subplot(4, 1, 1)
plt.plot(time, voltage)
plt.ylabel('Voltage [V]')

plt.subplot(4, 1, 2)
plt.plot(time, current)
plt.ylabel('Current [A]')

plt.subplot(4, 1, 3)
plt.plot(time, torque)
plt.ylabel('Torque [Nmm]')

plt.subplot(4, 1, 4)
plt.plot(time, speed)
plt.ylabel('Speed [krpm]')

plt.figure()
colored_line(voltage, current, time, plt.gca())
plt.ylabel('Current [A]')
plt.xlabel('Voltage [V]')
plt.xlim([0, max(voltage)])
plt.ylim([0, max(current)])

plt.figure()
colored_line(speed, torque, time, plt.gca())
plt.ylabel('Torque [Nmm]')
plt.xlabel('Speed [krpm]')
plt.xlim([0, max(speed)])
plt.ylim([0, max(torque)])

plt.show()



