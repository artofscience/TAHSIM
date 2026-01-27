import numpy as np
from matplotlib import pyplot as plt
from math import pi
from utils import Sigmoid, colored_line

from motors import DCMotor

# voltage: step of 6V at t=0.3 and step of sine at t=1
voltage = lambda t: Sigmoid(6.0, 0.5, 20)(t) + Sigmoid(1.0, 1.5, 20)(t) * np.sin(2 * (2 * pi) * t) - Sigmoid(1.0, 2.5, 20)(t) * np.sin(2 * (2 * pi) * t)

# torque: quadratic dependence on shaft speed and step of sine at t=2
torque = lambda t, w: Sigmoid(5e-4, 3.5, 20)(t) +  1e-9 * np.power(w, 2)

# setup DC Motor (Maxxon Amax 22, 5 Watt motor) with voltage and torque functions
motor = DCMotor(voltage, torque)

# solve for t=0 to t=4
sol = motor(t=4.5)

# POSTPROC

time = sol.t # time
voltage = voltage(time) # voltage
current = sol.y[0] # current
torque = torque(time, sol.y[1]) * 1000 # torque
speed = sol.y[1] * motor.radpstorpm / 1000 # speed

plt.figure()

plt.subplot(4, 1, 1)
plt.plot(time, voltage, label='v')
plt.plot(time, motor.kb * sol.y[1], label='kb x w')
plt.plot(time, motor.R * current, label='r x i')
plt.plot(time, voltage - motor.R * current - motor.kb * sol.y[1], label='l x didt')
plt.ylabel('Voltage [V]')
plt.legend()
[plt.axvline(x, color='k', linestyle='--') for x in [0.5, 1.5, 2.5]]

plt.subplot(4, 1, 2)
plt.plot(time, current)
plt.ylabel('Current [A]')

plt.subplot(4, 1, 3)
plt.plot(time, torque, label='t')
plt.plot(time, motor.kt * current * 1000, label='kt x i')
plt.plot(time, motor.mu * sol.y[1] * 1000, label='mu x w')
plt.plot(time, 1000 * (motor.kt * current - motor.mu * sol.y[1]) - torque, label='j x dwdt')
plt.ylabel('Torque [Nmm]')
plt.legend()
plt.axvline(3.5, color='k', linestyle='--')

plt.subplot(4, 1, 4)
plt.plot(time, speed)
plt.ylabel('Speed [krpm]')

plt.figure()
plt.subplot(2,1,1)
im = colored_line(current, voltage, time, plt.gca(), cmap='jet')
plt.xlabel('Current [A]')
plt.ylabel('Voltage [V]')
plt.ylim([0, max(voltage)])
plt.xlim([0, max(current)])
plt.colorbar(im)

voltage = np.linspace(0, max(voltage))
current = np.linspace(0, max(current))
V, I = np.meshgrid(voltage, current)
im2 = plt.contourf(I, V, I * V)
plt.clim(0, voltage[-1] * current[-1])
plt.colorbar(im2)

plt.subplot(2,1,2)
im = colored_line(speed, torque, time, plt.gca(), cmap='jet')
plt.ylabel('Torque [Nmm]')
plt.xlabel('Speed [krpm]')
plt.xlim([0, max(speed)])
plt.ylim([0, max(torque)])
plt.colorbar(im)

speed = np.linspace(0, max(speed))
torque = np.linspace(0, max(torque))
T, W = np.meshgrid(torque, speed)
plt.plot(speed, motor.torque(6, speed * 1000 * motor.rpmtoradps) * 1000, 'k--', label='tau @ 6V')

im2 = plt.contourf(W, T, T * W / motor.radpstorpm)
plt.clim(0, voltage[-1] * current[-1])
plt.colorbar(im2)


plt.show()



