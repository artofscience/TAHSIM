import numpy as np
from math import pi
from matplotlib import pyplot as plt

from motors import DCMotor


motor = DCMotor() # setup DC-Motor instance

speed = np.linspace(0, 1000, 100)
torpm = 60 / (2 * pi) / 1000
tomilli = 1e3
rpm = [i * torpm for i in speed]
V = 6.0

no_load_speed = motor.max_speed(V)
stall_torque = motor.stall_torque(V)
max_power = 0.25 * stall_torque * no_load_speed
max_power_speed = motor.max_power_speed(V)
efficient_speed = motor.efficient_speed(V)

torque = motor.torque(V, speed)
electrical_power = V * motor.current(V, speed)
mechanical_power = speed * motor.torque(V, speed)
efficiency = motor.efficiency(V, speed)

plt.figure()

plt.subplot(2, 1, 1)

plt.plot(rpm, torque * tomilli)

plt.xlim([0, no_load_speed * torpm])
plt.ylim([0, stall_torque * tomilli])

plt.xlabel('kRPM')
plt.ylabel('Torque [mN x m]')

plt.subplot(2, 1, 2)

plt.plot(rpm, electrical_power, label='Electrical')
plt.plot(rpm, mechanical_power, label='Mechanical')
plt.axhline(y=max_power, linestyle='--', color='k', label='Max mechanical power')
plt.axvline(x=max_power_speed * torpm, linestyle='--', color='k', label='Speed at max mechanical power')
plt.plot(rpm, electrical_power - mechanical_power, label='Lost')
plt.plot(rpm, max_power * efficiency, label='Efficiency x max mechanical power')
plt.axvline(x=efficient_speed * torpm, linestyle='--', color='k', label='Speed at max efficiency')

plt.xlim([0, no_load_speed * torpm])
plt.gca().set_ylim(bottom=0)
plt.legend()

plt.xlabel('kRPM')
plt.ylabel('Power [W]')

plt.show()


