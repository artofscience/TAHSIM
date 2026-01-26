from tahs import PressureActuatedLinearMembrane
import numpy as np
from matplotlib import pyplot as plt


heart = PressureActuatedLinearMembrane(Pact=lambda t: t, E=0.4, V0=80)

dv = np.linspace(-19, 79, 100)
dp = heart.fcn(dv)

plt.figure()
plt.plot(dv, dp)
plt.xlabel(r'$\Delta v = v_{\text{v},0} - v_\text{v} = v_\text{a} - v_{\text{a}, 0}$')
plt.ylabel(r'$\Delta P = P - P_\text{v}$')
plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.figure()
P = np.linspace(0, 50, 10)
vc = np.linspace(0.01, 99, 100)

for p in P:
    plt.plot(vc, heart.pressure(vc, p))

plt.xlim([0, 100])
plt.ylim([-10, 30])
plt.axhline(0, color='k')
plt.axvline(80, color='k')

plt.xlabel(r'$v_\text{v} = v_{\text{v},0} - \Delta v$')
plt.ylabel(r'$P_\text{v} = P - \Delta P\left[\Delta v\right]$')

plt.show()