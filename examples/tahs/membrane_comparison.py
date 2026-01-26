from tahs import PressureActuatedLinearMembrane, PressureActuatedNonlinearMembrane
import numpy as np
from matplotlib import pyplot as plt

heartlin = PressureActuatedLinearMembrane(Pact=lambda t: t, E=0.45, V0=80)

heartnonlin = PressureActuatedNonlinearMembrane(Pact=lambda t: t, V0=80)

dv = np.linspace(-19, 79, 100)
dp = heartlin.fcn(dv)

plt.figure()
plt.plot(dv, heartlin.fcn(dv))
plt.plot(dv, heartnonlin.fcn(dv))
plt.xlabel(r'$\Delta v = v_{\text{v},0} - v_\text{v} = v_\text{a} - v_{\text{a}, 0}$')
plt.ylabel(r'$\Delta P = P - P_\text{v}$')
plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.figure()
P = np.linspace(0, 50, 10)
vc = np.linspace(0.01, 99, 100)

for p in P:
    plt.plot(vc, heartlin.pressure(vc, p))

plt.xlim([0, 100])
plt.ylim([-10, 30])
plt.axhline(0, color='k')
plt.axvline(80, color='k')

plt.xlabel(r'$v_\text{v} = v_{\text{v},0} - \Delta v$')
plt.ylabel(r'$P_\text{v} = P - \Delta P\left[\Delta v\right]$')


plt.figure()

for p in P:
    plt.plot(vc, heartnonlin.pressure(vc, p))

plt.xlim([0, 100])
plt.ylim([-10, 30])
plt.axhline(0, color='k')
plt.axvline(80, color='k')

plt.xlabel(r'$v_\text{v} = v_{\text{v},0} - \Delta v$')
plt.ylabel(r'$P_\text{v} = P - \Delta P\left[\Delta v\right]$')

plt.show()