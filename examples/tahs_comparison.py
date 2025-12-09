import numpy as np
from matplotlib import pyplot as plt
from utils import TDP


from tahs import PressureActuatedLinearMembrane, PressureActuatedNonlinearMembrane, TimeVaryingElastance

V0 = 20

actuation = TDP(min=0, max=120)
PALM = PressureActuatedLinearMembrane(actuation, E=0.1, V0=V0)
PANLM = PressureActuatedNonlinearMembrane(actuation, V0=V0)
TVE = TimeVaryingElastance(TDP(min=0.06, max=2.31), V0=V0)

plt.axhline(0, ls='--', color='k')
plt.axvline(0, ls='--', color='k')
plt.axvline(V0, ls='--', color='k')
volume = np.linspace(0.1, 99.9, 100)
time = np.linspace(0, 1, 11)

for t in time:
    plt.plot(volume, TVE.pressure(volume, t), "k-")
    plt.plot(volume, PALM.pressure(volume, t), "b-")
    plt.plot(volume, PANLM.pressure(volume, t), "r-")

plt.show()