from math import pi
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from normalized_pouch import Pouch, CylindricalPouchArray
import pickle

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large'}
pylab.rcParams.update(params)

poucharray = CylindricalPouchArray(Lsh=0.1, N=8)


Pc, P, state = pickle.load(open("cylindrical_force_data.p", "rb"))

for ip, p in enumerate(P):
    vc = [poucharray.cylinder_volume(state[i, ip, :]) for i in range(0, len(Pc))]
    plt.scatter(vc, Pc, c=p * np.ones_like(Pc), cmap='jet')
    plt.plot(vc, Pc, 'r--', alpha=0.1)
    plt.clim(0, 2.5)

Pc, P, state = pickle.load(open("cylindrical_pressure_data.p", "rb"))
for ip, p in enumerate(P):
    vc = [poucharray.cylinder_volume(state[i, ip, :]) for i in range(0, len(Pc))]
    plt.scatter(vc, Pc, c=p * np.ones_like(Pc), cmap='jet')
    plt.plot(vc, Pc, 'r--', alpha=0.1)
    plt.clim(0, 2.5)

plt.colorbar()

plt.xlim([0, 1])
plt.ylim([0, 2.5])

plt.title(r'$\frac{PL}{\mu H}$')
plt.xlabel(r'$\Delta \hat{v}_\text{c} = \frac{\Delta v_\text{c}}{NL^2D}$')
plt.ylabel(r'$\frac{P_\text{c}NL}{\mu H}$')
plt.show()