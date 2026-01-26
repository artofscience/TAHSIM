from math import pi
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from normalized_pouch import Pouch, CylindricalPouchArray
import pickle
from scipy.optimize import curve_fit

def func(xy, a, b, c, d, e, f, g, h):
    x, y = xy
    return a + b * x + c * y + d * x**2 + e * y**2 + f * x * y + g * x**3 + h * y**3


params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large'}
pylab.rcParams.update(params)

poucharray = CylindricalPouchArray(Lsh=0.1, N=8)

Pc, P, state = pickle.load(open("cylindrical_force_data.p", "rb"))
Pc = Pc[60:]
P = P[5:-1]
state = state[60:, 5:-1,:]

y = np.repeat(P, len(Pc))
x = []
z = np.tile(Pc, len(P))

plt.figure()
for ip, p in enumerate(P):
    vc = [poucharray.cylinder_volume(state[i, ip, :]) for i in range(0, len(Pc))]
    x.extend(np.asarray(vc))
    plt.scatter(vc, Pc, c=p * np.ones_like(Pc), cmap='jet')
    plt.plot(vc, Pc, 'r--', alpha=0.1)
    plt.clim(0, 2.5)

plt.colorbar()
plt.xlim([-0.1, 1])
plt.ylim([-0.1, 2.5])
plt.axvline(0, color='k')
plt.axhline(0, color='k')

popt, pcov = curve_fit(func, (x, y), z)

plt.figure()
for ip, p in enumerate(P):
    vc = np.asarray([poucharray.cylinder_volume(state[i, ip, :]) for i in range(0, len(Pc))])
    plt.scatter(vc, func((vc, p), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]), c=p * np.ones_like(Pc), cmap='jet')
    plt.plot(vc, func((vc, p), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]), 'b--', alpha=0.1)
    plt.clim(0, 2.5)

plt.colorbar()
plt.xlim([-0.1, 1])
plt.ylim([-0.1, 2.5])

plt.axvline(0, color='k')
plt.axhline(0, color='k')

plt.title(r'$\frac{PL}{\mu H}$')
plt.xlabel(r'$\hat{v}_\text{c} = \frac{v_\text{c}}{NL^2D}$')
plt.ylabel(r'$\frac{P_\text{c}NL}{\mu H}$')

plt.figure()

PP = np.linspace(0.001, 2.5, 100)
vcc = np.linspace(0.1, 1.0, 100)
for PPP in PP:
    plt.scatter(vcc, func((vcc, PPP), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]), c=PPP * np.ones_like(vcc), cmap='jet')
    plt.plot(vcc, func((vcc, PPP), popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]), 'b--', alpha=0.1)
    plt.clim(0, 2.5)

plt.colorbar()
plt.xlim([-0.1, 1])
plt.ylim([-0.1, 2.5])
plt.show()