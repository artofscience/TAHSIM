from math import pi
from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from normalized_pouch import Pouch

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large'}
pylab.rcParams.update(params)

pouch = Pouch()

bnds = [(0, None), (0, None), (0, None), (0, pi)]

cons = [{'type': 'eq', 'fun': lambda x: x[0] * x[1] * x[2] - 1},
        {'type': 'eq', 'fun': lambda x: x[2] - 1}]

F = np.geomspace(0.01, 1, 5)
P = np.geomspace(0.0001, 1.5, 100)

fig, ax = plt.subplots(2, 3, sharex=True)
for f in F:
    results = []
    x0 = np.array([1, 1, 1, 1e-16], dtype=float)
    for p in P:
        result = minimize(pouch.energy, x0=x0, bounds=bnds, args=(f, p), constraints=cons,
                          method='trust-constr')
        results.append(result)
        x0 = result.x

    ax[0,0].plot(P, [i.x[0] for i in results])
    ax[1,0].plot(P, [i.x[1] for i in results])
    ax[0,1].plot(P, [i.x[3] / pi for i in results])
    ax[1,1].plot(P, [pouch.width(i.x) for i in results])
    ax[0,2].plot(P, [pouch.volume(i.x) for i in results], 'o-')
    ax[1,2].plot(P, [i.x[2] for i in results], label=r'$\frac{{F}}{{\mu HD}}={{{}}}$'.format(f))

ax[0,0].set_ylabel(r'$\frac{l}{L}$')
ax[1,0].set_ylabel(r'$\frac{h}{H}$')
ax[0,1].set_ylabel(r'$\frac{\theta}{\pi}$')
ax[1,1].set_ylabel(r'$\frac{w}{L}$')
ax[0,2].set_ylabel(r'$\frac{v_\text{p}}{L^2D}$')

ax[1,0].set_xlabel(r'$\frac{PL}{\mu H}$')
ax[1,1].set_xlabel(r'$\frac{PL}{\mu H}$')
ax[1,2].set_xlabel(r'$\frac{PL}{\mu H}$')

ax[1,2].set_ylabel(r'$\frac{d}{D}$')
ax[1,2].legend()


plt.show()