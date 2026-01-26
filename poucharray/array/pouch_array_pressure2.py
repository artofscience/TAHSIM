from math import pi
from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
from normalized_pouch import PouchArray2, Pouch

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large'}
pylab.rcParams.update(params)

poucharray = PouchArray2(Lwh=1/55, N=5)

bnds = [(0, None), (0, None), (0, None), (0, pi), (0, None), (0, None)]

cons = [{'type': 'eq', 'fun': lambda x: x[0] * x[1] * x[2] - 1},
        {'type': 'eq', 'fun': lambda x: x[4] * x[5] * x[2] - 1},
        {'type': 'eq', 'fun': lambda x: x[2] - 1}]

F = np.linspace(0.01, 1, 5)
Pa = np.linspace(0.0001, 1.5, 100)

fig, ax = plt.subplots(3, 3, sharex=True)
for f in F:
    results = []
    x0 = np.array([1, 1, 1, 1e-16, 1, 1], dtype=float)
    for p in Pa:
        result = minimize(poucharray.energy, x0=x0, bounds=bnds, args=(f, p), constraints=cons,
                          method='trust-constr')
        results.append(result)
        x0 = result.x

    ax[0,0].plot(Pa, [i.x[0] for i in results])
    ax[1,0].plot(Pa, [i.x[1] for i in results])
    ax[2,0].plot(Pa, [i.x[2] for i in results], label=r'$\frac{{F}}{{\mu HD}}={{{}}}$'.format(f))
    ax[0,1].plot(Pa, [i.x[3] / pi for i in results])
    ax[1,1].plot(Pa, [i.x[4] for i in results])
    ax[2,1].plot(Pa, [i.x[5] for i in results])
    ax[0,2].plot(Pa, [Pouch.width(i.x[:4]) for i in results])
    ax[1,2].plot(Pa, [Pouch.volume(i.x[:4]) for i in results])
    ax[2,2].plot(Pa, [poucharray.volume(i.x) for i in results])

ax[0,0].set_ylabel(r'$\frac{l}{L}$')

ax[1,0].set_ylabel(r'$\frac{h}{H}$')
ax[2,0].set_ylabel(r'$\frac{d}{D}$')
ax[2,0].legend()

ax[0,1].set_ylabel(r'$\frac{\theta}{\pi}$')
ax[1,1].set_ylabel(r'$\frac{l_\text{s}}{L_\text{s}}$')
ax[2,1].set_ylabel(r'$\frac{t}{H}$')

ax[0,2].set_ylabel(r'$\frac{w}{L}$')
ax[1,2].set_ylabel(r'$\hat{v}_\text{a} = \hat{v}_\text{p} = \frac{v_\text{a}}{NL^2D}$')
ax[2,2].set_ylabel(r'$\bar{v}_\text{a} = \frac{Nv_\text{a}}{DW_\text{a}^2}$')

ax[2,0].set_xlabel(r'$\frac{PW_\text{a}}{\mu NH}$')
ax[2,1].set_xlabel(r'$\frac{PW_\text{a}}{\mu NH}$')
ax[2,2].set_xlabel(r'$\frac{PW_\text{a}}{\mu NH}$')

fig.suptitle(r'$\hat{{L}}_\text{{h}} = {{{}}}, N = {{{}}}, \frac{{L_\text{{s}}}}{{L}} = {{{}}}$'.format(poucharray.Lwh, poucharray.N, poucharray.Lsh))
plt.show()