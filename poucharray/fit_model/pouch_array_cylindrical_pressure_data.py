from math import pi
from scipy.optimize import minimize
import numpy as np
from normalized_pouch import CylindricalPouchArray
import pickle

poucharray = CylindricalPouchArray(Lsh=0.1, N=8)

bnds = [(0, None), (0, None), (0, None), (0, pi), (0, None), (0, None)]

cons = [{'type': 'eq', 'fun': lambda x: x[0] * x[1] * x[2] - 1},
        {'type': 'eq', 'fun': lambda x: x[4] * x[5] * x[2] - 1},
        {'type': 'eq', 'fun': lambda x: x[2] - 1}]

Pc = np.linspace(0.001, 5, 20)
P = np.geomspace(0.00001, 1.75, 100)

state = np.zeros((np.size(Pc), np.size(P), 6), dtype=float)
for ipc, pc in enumerate(Pc):
    x0 = np.array([1, 1, 1, 1e-6, 1, 1], dtype=float)
    for ip, p in enumerate(P):
        state[ipc, ip, :] = minimize(poucharray.energy, x0=x0, bounds=bnds, args=(pc, p), constraints=cons,
                          method='trust-constr').x
        x0[:] = state[ipc, ip, :]

pickle.dump([Pc, P, state], open("cylindrical_pressure_data.p", "wb"))