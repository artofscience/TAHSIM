import numpy as np
from matplotlib import pyplot as plt

from utils import Sigmoid

fcn = Sigmoid(10, 1, 10)

a = 10 / fcn.k
time = np.linspace(fcn.x0 - a, fcn.x0 + a, 100)

fig, ax = plt.subplots(2,1)
ax[0].axhline(0, ls="--")
ax[0].plot(time, fcn(time))
ax[1].axhline(0, ls="--")
ax[1].plot(time, fcn.diff(time))

plt.show()