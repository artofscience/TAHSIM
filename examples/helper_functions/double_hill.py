import numpy as np
from matplotlib import pyplot as plt

from utils import DoubleHill

fcn = DoubleHill()

time = np.linspace(0, fcn.period, 1000)

fig, ax = plt.subplots(2,1)

ax[0].axhline(0, ls="--")
ax[0].plot(time, fcn(time))

ax[1].axhline(0, ls="--")
ax[1].plot(time, fcn.diff(time))
plt.show()