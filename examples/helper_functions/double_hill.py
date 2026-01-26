import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab
from utils import DoubleHill
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large'}
pylab.rcParams.update(params)

fcn = DoubleHill()

time = np.linspace(0, fcn.period, 1000)

fig, ax = plt.subplots(2,1, sharex=True)

ax[0].axhline(0, ls="--")
ax[0].plot(time, fcn(time))
ax[0].set_ylabel(r'$\alpha$')

ax[1].axhline(0, ls="--")
ax[1].plot(time, fcn.diff(time))
ax[1].set_ylabel(r'$\frac{\text{d} \alpha}{\text{d} t}$')
ax[1].set_xlabel(r'$t$')
plt.show()