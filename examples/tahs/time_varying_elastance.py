import numpy as np
from matplotlib import pyplot as plt
from utils import TDP
from tahs import TimeVaryingElastance
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large'}
pylab.rcParams.update(params)


V0 = 20

TVE = TimeVaryingElastance(TDP(min=0.06, max=2.31), V0=V0)

plt.axhline(0, ls='--', color='k')
plt.axvline(0, ls='--', color='k')
plt.axvline(V0, ls='--', color='k')
volume = np.linspace(0.1, 99.9, 100)
time = np.linspace(0, 1, 11)

for t in time:
    plt.plot(volume, TVE.pressure(volume, t), 'k')

plt.xlabel('Ventricle volume [m^3]')
plt.ylabel('Ventricle pressure [Pa]')

plt.show()