from hemodynamics import VAV
from tahs import TimeVaryingElastance, PressureActuatedLinearMembrane, PressureActuatedNonlinearMembrane
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from utils import colored_line, TDP


E = TDP(min=0.06, max=1)
V0 = 20

TVE = TimeVaryingElastance(E, V0)

# L: float = 0.01,
# C: float = 1.5,
# R: float = 1.05,
# Z: float = 0.1,
# Pv: float = 7.5,
# Rvo: float = 0.01,
# Rvc: float = 1000

system = VAV(TVE)

y0 = [60, 60, 60] # [Vv, P1, Part]
t, [Vventricular, Parterial, Paortic, Pventricular, Inflow, Outflow, Inflowresistance, Outflowresistance] = system \
    (y0, 0, 6)

# PLOTTING

fig = plt.figure(constrained_layout=True)
gs = GridSpec(4, 2, figure=fig)

axact = fig.add_subplot(gs[0, 0])
axq = fig.add_subplot(gs[1, 0])
axp = fig.add_subplot(gs[2:, 0])
axpv = fig.add_subplot(gs[:, 1])

# axact.plot(t, system.E(t), label="Elastance")
# axv = axact.twinx()
axact.plot(t, Vventricular, 'r-', label="Ventricular volume")
axp.plot(t, Parterial, label="Arterial pressure")
axp.plot(t, Paortic, label="Aortic pressure")
axp.plot(t, Pventricular, label="Ventricular pressure")
colored_line(Vventricular, Pventricular, t, axpv, cmap='jet')
axpv.set_xlim([0, 150])
axpv.set_ylim([0, 120])

volume = np.linspace(0.1, 140, 10)

axpv.plot(volume, TVE.pressure(volume, 0), 'k', label="t=0")
plt.plot(volume, TVE.pressure(volume, 0.2), 'r', label="t=0.2")
plt.plot(volume, TVE.pressure(volume, 0.4), 'g', label="t=0.4")
plt.plot(volume, TVE.pressure(volume, 0.6), 'c', label="t=0.6")
plt.plot(volume, TVE.pressure(volume, 0.8), 'b', label="t=0.8")
plt.plot(volume, TVE.pressure(volume, 1), 'y', label="t=1")

axq.plot(t, Inflow, label="Ventricular inflow")
axq.plot(t, Outflow, label="Ventricular outflow")

axr = axp.twinx()
axr.plot(t, Inflowresistance, alpha=0.2)
axr.plot(t, Outflowresistance, alpha=0.2)

axr2 = axact.twinx()
axr2.plot(t, Inflowresistance, alpha=0.2)
axr2.plot(t, Outflowresistance, alpha=0.2)

axr3 = axq.twinx()
axr3.plot(t, Inflowresistance, alpha=0.2)
axr3.plot(t, Outflowresistance, alpha=0.2)

axact.legend()
axp.legend()
axq.legend()

plt.show()
