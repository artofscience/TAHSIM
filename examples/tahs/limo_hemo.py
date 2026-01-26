from hemodynamics import VAV
from tahs import TimeVaryingElastance, PressureActuatedLinearMembrane, PressureActuatedNonlinearMembrane, LIMO
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from examples.helper_functions.double_hill import DoubleHill

from utils import colored_line, TDP



Pact = TDP(min=0, max=70)
heart = LIMO(Pact)

# L: float = 0.01,
# C: float = 1.5,
# R: float = 1.05,
# Z: float = 0.1,
# Pv: float = 7.5,
# Rvo: float = 0.01,
# Rvc: float = 1000

system = VAV(heart, Pv=5)

y0 = [80,60,60] # [Vv, P1, Part]
t, [Vventricular, Parterial, Paortic, Pventricular, Inflow, Outflow, Inflowresistance, Outflowresistance] = system \
(y0, 0, 6)

tb = 500
# PLOTTING

fig = plt.figure(constrained_layout=True)
gs = GridSpec(4, 2, figure=fig)

axact = fig.add_subplot(gs[0, 0])
axq = fig.add_subplot(gs[1, 0])
axp = fig.add_subplot(gs[2:, 0])
axpv = fig.add_subplot(gs[:, 1])

axp.set_ylabel(r'$P~[mmHg]$')
axq.set_ylabel(r'$Q~[mL / s]$')
axact.set_ylabel(r'$V~[mL]$')

# axact.plot(t, system.E(t), label="Elastance")
# axv = axact.twinx()
axact.plot(t[-tb:], Vventricular[-tb:], 'r-', label="Ventricular volume")
axp.plot(t[-tb:], Parterial[-tb:], label="Arterial pressure")
axp.plot(t[-tb:], Paortic[-tb:], label="Aortic pressure")
axp.plot(t[-tb:], Pventricular[-tb:], label="Ventricular pressure")
axp.plot(t[-tb:], Pact(t)[-tb:], label="Pact")

volume = np.linspace(10, 100, 100)


for time in np.linspace(t[-tb], t[-1], 100):
    im3 = colored_line(volume, heart.pressure(volume, time), Pact(time)*np.ones_like(volume), axpv, cmap='jet')
    im3.set_clim(0, 70)
fig.colorbar(im3)

axpv.plot(Vventricular[-tb:], Pventricular[-tb:], 'k', linewidth=4)
im = colored_line(Vventricular[-tb:], Pventricular[-tb:], t[-tb:], axpv, cmap='jet')
# axpv.plot(Va[-tb:], Pact(t)[-tb:], 'k', linewidth=4)
# im2 = colored_line(Va[-tb:], Pact(t)[-tb:], t[-tb:], axpv, cmap='jet')
# fig.colorbar(im2)

axpv.set_xlim([0, 120])
axpv.set_ylim([0, 80])

axpv.set_xlabel(r'$V~[mL]$')
axpv.set_ylabel(r'$P~[mmHg]$')


plt.axhline(system.Pv, color='k', linestyle='--')
# plt.axvline(heart.V0, color='k', linestyle='--')
# plt.axvline(V0A, color='k', linestyle='--')
plt.axhline(np.max(Pact(np.linspace(0,1,100))), color='k', linestyle='--')


axq.plot(t[-tb:], Inflow[-tb:], label="Ventricular inflow")
axq.plot(t[-tb:], Outflow[-tb:], label="Ventricular outflow")

axr = axp.twinx()
axr.plot(t[-tb:], Inflowresistance[-tb:], alpha=0.2)
axr.plot(t[-tb:], Outflowresistance[-tb:], alpha=0.2)

axr2 = axact.twinx()
axr2.plot(t[-tb:], Inflowresistance[-tb:], alpha=0.2)
axr2.plot(t[-tb:], Outflowresistance[-tb:], alpha=0.2)

axr3 = axq.twinx()
axr3.plot(t[-tb:], Inflowresistance[-tb:], alpha=0.2)
axr3.plot(t[-tb:], Outflowresistance[-tb:], alpha=0.2)

axact.legend()
axp.legend()
axq.legend()

plt.show()
