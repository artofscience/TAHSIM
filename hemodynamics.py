""""
Hemodynamic lumped-parameter models describing the dynamics of the systemic and/or pulmonary circulation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from tahs import TAH, TimeVaryingElastance, PressureActuatedLinearMembrane, PressureActuatedNonlinearMembrane
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from utils import colored_line, TDP

class VAV:
    def __init__(self, tah: TAH = TimeVaryingElastance(),
                 L: float = 0.01,
                 C: float = 1.5,
                 R: float = 1.05,
                 Z: float = 0.1,
                 Pv: float = 7.5,
                 Rvo: float = 0.01,
                 Rvc: float = 1000):
        self.tah = tah
        self.L = L
        self.C = C
        self.R = R
        self.Z = Z
        self.Pv = Pv
        self.Rvo = Rvo
        self.Rvc = Rvc

        self.tau = self.C * self.R


    def __call__(self, y0, t_begin: float = 0.0, t_end: float = 10.0):
        sol = solve_ivp(self.solve, [t_begin, t_end], y0, atol=1e-10, rtol=1e-10)
        z = [self.flow(sol.t[i], y) for i, y in enumerate(sol.y.T)]
        return sol.t, np.vstack([sol.y, np.asarray(z).T])

    def solve(self, t, y):
        _, Qvv, Qart, _, Rva = self.flow(t, y)
        return self.ode_vars(t, y, Rva, Qvv, Qart)

    def flow(self, t, y):
        Pv = self.tah.pressure(y[0], t)

        dPVV = self.Pv - Pv # P_venous - P_ventricle
        Rvv = 1.0 * self.Rvo if dPVV >= 0 else 1.0 * self.Rvc # Venous-ventricular resistance
        Qvv = dPVV / Rvv # Venous-ventricular flow (ventricle inflow)

        dPVA = Pv - y[2] # P_ventricle - P_arterial
        Rva = 1.0 * self.Rvo if dPVA >= 0 else 1.0 * self.Rvc # Ventricular-arterial resistance
        Qart = dPVA / Rva # Ventricular-arterial flow (ventricle outflow)

        return Pv, Qvv, Qart, Rvv, Rva

    def ode_vars(self, t, y, Rva, Qvv, Qart):
        dVv = Qvv - Qart # Ventricular flow (inflow - outflow)
        dP1 = (Qart / self.C) - (y[1] / self.tau)
        tmp = self.Z / Rva
        dPa = y[1] - y[2]

        dPart = tmp * self.tah.pressure_diff(y[0], dVv, t)
        dPart += dP1 + (self.Z / self.L) * dPa
        dPart /= 1 + tmp

        return [dVv, dP1, dPart]

if __name__ == "__main__":

    Pact = TDP(min=0.0, max=40)

    for tah in [TimeVaryingElastance(), PressureActuatedLinearMembrane(Pact=Pact), PressureActuatedNonlinearMembrane(Pact=Pact)]:
        system = VAV(tah)

        y0 = [60, 60, 60] # [Vv, P1, Part]
        t, [Vventricular, Parterial, Paortic, Pventricular, Inflow, Outflow, Inflowresistance, Outflowresistance] = system(y0, 0, 6)

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

