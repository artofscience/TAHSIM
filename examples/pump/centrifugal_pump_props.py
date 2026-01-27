import numpy as np
from matplotlib import pyplot as plt
from utils import colored_line

from pumps import CentrifugalPump

pump = CentrifugalPump()

tol=1e-3
ma = 1.1
q0 = np.linspace(tol, ma*pump.qm0-tol, 100)
h0 = np.linspace(0.0, ma* pump.hm0, 100)

plt.xlabel(r"$\frac{q}{\bar{q}_0}$ [L/min / L/min]")
plt.ylabel(r"$\frac{h}{\bar{h}_0}$ [m/m]")

Q, H = np.meshgrid(q0, h0)
im = plt.contourf(Q / pump.qm0, H / pump.hm0, Q * H * pump.lmintocubps * pump.gamma)
plt.colorbar(im)

w = pump.w0 * np.linspace(0.5, ma, 10)
for i in w:
    zi = i / pump.w0
    q = zi * q0
    hi = pump.hq(q, i)
    im2 = colored_line(q / pump.qm0, hi / pump.hm0, i * np.ones_like(q) / pump.w0, plt.gca(), cmap='jet')
    im2.set_clim(0, max(w) / pump.w0)
plt.colorbar(im2)

plt.plot(q0 / pump.qm0, pump.hq0(q0) / pump.hm0, 'r--', label='reference hq pump curve')
plt.scatter(pump.q0p / pump.qm0, pump.h0p / pump.hm0, c='r')

plt.plot(q0 / pump.qm0, pump.eff0(q0), 'b--', label='efficiency')
plt.scatter(pump.q0p / pump.qm0, pump.eff0p, c='b')

plt.xlim([0, ma])
plt.ylim([0, ma])

plt.plot(q0 / pump.qm0, 0.1 + 2 * (q0 / pump.qm0)**2, 'k-', label="system curve")
plt.legend()

plt.show()
