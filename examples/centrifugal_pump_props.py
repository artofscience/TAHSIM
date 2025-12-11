import numpy as np
from matplotlib import pyplot as plt

from pumps import CentrifugalPump

pump = CentrifugalPump()

q0, h0 = pump.get_operating_points(100)

z = np.linspace(0.5, 1.5, 5)
w = pump.w0 * z

fig, host = plt.subplots(figsize=(8,5), layout='constrained')

ax2 = host.twinx()

host.set_xlim([0, pump.qm0])
host.set_ylim([0, z[-1]**2 * pump.hm0])
ax2.set_ylim([0, 1])

host.set_xlabel("Capacity [L/min]")
host.set_ylabel("Head [m]")

ax2.set_ylabel("Efficiency [-]")

for i in w:
    zi = i / pump.w0
    q = zi * q0
    hi = pump.hq(q, i)
    host.plot(q, hi, color="black", label=f"w/w0 = {zi:.2f}")
host.legend()

host.plot(q0, pump.hq0(q0))
host.scatter(pump.q0p, pump.h0p)

ax2.plot(q0, pump.eff0(q0))
ax2.scatter(pump.q0p, pump.eff0p)

plt.show()
