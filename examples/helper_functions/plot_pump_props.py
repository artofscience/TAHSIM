import numpy as np
from matplotlib import pyplot as plt

def plot_pump_props(pump):
    q0, h0 = pump.get_operating_points(1000)

    z = np.linspace(0.1, 2, 10)
    w = pump.w0 * z

    for i in w:
        zi = i / pump.w0
        q = zi * q0
        hi = pump.hq(q, i)
        plt.plot(q, hi, color="black", label=f"w/w0 = {zi:.2f}")

    plt.scatter(pump.q0p, pump.h0p, color="blue")
    plt.plot(q0, pump.hq0(q0), color="blue", linewidth=4)

    plt.xlim([0.0, 1.5 * pump.q0p[-1]])
    plt.ylim([0.0, 1.5 * pump.h0p[0]])

    plt.legend()