"""
Utilitarian functionalities.
"""

import numpy as np
from matplotlib.collections import LineCollection
from sympy import symbols

class Sigmoid:
    def __init__(self, L: float = 1.0, x0: float = 0.0, k: float = 50):
        self.L = L
        self.k = k
        self.x0 = x0

    def tmp(self, x: float):
        return np.exp(-self.k * (x - self.x0))

    def __call__(self, x: float) -> float:
        return self.L / (1 + self.tmp(x))

    def diff(self, x: float) -> float:
        return self.L * self.k * self.tmp(x) / (1 + self.tmp(x))**2

class DoubleHill:
    def __init__(self, period: float = 1.0,
                 alpha_systole: float = 0.303,
                 alpha_diastole: float = 0.508,
                 rc: float = 1.32,
                 rr: float = 21.9):
        self.period = period
        self.alpha_systole = alpha_systole
        self.alpha_diastole = alpha_diastole
        self.rc = rc
        self.rr = rr

        self.max = 1
        self.max = np.max(self(np.linspace(0, self.period, 100)))

        # default values from Stergiopulos et al. (1996) Table 1. "Basic model parameters"

    def __call__(self, t) -> float:
        t = t % self.period

        tmp1 = (t / (self.alpha_systole * self.period)) ** self.rc
        tmp2 = (t / (self.alpha_diastole * self.period)) ** self.rr

        return (tmp1 / (1 + tmp1)) * (1 / (1 + tmp2)) / self.max

    def diff(self, t) -> float:
        t = t % self.period + 1e-16


        return (self.period * self.alpha_diastole) ** self.rr * (self.rc * t ** (self.rc + 2) * (
                        t ** self.rc + (self.period * self.alpha_systole) ** self.rc) * (t ** self.rr + (
                        self.period * self.alpha_diastole) ** self.rr) - self.rc * t ** (2 * self.rc + 2) * (
                                                                                 t ** self.rr + (
                                                                                     self.period * self.alpha_diastole) ** self.rr) - self.rr * t ** (
                                                                                 self.rc + self.rr + 2) * (
                                                                                 t ** self.rc + (
                                                                                     self.period * self.alpha_systole) ** self.rc)) / (
                        t ** 3 * (t ** self.rc + (self.period * self.alpha_systole) ** self.rc) ** 2 * (
                            t ** self.rr + (self.period * self.alpha_diastole) ** self.rr) ** 2)

    def symbolic(self):
        rc, rr, alc, ald, t, T = symbols('rc, rr, alc, ald, t, T', positive=True)

        tmp1 = (t / (alc * T)) ** rc
        tmp2 = (t / (ald * T)) ** rr

        f = (tmp1 / (1 + tmp1)) * (1 / (1 + tmp2))
        df = f.diff(t)
        return f, df

class TDP:
    """
    Time-dependent parameter funtion.

    y = min + alpha * (max - min) * f(t)
    dy/dt = alpha * (max - min) * df/dt(t)
    """
    def __init__(self, activation_function = DoubleHill(), alpha: float = 1.0, min: float = 0.1, max = 1.0):
        self.activation_function = activation_function
        self.alpha = alpha
        self.min = min
        self.max = max

    def __call__(self, t: float) -> float:
        return self.min + self.alpha * (self.max - self.min) * self.activation_function(t)

    def diff(self, t: float) -> float:
        return self.alpha * (self.max - self.min) * self.activation_function.diff(t)

def cubic_fit(x: np.ndarray, y: np.ndarray, id: int, m: float):
    """
    Cubic fit of y[i] = ax[i]**3 + bx[i]**2 + cx[i] + d through 3 points
    and its derivative dy = 3ax[id]**2 + 2bx[id] + c = m

    Returned are coefficients a, b, c, and d.
    """
    A = np.array([[x[0]**3, x[0]**2, x[0], 1],
                  [x[1]**3, x[1]**2, x[1], 1],
                  [x[2]**3, x[2]**2, x[2], 1],
                  [3*x[id]**2, 2*x[id], 1, 0]])
    b = np.array([y[0], y[1], y[2] , m])

    return np.linalg.solve(A, b)

def quadratic_fit(x: np.ndarray, y: np.ndarray):
    """
    Quadratic fit through three points.
    """
    A = np.array([[x[0]**2, x[0], 1],
                  [x[1]**2, x[1], 1],
                  [x[2]**2, x[2], 1]])
    b = np.array([y[0], y[1], y[2]])

    return np.linalg.solve(A, b)

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)