from sympy import Symbol, pi, sin, cos, diff, simplify

"""
Based on "Stretchable pouch motors" by J.T.B. Overvelde
"""

N = Symbol('N') # number of pouches
Ls = Symbol('Ls') # normalized sealing width
theta = Symbol('theta')
lam_theta = Symbol('lam_theta')
lam_s = Symbol('lam_s')
Pc = Symbol('Pc') # normalized ventricle pressure
Pa = Symbol('Pa') # normalized actuator pressure

tmp1 = Ls * (lam_s**2 + 1/lam_s**2 - 2)
tmp2 = lam_theta**2 + 1/lam_theta**2 - 2
tmp3 = - (1/2) * Pa * lam_theta**2 * (theta - sin(theta)*cos(theta)) / theta**2
tmp4 = lam_theta * sin(theta) / theta - Ls * lam_s - (1 + Ls)
tmp5 = - Pc * N * (lam_theta * sin(theta) / theta + Ls) / (2 * pi)

E = N * (tmp5 * tmp4 + tmp3 + tmp2 + tmp1)

dEdtheta = diff(E, theta)
dEdlam_theta = diff(E, lam_theta)
dEdlam_s = diff(E, lam_s)

for dE in [dEdtheta, dEdlam_theta, dEdlam_s]:
    print(simplify(dE))