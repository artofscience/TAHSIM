import sympy as sp

"""
Simple model of wire-driven type heart.
"""

vp, vp0 = sp.symbols('vp, vp0')
vl, vl0 = sp.symbols('vl, vl0')
vr, vr0 = sp.symbols('vr, vr0')
el, er, ep, ew = sp.symbols('el, er, ep, ew')
pl, pr, pp = sp.symbols('pl, pr, pp')

dvp = vp - vp0
dvl = vl - vl0
dvr = vr - vr0
# dvp = vp
# dvl = vl
# dvr = vr

pwl = ew * (dvl + dvp)
pwr = ew * (dvr + dvp)

eq1 = sp.Eq(pl, el * dvl + pwl)
eq2 = sp.Eq(pr, er * dvr + pwr) # pressure in right ventricle equals pressure from volume plus pressure from wire
eq3 = sp.Eq(pp, ep * dvp + pwl + pwr) # pressure in pouch equals pressure from volume plus pressure from two wires

solution = sp.solve([eq1, eq2, eq3], [pl, pr, vp])

pl = solution[pl]
pr = solution[pr]
vp = solution[vp]

gamma = sp.symbols('gamma')
gamma_sub = {ep: (ew / gamma) - 2 * ew}

pl = sp.simplify(pl.subs(gamma_sub))
pr = sp.simplify(pr.subs(gamma_sub))

print("pl:", pl)
print("pr:", pr)

p = sp.Matrix([pl, pr])
v = sp.Matrix([vl, vr])
v0 = sp.Matrix([vl0, vr0])

elastance_matrix = p.jacobian(v)
elastance_matrix0 = p.jacobian(v0)

print("elastance_matrix:", sp.simplify(elastance_matrix))
print("elastance_matrix0:", -sp.simplify(elastance_matrix0))


pp = sp.Matrix([pp])

source_matrix = p.jacobian(pp)

print("source_matrix:", sp.simplify(source_matrix))

rest = p - elastance_matrix @ v - elastance_matrix0 @ v0 - source_matrix @ pp
print("rest:", sp.simplify(rest))


