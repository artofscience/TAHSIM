import sympy as sp

"""
Simple model of single soft pouch-ventricle with linear membrane
"""

pp, pv = sp.symbols('pp, pv')
em, ewp, ewv = sp.symbols('em, ewp, ewv')
dvp, dvv = sp.symbols('dvp, dvv')

dvm = (pp - pv) / em
dvwp = dvp - dvm
dvwv = dvv + dvm

eq1 = sp.Eq(pp, ewp * dvwp)
eq2 = sp.Eq(pv, ewv * dvwv)

pv2 = sp.solve(eq2, pv)[0]

print("Ventricle pressure:", pv2)

solution = sp.simplify(sp.solve((eq1, eq2), (pp, pv)))

pp = solution[pp]
pv = solution[pv]
p = sp.Matrix([pp, pv])
dv = sp.Matrix([dvp, dvv])

elastance_matrix = p.jacobian(dv)
common_denom = em + ewp + ewv
elastance_matrix *= common_denom

print("Elastance matrix:", elastance_matrix)


