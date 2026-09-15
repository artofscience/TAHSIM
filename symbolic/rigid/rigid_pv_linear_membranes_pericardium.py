import sympy as sp

"""
Simple model of two rigid chamber with linear membranes, where ventricle volumes are constrained by pericardium.
"""

ppl, pvl, ppr, pvr = sp.symbols('ppl, pvl, ppr, pvr')
eml, emr = sp.symbols('eml, emr')
vvl, vvr, vvl0, vvr0 = sp.symbols('vvl, vvr, vvl0, vvr0')

ec, vs0 = sp.symbols('ec, vc0') # pericardium elastance and initial volume

# ventricle change in volume
dvvl = vvl - vvl0
dvvr = vvr - vvr0

# pericardium volume
vs = vvl + vvr

# pericardium change in volume
dvs = vs - vs0

# pericardium pressure diff
pc = ec * dvs

# membrane change in volume
dvml = -dvvl
dvmr = -dvvr

# membrane equations with additional pressure from pericardium
eq1 = sp.Eq(ppl + pc - pvl, eml * dvml)
eq2 = sp.Eq(ppr + pc - pvr, emr * dvmr)

solution = sp.solve((eq1, eq2), (pvl, pvr))

pvl = solution[pvl]
pvr = solution[pvr]
pv = sp.Matrix([pvl, pvr])
vv = sp.Matrix([vvl, vvr])
vv0 = sp.Matrix([vvl0, vvr0])
pp = sp.Matrix([ppl, ppr])


print("Ventricle pressures:", pv)

elastance_matrix = pv.jacobian(vv)
print("Elastance matrix:", elastance_matrix)

elastance0_matrix = pv.jacobian(vv0)
print("Elastance0 matrix:", elastance_matrix)

source_matrix = pv.jacobian(pp)
print("Source matrix:", source_matrix)

rest = sp.simplify(pv - elastance_matrix @ vv - elastance0_matrix @ vv0 - source_matrix @ pp)
print("Rest:", rest)






