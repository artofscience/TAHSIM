import sympy as sp

"""
Simple model of rigid chamber with linear membranes, where ventricle are connected by a linear membrane (septum).
"""

ppl, pvl, ppr, pvr = sp.symbols('ppl, pvl, ppr, pvr')
eml, emr = sp.symbols('eml, emr')
vvl, vvr, vvl0, vvr0 = sp.symbols('vvl, vvr, vvl0, vvr0')

es, vs, vs0 = sp.symbols('es, vs, vs0')

# septum pressure diff
dvs = vs - vs0
ps = es * dvs

# ventricle change in volume
dvvl = vvl - vvl0
dvvr = vvr - vvr0

# change in membrane volume
dvml = -dvvl + dvs
dvmr = -dvvr - dvs

eq1 = sp.Eq(ppl - pvl, eml * dvml)
eq2 = sp.Eq(ppr - pvr, emr * dvmr)
eq3 = sp.Eq(ps, pvl - pvr) # pressure diff equals septum pressure

solution = sp.solve((eq1, eq2, eq3), (pvl, pvr, dvs))

pvl = solution[pvl]
pvr = solution[pvr]
pv = sp.Matrix([pvl, pvr])
vv = sp.Matrix([vvl, vvr])
pp = sp.Matrix([ppl, ppr])

common_denom = eml + emr + es

print("Ventricle pressures:", pv * common_denom)

elastance_matrix = pv.jacobian(vv)
print("Elastance matrix:", elastance_matrix * common_denom)

source_matrix = pv.jacobian(pp)
print("Source matrix:", source_matrix * common_denom)

rest = sp.simplify((pv - elastance_matrix @ vv - source_matrix @ pp) * common_denom)
print("Rest:", rest)

dvs = solution[dvs]
print("dvs:", dvs * common_denom)






