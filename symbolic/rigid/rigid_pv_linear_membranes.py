import sympy as sp

"""
Simple model of two rigid chambers with linear membranes.
"""

ppl, pvl, ppr, pvr = sp.symbols('ppl, pvl, ppr, pvr')
eml, emr = sp.symbols('eml, emr')
vvl, vvr, vvl0, vvr0 = sp.symbols('vvl, vvr, vvl0, vvr0')

# change in ventricle volumes
dvvl = vvl - vvl0
dvvr = vvr - vvr0

# change in membrane volumes
dvml = -dvvl
dvmr = -dvvr

# membrane equations
eq1 = sp.Eq(ppl - pvl, eml * dvml)
eq2 = sp.Eq(ppr - pvr, emr * dvmr)

solution = sp.solve((eq1, eq2), (pvl, pvr))

pvl = solution[pvl]
pvr = solution[pvr]
pv = sp.Matrix([pvl, pvr])
vv = sp.Matrix([vvl, vvr])
pp = sp.Matrix([ppl, ppr])


print("Ventricle pressures:", pv)

elastance_matrix = pv.jacobian(vv)
print("Elastance matrix:", elastance_matrix)

source_matrix = pv.jacobian(pp)
print("Source matrix:", source_matrix)

rest = sp.simplify(pv - elastance_matrix @ vv - source_matrix @ pp)
print("Rest:", rest)







