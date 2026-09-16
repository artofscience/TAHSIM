import sympy as sp

ppl, pvl, ppr, pvr = sp.symbols('ppl, pvl, ppr, pvr')
eml, emr, es, epwl, evwl, epwr, evwr = sp.symbols('eml, emr, es, epwl, evwl, epwr, evwr')
dvvl, dvvr, dvs, dvpl, dvpr = sp.symbols('dvvl, dvvr, dvs, dvpl, dvpr')

ps = es * dvs

# dvpwl = ppl / epwl
# dvpwr = ppr / epwr

dvvwl = pvl / evwl
dvvwr = pvr / evwr

dvml = -dvvl + dvs + dvvwl # dvvl = dvvwl + dvs - dvml
dvmr = -dvvr - dvs + dvvwr # dvvr = dvvwr - dvs - dvmr

eq1 = sp.Eq(ppl - pvl, eml * dvml)
eq2 = sp.Eq(ppr - pvr, emr * dvmr)
eq3 = sp.Eq(ps, pvl - pvr)
# eq4 = sp.Eq(dvpl, dvpwl + dvml)
# eq5 = sp.Eq(dvpr, dvpwr + dvmr)

solution = sp.solve((eq1, eq2, eq3), (pvl, pvr, dvs))

pvl = solution[pvl]
pvr = solution[pvr]
dvs = solution[dvs]
# ppl = solution[ppl]
# ppr = solution[ppr]

elastance_matrix = sp.Matrix([pvl, pvr, ppl, ppr]).jacobian(sp.Matrix([dvvl, dvvr, dvpl, dvpr]))
common_denom = (eml*emr*es + eml*emr*evwl + eml*emr*evwr + eml*es*evwr + eml*evwl*evwr + emr*es*evwl + emr*evwl*evwr + es*evwl*evwr)
print(dvs * common_denom)
print(sp.Matrix([dvs * common_denom]).jacobian(sp.Matrix([dvvl, dvvr, ppl, ppr])))






