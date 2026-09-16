from models import FourCompartment, PhysiologicValues
from presentation.biventricular_membrane import Valve
from events_lab.biventricular_pressure_source import Source
from scipy.integrate import solve_ivp
import numpy as np
from matplotlib import pyplot as plt
from math import pi


heart = PhysiologicValues()

v_ref = heart.v_stroke
vl0 = heart.v_left_dead / v_ref
vs0 = heart.v_systemic_dead / v_ref
vr0 = heart.v_right_dead / v_ref
vp0 = heart.v_pulmonary_dead / v_ref

p_ref = heart.p_mean_arterial
e_ref = p_ref / v_ref
el = heart.e_left_ventricle / e_ref
er = heart.e_right_ventricle / e_ref

c_ref = 1 / e_ref
cs = heart.c_systemic_circulation / c_ref
cp = heart.c_pulmonary_circulation / c_ref

t_ref = heart.t_act
q_ref = v_ref / t_ref
r_ref = p_ref / q_ref
ra = heart.r_valve_aortic / r_ref
rt = (heart.r_valve_tricuspid + heart.r_systemic_circulation) / r_ref
rp = heart.r_valve_pulmonary / r_ref
rm = (heart.r_valve_mitral + heart.r_pulmonary_circulation) / r_ref

psl = heart.p_left_ventricle_max / p_ref
psr = heart.p_right_ventricle_max / p_ref

model = FourCompartment(sl=Source(psl, 1/t_ref, phase=pi/2, duty=1/3),
                         sr=Source(psr, 1/t_ref, phase=pi, duty=1/3),
                         ra=Valve(Ropen=ra, Rclosed=10000*r_ref, dhopen=0.0, dhclose=0.0),
                        rt=Valve(Ropen=rt, Rclosed=10000*r_ref, dhopen=0.0, dhclose=0.0),
                        rm=Valve(Ropen=rm, Rclosed=10000*r_ref, dhopen=0.0, dhclose=0.0),
                         rp=Valve(Ropen=rp, Rclosed=10000*r_ref, dhopen=0.0, dhclose=0.0),
                         el=el, er=er, cs=cs, cp=cp,
                         vl0=vl0, vs0=vs0, vr0=vr0, vp0=vp0)

events = [model.aortic_valve_open, model.aortic_valve_close,
          model.tricuspid_valve_open, model.tricuspid_valve_close,
          model.pulmonary_valve_open, model.pulmonary_valve_close,
          model.mitral_valve_open, model.mitral_valve_close]

initial_state = (vl0, vs0, vr0, vp0)

t_start = 0.0
t_end = 10.0

t_full = []
y_full = []
derivatives = []
event_times = []

while t_start < t_end:
    sol = solve_ivp(model.solve, [t_start, t_end], initial_state, events=events, rtol=1e-9, atol=1e-9)
    derivatives.append(model.solve(sol.t, sol.y))

    t_full.append(sol.t)
    y_full.append(sol.y)


    if any([i.size > 0 for i in sol.t_events]):

        event = next(i for i, j in enumerate(sol.t_events) if len(j))
        if event == 0:
            model.ra.open()
        elif event == 1:
            model.ra.close()
        elif event == 2:
            model.rt.open()
        elif event == 3:
            model.rt.close()
        elif event == 4:
            model.rp.open()
        elif event == 5:
            model.rp.close()
        elif event == 6:
            model.rm.open()
        elif event == 7:
            model.rm.close()
        else:
            print("no valid event")

        event_time = sol.t_events[event][0]
        event_times.append(event_time)
        t_start = event_time
        initial_state = sol.y_events[event][0]
    else:
        t_start = t_end

t_full = np.concatenate(t_full)
y_full = np.concatenate(y_full, axis=1)
derivatives = np.concatenate(derivatives, axis=1)

vl, vs, vr, vp = y_full
ql, qs, qr, qp = derivatives

plt.figure()
plt.plot(t_full, model.sl(t_full))
plt.plot(t_full, model.sr(t_full))

plt.figure()

plt.plot(t_full, vl)
plt.plot(t_full, vr)
plt.plot(t_full, vs)
plt.plot(t_full, vp)

plt.show()







