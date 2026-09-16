from math import pi
import numpy as np
from scipy.interpolate import interp1d

from utils import Sigmoid
from tahs import LinearMembrane, NonlinearMembrane
from models import Source, Actuator, Actuation, Circulation, Controller, Circulations, Heart, Valve, BiVen, find_one_to_zero, find_zero_to_one, ejected_volume

from matplotlib import pyplot as plt

data = {"relative duration systole": 1/3, # relative duration of systole vs full cycle
        "total blood volume": 5000, # total amount of blood in system in mL
        "bpm": 60, # characteristic bpm or actuation frequency
        "phase left": pi, # phase delay for left ventricle actuation
        "phase right": pi, # phase delay for right ventricle actuation
        "mean arterial pressure": 100, # characteristic mean arterial pressure in mmHg
        "stroke volume": 100, # characteristic stroke volume in mL
        "source pressure left ventricle": 350, # pressure of left source in mmHg (50 kPa)
        "source pressure right ventricle": 200, # pressure of right source in mmHg (30 kPa)
        "source capacitance left": 0.1, # capacitance of left pressure source in mL / mmHg
        "source capacitance right": 0.1, # capacitance of right pressure source in mL / mmHg
        "source unstress volume left": 0.0, # amount of water in left source line in mL
        "source unstress volume right": 0.0,  # amount of water in left source line in mL
        "source resistance left": 0.05, # resistance of line between left source and ventricle in mmHg * s / mL
        "source resistance right": 0.05,  # resistance of line between right source and ventricle in mmHg * s / mL
        "resistance aortic valve": 0.005, # resistance of aortic valve in mmHg * s / mL
        "resistance mitral valve": 0.005,  # resistance of mitral valve in mmHg * s / mL
        "resistance tricuspid valve": 0.005,  # resistance of tricuspid valve in mmHg * s / mL
        "resistance pulmonary valve": 0.005,  # resistance of pulmonary valve in mmHg * s / mL
        "systemic arterial capacitance": 1.0, # systemic arterial capacitance in mL / mmHg
        "systemic venous capacitance": 100.0,  # systemic venous capacitance in mL / mmHg
        "systemic arterial unstressed volume": 100.0, # unstressed volume of systemic arterial system in mL
        "systemic venous unstressed volume": 2500.0,  # unstressed volume of systemic arterial system in mL
        "systemic vascular resistance": 2.0, # systemic vascular resistance in mmHg * s / mL
        "pulmonary arterial capacitance": 1.0,  # pulmonary arterial capacitance in mL / mmHg
        "pulmonary venous capacitance": 4.0,  # pulmonary venous capacitance in mL / mmHg
        "pulmonary arterial unstressed volume": 100.0,  # unstressed volume of pulmonary arterial system in mL
        "pulmonary venous unstressed volume": 500.0,  # unstressed volume of pulmonary arterial system in mL
        "pulmonary vascular resistance": 0.2,  # pulmonary vascular resistance in mmHg * s / mL
        "ventricle left unstressed volume": 200, # unstressed blood volume of left ventricle in mL
        "ventricle right unstressed volume": 200, # unstressed blood volume of right ventricle in mL
        "pouch left unstressed volume": 200, # unstressed water/air volume of left pouch in mL
        "pouch right unstressed volume": 200, # unstressed water/air volume of right pouch in mL
        "left ventricle elastance": 1.0, # elastance of left ventricle (if linear) in mmHg / mL
        "right ventricle elastance": 0.1,  # elastance of left ventricle (if linear) in mmHg / mL
        "septum elastance": 0.5,  # somewhere in between left and right elastance
        "septum unstressed volume": 0.0,
        "pericardium elastance": 0.1,
        "pericardium unstressed volume": 400.0,  # equals unstressed volume of left and right ventricle
        "controller variable": 4,
        "controller reference": 280,
        "controller proportinal gain": 30.0,
        "controller integral gain": 1.0,
        "controller derivative gain": 0.0,
        "controller moving average time constant": 6.0
        }

p_ref = data["mean arterial pressure"]
t_ref = 1 / (data["bpm"] / 60)
v_ref = data["stroke volume"]
f_ref = 1/ t_ref
e_ref = p_ref / v_ref
c_ref = 1 / e_ref
q_ref = v_ref / t_ref
r_ref = p_ref / q_ref

svr = lambda t: data["systemic vascular resistance"] / r_ref - Sigmoid(1.0 / r_ref, 30)(t) + Sigmoid(2.0 / r_ref, 40)(t)
pvr = lambda t: data["pulmonary vascular resistance"] / r_ref + Sigmoid(0.2 / r_ref, 50)(t)

sac = lambda t: data["systemic arterial capacitance"] / c_ref
svc = lambda t: data["systemic venous capacitance"] / c_ref - Sigmoid(99.0 / c_ref, 10.0)(t) + Sigmoid(200.0 / c_ref, 20.0)(t)

pac = lambda t: data["pulmonary arterial capacitance"] / c_ref
pvc = lambda t: data["pulmonary venous capacitance"] / c_ref

# pressure source
pressure_source_left = Source(data["source pressure left ventricle"] / p_ref, data["bpm"] / 60 / f_ref,
                              data["phase left"], data["relative duration systole"])

pressure_source_right = Source(data["source pressure right ventricle"] / p_ref, data["bpm"] / 60 / f_ref,
                               data["phase right"], data["relative duration systole"])

# pressure actuation circuit
actuator_left = Actuator(pressure_source_left, data["source capacitance left"] / c_ref,
                         data["source unstress volume left"] / v_ref, data["source resistance left"] / r_ref)

actuator_right = Actuator(pressure_source_right, data["source capacitance right"] / c_ref,
                          data["source unstress volume right"] / v_ref, data["source resistance right"] / r_ref)

actuation = Actuation(actuator_left, actuator_right)

# valves
aortic_valve = Valve(data["resistance aortic valve"] / r_ref, dhopen=1/p_ref)

tricuspid_valve = Valve(data["resistance tricuspid valve"] / r_ref, dhopen=1/p_ref)

pulmonary_valve = Valve(data["resistance pulmonary valve"] / r_ref, dhopen=1/p_ref)

mitral_valve = Valve(data["resistance mitral valve"] / r_ref, dhopen=1/p_ref)


# systemic and pulmonary circulation
systemic_circulation = Circulation(aortic_valve, tricuspid_valve, sac,
                                   data["systemic arterial unstressed volume"] / v_ref, svc,
                                   data["systemic venous unstressed volume"] / v_ref, svr)

pulmonary_circulation = Circulation(pulmonary_valve, mitral_valve, pac,
                                   data["pulmonary arterial unstressed volume"] / v_ref, pvc,
                                   data["pulmonary venous unstressed volume"] / v_ref, pvr)

circulations = Circulations(systemic_circulation, pulmonary_circulation)

# heart
left_ventricle = LinearMembrane(data["left ventricle elastance"] / e_ref, data["ventricle left unstressed volume"] / v_ref,
                                data["pouch left unstressed volume"] / v_ref)

right_ventricle = LinearMembrane(data["right ventricle elastance"] / e_ref, data["ventricle right unstressed volume"] / v_ref,
                                 data["pouch right unstressed volume"] / v_ref)

heart = Heart(left_ventricle, right_ventricle)

controller = Controller(data["controller variable"], data["controller reference"] / p_ref, data["controller moving average time constant"],
                        data["controller proportinal gain"], data["controller integral gain"], data["controller derivative gain"])

# system
system = BiVen(actuation, heart, circulations, controller=controller)

# initial guess

vl0 = data["ventricle left unstressed volume"] / v_ref
vr0 = data["ventricle right unstressed volume"] / v_ref

vs0a = data["systemic arterial unstressed volume"] / v_ref
vs0v = data["systemic venous unstressed volume"] / v_ref

vp0a = data["pulmonary arterial unstressed volume"] / v_ref
vp0v = data["pulmonary venous unstressed volume"] / v_ref

total_volume = data["total blood volume"] / v_ref
unstressed_volume = vl0 + vr0 + vp0a + vp0v + vs0a + vs0v
stressed_volume = total_volume - unstressed_volume

vp10 = vp0a + 1/7 * stressed_volume
vp20 = vp0v + 1/7 * stressed_volume
vs10 = vs0a + 1/7 * stressed_volume
vs20 = vs0v + 4/7 * stressed_volume

initial_guess_volumes = (0.0, 0.0, vl0, vr0, vs10, vs20, vp10, vp20, data["controller reference"] / p_ref, 0.0)

no_actuation_cycles = 60


# solve dynamics
times, volumes, pressures, flows, valves, control_vars = system.dynamics(initial_guess_volumes, no_actuation_cycles)
haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = pressures

pleft, pright, _, _ = system.set_actuation_pressures(control_vars, hs1)

valve_times, valve_values, state_aortic_valve, state_tricuspid_valve, state_pulmonary_valve, state_mitral_valve = valves

# index = np.searchsorted(times, no_actuation_cycles-1, side="right")
# idxv = np.searchsorted(valve_times, no_actuation_cycles-1, side="right")
index = 0
idxv = 0

plt.figure()

vaL, vaR, vvL, vvR, vs1, vs2, vp1, vp2 = volumes

[plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0][idxv:]]
# plt.axhline(heart.left.Vv0, linestyle='--', color='black', label="LVv0")
plt.plot(times[index:], vvL[index:], 'r-', label="left ventricular volume")
# plt.axhline(heart.right.Vv0, linestyle='--', color='blue', label="RVv0")
plt.plot(times[index:], vvR[index:], 'b-', label="right ventricular volume")
plt.plot(times[index:], vp1[index:], 'b--', label="volume pulmonary arteries")
plt.plot(times[index:], vs1[index:], 'r--', label="volume systemic arteries")
plt.plot(times[index:], vaL[index:], 'r-.', label="left source volume")
plt.plot(times[index:], vaR[index:], 'b-.', label="right source volume")

plt.axhline(0, color='k')

# plt.plot(times[index:], vs2[index:], 'g-', label="volume systemic veins")
# plt.plot(times[index:], vp2[index:], 'c-', label="volume pulmonary veins")
# plt.plot(times, vvL + vvR + vp1 + vp2 + vs1 + vs2)
# plt.axhline(5000 / v_ref)

plt.xlabel("Time (s) / TO (s)")
plt.ylabel("Volume (mL) / SV0 (mL)")
plt.legend()


plt.figure()

system.actuation.left.source.magnitude = 1.0
system.actuation.right.source.magnitude = 1.0

hsL, hsR = pleft * system.actuation.left.source(times), pright * system.actuation.right.source(times)
# hsL, hsR = system.actuation.left.source(times), system.actuation.right.source(times)

xavg = control_vars[0]

[plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0][idxv:]]
plt.plot(times[index:], pleft[index:], 'k:', alpha=0.2)
plt.plot(times[index:], pright[index:], 'k:', alpha=0.2)
plt.plot(times[index:], hsL[index:], 'r--', label="Source left")
plt.plot(times[index:], hsR[index:], 'b--', label="Source right")
plt.plot(times[index:], haL[index:], 'r-.', label='Pouch left')
plt.plot(times[index:], haR[index:], 'b-.', label='Pouch right')
plt.plot(times[index:], hvL[index:], 'ro-', label='Ventricle left')
plt.plot(times[index:], hvR[index:], 'bo-', label='Ventricle right')
plt.plot(times[index:], hs1[index:], 'r-', label='Afterload left ventricle (aorta)')
plt.plot(times[index:], hp2[index:], 'r:', label='Preload left ventricle (LAP, pulmonary)')
plt.plot(times[index:], hp1[index:], 'b-', label='Afterload right ventricle (pulmonary artery)')
plt.plot(times[index:], hs2[index:], 'b:', label='Preload right ventricle (RAP, systemic)')
plt.axhline(2.8, color='k', linestyle='--', label='Reference afterload')
plt.plot(times[index:], xavg[index:], 'k-', label='Averaged afterload')
plt.axhline(0, color='k')

plt.xlabel("Time (s) / T0 (s)")
plt.ylabel("Pressure (mmHg) / MAP (mmHg)")
plt.legend()

plt.figure()

[plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0][idxv:]]
plt.plot(times[index:], data["mean arterial pressure"] *pleft[index:], 'k:', alpha=0.2)
plt.plot(times[index:], data["mean arterial pressure"] *pright[index:], 'k:', alpha=0.2)
plt.plot(times[index:], data["mean arterial pressure"] *hsL[index:], 'r--', label="Source left")
plt.plot(times[index:], data["mean arterial pressure"] *hsR[index:], 'b--', label="Source right")
plt.plot(times[index:], data["mean arterial pressure"] *haL[index:], 'r-.', label='Pouch left')
plt.plot(times[index:], data["mean arterial pressure"] *haR[index:], 'b-.', label='Pouch right')
plt.plot(times[index:], data["mean arterial pressure"] *hvL[index:], 'ro-', label='Ventricle left')
plt.plot(times[index:], data["mean arterial pressure"] *hvR[index:], 'bo-', label='Ventricle right')
plt.plot(times[index:], data["mean arterial pressure"] *hs1[index:], 'r-', label='Afterload left ventricle (aorta)')
plt.plot(times[index:], data["mean arterial pressure"] *hp2[index:], 'r:', label='Preload left ventricle (LAP, pulmonary)')
plt.plot(times[index:], data["mean arterial pressure"] *hp1[index:], 'b-', label='Afterload right ventricle (pulmonary artery)')
plt.plot(times[index:], data["mean arterial pressure"] *hs2[index:], 'b:', label='Preload right ventricle (RAP, systemic)')
plt.axhline(2.8, color='k', linestyle='--', label='Reference afterload')
plt.plot(times[index:], data["mean arterial pressure"] *xavg[index:], 'k-', label='Averaged afterload')
plt.axhline(0, color='k')

plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.legend()



# plt.figure()
#
# qaL, qaR, qvL, qvR, qs1, qs2, qp1, qp2 = flows
#
# qsL, qsR = system.source_flow(times, haL, haR)
#
# [plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0][idxv:]]
# plt.plot(times[index:], qsL[index:], 'r--', label='Source left')
# plt.plot(times[index:], qsR[index:], 'b--', label='Source right')
# plt.plot(times[index:], qvL[index:], 'r-', label='Ventricle left')
# plt.plot(times[index:], qvR[index:], 'b-', label='Ventricle right')
# plt.plot(times[index:], qs1[index:], 'r:', label='Systemic arterial')
# plt.plot(times[index:], qp1[index:], 'b:', label='Pulmonary arterial')
# plt.axhline(0, color='k')
# plt.xlabel("Time (s) / T0 (s)")
# plt.ylabel("Flow rate (mL/s) / q0 (mL/s)")
# plt.legend()

plt.figure()
plt.plot(vvL[index:], hvL[index:], 'r-', label="left ventricle PV")
plt.plot(heart.left.Vp0 + heart.left.Vv0 - vvL[index:], haL[index:], 'r--', label="left pouch PV", alpha=0.1)
# plt.axvline(heart.left.Vp0, linestyle="dotted", color='red', label="initial left pouch volume")
# plt.axvline(heart.left.Vv0, linestyle="dashed", color="red", label="initial left ventricle volume")

plt.plot(vvR[index:], hvR[index:], 'k-', label="right ventricle PV")
plt.plot(heart.right.Vp0 + heart.right.Vv0 - vvR[index:], haR[index:], 'k--', label="right pouch PV", alpha=0.1)
# plt.axvline(heart.right.Vp0, linestyle="dotted", color='black', label="initial right pouch volume")
# plt.axvline(heart.right.Vv0, linestyle="dashed", color="black", label="initial right ventricle volume")

plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.xlabel("Volume (mL) / SV0 (mL)")
plt.ylabel("Pressure (mmHg) / MAP (mmHg)")
plt.legend()

plt.figure()
plt.plot(data["stroke volume"] * vvL[index:], data["mean arterial pressure"] *hvL[index:], 'r-', label="left ventricle PV")
plt.plot(data["stroke volume"] * heart.left.Vp0 + data["stroke volume"] * heart.left.Vv0 - data["stroke volume"] * vvL[index:], data["mean arterial pressure"] *haL[index:], 'r--', label="left pouch PV", alpha=0.1)
# plt.axvline(heart.left.Vp0, linestyle="dotted", color='red', label="initial left pouch volume")
# plt.axvline(heart.left.Vv0, linestyle="dashed", color="red", label="initial left ventricle volume")

plt.plot(data["stroke volume"] *vvR[index:], data["mean arterial pressure"] *hvR[index:], 'k-', label="right ventricle PV")
plt.plot(data["stroke volume"] *(heart.right.Vp0 + heart.right.Vv0 - vvR[index:]),data["mean arterial pressure"] * haR[index:], 'k--', label="right pouch PV", alpha=0.1)
# plt.axvline(heart.right.Vp0, linestyle="dotted", color='black', label="initial right pouch volume")
# plt.axvline(heart.right.Vv0, linestyle="dashed", color="black", label="initial right ventricle volume")

plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.xlabel("Volume (mL)")
plt.ylabel("Pressure (mmHg)")
plt.legend()


# plt.figure()
# plt.plot(vvL[index:]/vvR[index:], qvL[index:]/qvR[index:], 'k-')
# plt.xlabel("Ventricular volume ratio")
# plt.ylabel("Ventricular flow ratio")

plt.figure()
plt.plot(vvL, vvR, 'k-')
plt.xlabel("Left ventricular volume")
plt.ylabel("right ventricular volume")

plt.figure()

aortic_valve_opening = find_zero_to_one(state_aortic_valve)
left_stroke_volume = state_aortic_valve * ejected_volume(vvL, aortic_valve_opening)
aortic_valve_closing = find_one_to_zero(state_aortic_valve)
idx_left = [i - 1  for i, val in enumerate(aortic_valve_closing) if val == 1]
lsv = left_stroke_volume[idx_left]

pulmonary_valve_opening = find_zero_to_one(state_pulmonary_valve)
right_stroke_volume = state_pulmonary_valve * ejected_volume(vvR, pulmonary_valve_opening)
pulmonary_valve_closing = find_one_to_zero(state_pulmonary_valve)
idx_right = [i - 1 for i, val in enumerate(pulmonary_valve_closing) if val == 1]
rsv = right_stroke_volume[idx_right]

plt.plot(times, state_aortic_valve, 'k--', label="state aortic valve")
plt.plot(times, left_stroke_volume, 'k-.', label="left ejected volume")
plt.plot(times[idx_left], left_stroke_volume[idx_left], 'k.-', label="lsv")

plt.plot(times, state_pulmonary_valve, 'b--', label="state pulmonary valve")
plt.plot(times, right_stroke_volume, 'b-.', label="right ejected volume")
plt.plot(times[idx_right], right_stroke_volume[idx_right], 'b.-', label="rsv")

t = np.linspace(0, no_actuation_cycles, no_actuation_cycles)
aligned_left = np.interp(t, times[idx_left], left_stroke_volume[idx_left])
aligned_right = np.interp(t, times[idx_right], right_stroke_volume[idx_right])
stroke_difference = np.abs(aligned_right - aligned_left)
stroke_difference_grad = np.gradient(stroke_difference, t)
plt.plot(t, stroke_difference, 'r.-', label="stroke difference")
plt.plot(t, stroke_difference_grad, 'r.-', label="stroke difference gradient")

plt.xlabel("Time (s) / TO (s)")
plt.ylabel("Volume (mL) / SV0 (mL)")
plt.legend()

resistance_av = aortic_valve.Rclosed - state_aortic_valve * (aortic_valve.Rclosed - aortic_valve.Ropen)
qav = (hvL - hs1) / resistance_av

plt.figure()
plt.plot(times, state_aortic_valve, 'k--', alpha=0.1, label="state aortic valve")
plt.plot(times, qav/10, 'ko-', label="Flow aortic valve / 10")
plt.plot(times, vvL, 'r-', label="left ventricular volume")

flow_interpolator = interp1d(times, qav, kind='linear', fill_value="extrapolate")
no_points = 100000
time_uniform = np.linspace(0, no_actuation_cycles, no_points)
qav_uniform = flow_interpolator(time_uniform)
plt.plot(time_uniform, qav_uniform/10, 'bo-', label="Flow aortic valve uniform / 10")

no_points_per_beat = no_points / no_actuation_cycles
window_size = int(no_points_per_beat)
window = np.ones(window_size) / no_points_per_beat
left_stroke_volume_integrated = np.convolve(qav_uniform, window, mode='same')
plt.plot(time_uniform, left_stroke_volume_integrated, 'k--', label="left stroke volume")
plt.plot(times, left_stroke_volume, 'k:', label="left ejected volume")

plt.legend()


plt.figure()
plt.axhline(0, color='k', linestyle="dashed")
plt.axvline(0, color='k', linestyle="dashed")
plt.plot(stroke_difference, stroke_difference_grad, 'ko-')
plt.ylabel("[ Change of imbalance (mL) * T0 (s) ] / [ SV0 (mL) * Time (s) ] ")
plt.xlabel("Imbalance (mL) / SV0 (mL)")

plt.figure()
plt.title("Example of an imbalance measure")
plt.plot(t, np.sign(stroke_difference_grad) * np.linalg.norm(np.stack((stroke_difference, stroke_difference_grad)).T, axis=1))




plt.show()