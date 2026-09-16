from math import pi
import numpy as np

from tahs import LinearMembrane
from left_right_balance.models import Source, Actuator, Actuation, Circulation, Circulations, Heart, Valve, BiVenSystem, find_one_to_zero, find_zero_to_one, ejected_volume
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from utils import Sigmoid

data = {"relative duration systole": 1/3, # relative duration of systole vs full cycle
        "total blood volume": 5000, # total amount of blood in system in mL
        "bpm": 60, # characteristic bpm or actuation frequency
        "phase left": pi, # phase delay for left ventricle actuation
        "phase right": pi, # phase delay for right ventricle actuation
        "mean arterial pressure": 100, # characteristic mean arterial pressure in mmHg
        "stroke volume": 100, # characteristic stroke volume in mL
        "source pressure left ventricle": 370, # pressure of left source in mmHg (50 kPa)
        "source pressure right ventricle": 220, # pressure of right source in mmHg (30 kPa)
        "source capacitance left": 0.1, # capacitance of left pressure source in mL / mmHg
        "source capacitance right": 0.1, # capacitance of right pressure source in mL / mmHg
        "source unstress volume left": 0.0, # amount of water in left source line in mL
        "source unstress volume right": 0.0,  # amount of water in left source line in mL
        "source resistance left": 0.07, # resistance of line between left source and ventricle in mmHg * s / mL
        "source resistance right": 0.07,  # resistance of line between right source and ventricle in mmHg * s / mL
        "resistance aortic valve": 0.005, # resistance of aortic valve in mmHg * s / mL
        "resistance mitral valve": 0.005,  # resistance of mitral valve in mmHg * s / mL
        "resistance tricuspid valve": 0.005,  # resistance of tricuspid valve in mmHg * s / mL
        "resistance pulmonary valve": 0.005,  # resistance of pulmonary valve in mmHg * s / mL
        "systemic arterial capacitance": 1.0, # systemic arterial capacitance in mL / mmHg
        "systemic venous capacitance": 100.0,  # systemic venous capacitance in mL / mmHg
        "systemic arterial unstressed volume": 100.0, # unstressed volume of systemic arterial system in mL
        "systemic venous unstressed volume": 2500.0,  # unstressed volume of systemic arterial system in mL
        "systemic vascular resistance": 1.5, # systemic vascular resistance in mmHg * s / mL
        "pulmonary arterial capacitance": 2.0,  # pulmonary arterial capacitance in mL / mmHg
        "pulmonary venous capacitance": 4.0,  # pulmonary venous capacitance in mL / mmHg
        "pulmonary arterial unstressed volume": 50,  # unstressed volume of pulmonary arterial system in mL
        "pulmonary venous unstressed volume": 450,  # unstressed volume of pulmonary arterial system in mL
        "pulmonary vascular resistance": 0.15,  # pulmonary vascular resistance in mmHg * s / mL
        "ventricle left unstressed volume": 200, # unstressed blood volume of left ventricle in mL
        "ventricle right unstressed volume": 200, # unstressed blood volume of right ventricle in mL
        "pouch left unstressed volume": 200, # unstressed water/air volume of left pouch in mL
        "pouch right unstressed volume": 200, # unstressed water/air volume of right pouch in mL
        "left ventricle elastance": 1.0, # elastance of left ventricle (if linear) in mmHg / mL
        "right ventricle elastance": 1.0,  # elastance of left ventricle (if linear) in mmHg / mL
        }

p_ref = data["mean arterial pressure"]
t_ref = 1 / (data["bpm"] / 60)
v_ref = data["stroke volume"]
f_ref = 1/ t_ref
e_ref = p_ref / v_ref
c_ref = 1 / e_ref
q_ref = v_ref / t_ref
r_ref = p_ref / q_ref

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
aortic_valve = Valve(data["resistance aortic valve"] / r_ref, Rclosed=1e16)

tricuspid_valve = Valve(data["resistance tricuspid valve"] / r_ref, Rclosed=1e16)

pulmonary_valve = Valve(data["resistance pulmonary valve"] / r_ref, Rclosed=1e16)

mitral_valve = Valve(data["resistance mitral valve"] / r_ref, Rclosed=1e16)

# systemic and pulmonary circulation

svr = lambda t: data["systemic vascular resistance"] / r_ref + Sigmoid(3.0 / r_ref, 40)(t) - Sigmoid(3.0 / r_ref, 41)(t)

systemic_circulation = Circulation(aortic_valve, tricuspid_valve, data["systemic arterial capacitance"] / c_ref,
                                   data["systemic arterial unstressed volume"] / v_ref, data["systemic venous capacitance"] / c_ref,
                                   data["systemic venous unstressed volume"] / v_ref, svr)

pulmonary_circulation = Circulation(pulmonary_valve, mitral_valve, data["pulmonary arterial capacitance"] / c_ref,
                                   data["pulmonary arterial unstressed volume"] / v_ref, data["pulmonary venous capacitance"] / c_ref,
                                   data["pulmonary venous unstressed volume"] / v_ref, data["pulmonary vascular resistance"] / r_ref)

circulations = Circulations(systemic_circulation, pulmonary_circulation)

# heart
left_ventricle = LinearMembrane(data["left ventricle elastance"] / e_ref, data["ventricle left unstressed volume"] / v_ref,
                                data["pouch left unstressed volume"] / v_ref)

right_ventricle = LinearMembrane(data["right ventricle elastance"] / e_ref, data["ventricle right unstressed volume"] / v_ref,
                                 data["pouch right unstressed volume"] / v_ref)

heart = Heart(left_ventricle, right_ventricle)

# system
system = BiVenSystem(actuation, heart, circulations)

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

initial_guess_volumes = (0, 0, vl0, vr0, vs10, vs20, vp10, vp20)

no_actuation_cycles = 120

# solve dynamics
times, volumes, pressures, flows, valves = system.dynamics(initial_guess_volumes, no_actuation_cycles)
valve_times, valve_values, state_aortic_valve, state_tricuspid_valve, state_pulmonary_valve, state_mitral_valve = valves

index = np.searchsorted(times, no_actuation_cycles-1, side="right")
idxv = np.searchsorted(valve_times, no_actuation_cycles-1, side="right")

haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = pressures
hsL, hsR = system.actuation.left.source(times), system.actuation.right.source(times)

vaL, vaR, vvL, vvR, vs1, vs2, vp1, vp2 = volumes


plt.figure()
[plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0]]
plt.plot(times, hsL, 'r--', label="Source left")
plt.plot(times, hsR, 'b--', label="Source right")
plt.plot(times, haL, 'r-.', label='Pouch left')
plt.plot(times, haR, 'b-.', label='Pouch right')
plt.plot(times, hvL, 'ro-', label='Ventricle left')
plt.plot(times, hvR, 'bo-', label='Ventricle right')
plt.plot(times, hs1, 'r-', label='Afterload left ventricle (aorta)')
plt.plot(times, hp2, 'r:', label='Preload left ventricle (LAP, pulmonary)')
plt.plot(times, hp1, 'b-', label='Afterload right ventricle (pulmonary artery)')
plt.plot(times, hs2, 'b:', label='Preload right ventricle (RAP, systemic)')
plt.axhline(0, color='k')
plt.xlabel("Time (s) / T0 (s)")
plt.ylabel("Pressure (mmHg) / MAP (mmHg)")
plt.legend()

plt.figure()
[plt.axvline(x * t_ref, linestyle='--', color='k', alpha=0.1) for x in valves[0]]
plt.plot(times * t_ref, hsL * p_ref, 'r--', label="Source left")
plt.plot(times * t_ref, hsR * p_ref, 'b--', label="Source right")
plt.plot(times * t_ref, haL * p_ref, 'r-.', label='Pouch left')
plt.plot(times * t_ref, haR * p_ref, 'b-.', label='Pouch right')
plt.plot(times * t_ref, hvL * p_ref, 'ro-', label='Ventricle left')
plt.plot(times * t_ref, hvR * p_ref, 'bo-', label='Ventricle right')
plt.plot(times * t_ref, hs1 * p_ref, 'r-', label='Afterload left ventricle (aorta)')
plt.plot(times * t_ref, hp2 * p_ref, 'r:', label='Preload left ventricle (LAP, pulmonary)')
plt.plot(times * t_ref, hp1 * p_ref, 'b-', label='Afterload right ventricle (pulmonary artery)')
plt.plot(times * t_ref, hs2 * p_ref, 'b:', label='Preload right ventricle (RAP, systemic)')
plt.axhline(0, color='k')
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.legend()

plt.figure()
[plt.axvline(x * t_ref, linestyle='--', color='k', alpha=0.1) for x in valves[0][idxv:]]
plt.plot(times[index:] * t_ref, hsL[index:] * p_ref, 'r--', label="Source left")
plt.plot(times[index:] * t_ref, hsR[index:] * p_ref, 'b--', label="Source right")
plt.plot(times[index:] * t_ref, haL[index:] * p_ref, 'r-.', label='Pouch left')
plt.plot(times[index:] * t_ref, haR[index:] * p_ref, 'b-.', label='Pouch right')
plt.plot(times[index:] * t_ref, hvL[index:] * p_ref, 'ro-', label='Ventricle left')
plt.plot(times[index:] * t_ref, hvR[index:] * p_ref, 'bo-', label='Ventricle right')
plt.plot(times[index:] * t_ref, hs1[index:] * p_ref, 'r-', label='Afterload left ventricle (aorta)')
plt.plot(times[index:] * t_ref, hp2[index:] * p_ref, 'r:', label='Preload left ventricle (LAP, pulmonary)')
plt.plot(times[index:] * t_ref, hp1[index:] * p_ref, 'b-', label='Afterload right ventricle (pulmonary artery)')
plt.plot(times[index:] * t_ref, hs2[index:] * p_ref, 'b:', label='Preload right ventricle (RAP, systemic)')
plt.axhline(0, color='k')
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.legend()

plt.figure()

# [plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0]]
plt.plot(times, state_aortic_valve, color='r', linestyle='-', label="aortic valve", alpha=0.2)
plt.plot(times, state_pulmonary_valve, color='b', linestyle='-', label="pulmonary valve", alpha=0.2)

plt.axhline(0, color='k')

plt.plot(times, vvL + vvR + vp1 + vp2 + vs1 + vs2, 'k--', label="total blood volume")

plt.axhline(heart.left.Vv0, linestyle=':', color='red', label="LVv0")
plt.axhline(heart.right.Vv0, linestyle=':', color='blue', label="RVv0")

plt.plot(times, vvL, 'r-', label="left ventricular volume")
plt.plot(times, vvR, 'b-', label="right ventricular volume")

plt.plot(times, vs1, 'r--', label="volume systemic arteries")
plt.plot(times, vp1, 'b--', label="volume pulmonary arteries")

plt.plot(times, vs2, 'r-.', label="volume systemic veins")
plt.plot(times, vp2, 'b-.', label="volume pulmonary veins")

plt.xlabel("Time (s) / TO (s)")
plt.ylabel("Volume (mL) / SV0 (mL)")
plt.legend()

plt.figure()

# [plt.axvline(x * t_ref, linestyle='--', color='k', alpha=0.1) for x in valves[0]]
plt.plot(times * t_ref, state_aortic_valve * v_ref, color='r', linestyle='-', label="aortic valve", alpha=0.2)
plt.plot(times * t_ref, state_pulmonary_valve * v_ref, color='b', linestyle='-', label="pulmonary valve", alpha=0.2)
plt.axhline(0, color='k')

plt.plot(times * t_ref, (vvL + vvR + vp1 + vp2 + vs1 + vs2) * v_ref, 'k--', label="total blood volume")

plt.axhline(heart.left.Vv0 * v_ref, linestyle=':', color='red', label="LVv0")
plt.axhline(heart.right.Vv0 * v_ref, linestyle=':', color='blue', label="RVv0")

plt.plot(times * t_ref, vvL * v_ref, 'r-', label="left ventricular volume")
plt.plot(times * t_ref, vvR * v_ref, 'b-', label="right ventricular volume")

plt.plot(times * t_ref, vs1 * v_ref, 'r--', label="volume systemic arteries")
plt.plot(times * t_ref, vp1 * v_ref, 'b--', label="volume pulmonary arteries")

plt.plot(times * t_ref, vs2 * v_ref, 'r-.', label="volume systemic veins")
plt.plot(times * t_ref, vp2 * v_ref, 'b-.', label="volume pulmonary veins")

plt.xlabel("Time (s)")
plt.ylabel("Volume (mL)")
plt.legend()

plt.figure()
plt.plot(times[index:] * t_ref, state_aortic_valve[index:] * v_ref, color='r', linestyle='-', label="aortic valve", alpha=0.2)
plt.plot(times[index:] * t_ref, state_pulmonary_valve[index:] * v_ref, color='b', linestyle='-', label="pulmonary valve", alpha=0.2)
# [plt.axvline(x * t_ref, linestyle='--', color='k', alpha=0.1) for x in valves[0][idxv:]]
plt.axhline(0, color='k')

plt.axhline(heart.left.Vv0 * v_ref, linestyle=':', color='red', label="LVv0")
plt.axhline(heart.right.Vv0 * v_ref, linestyle=':', color='blue', label="RVv0")

plt.plot(times[index:] * t_ref, vvL[index:] * v_ref, 'r-', label="left ventricular volume")
plt.plot(times[index:] * t_ref, vvR[index:] * v_ref, 'b-', label="right ventricular volume")

plt.plot(times[index:] * t_ref, vs1[index:] * v_ref, 'r--', label="volume systemic arteries")
plt.plot(times[index:] * t_ref, vp1[index:] * v_ref, 'b--', label="volume pulmonary arteries")

plt.xlabel("Time (s)")
plt.ylabel("Volume (mL)")
plt.legend()


aortic_valve_opening = find_zero_to_one(state_aortic_valve)
left_stroke_volume = state_aortic_valve * ejected_volume(vvL, aortic_valve_opening)
aortic_valve_closing = find_one_to_zero(state_aortic_valve)
idx_left = [i-1  for i, val in enumerate(aortic_valve_closing) if val == 1]

pulmonary_valve_opening = find_zero_to_one(state_pulmonary_valve)
right_stroke_volume = state_pulmonary_valve * ejected_volume(vvR, pulmonary_valve_opening)
pulmonary_valve_closing = find_one_to_zero(state_pulmonary_valve)
idx_right = [i-1 for i, val in enumerate(pulmonary_valve_closing) if val == 1]

t = np.linspace(0, no_actuation_cycles, 1000*no_actuation_cycles)
aligned_left = np.interp(t, times[idx_left], left_stroke_volume[idx_left])
aligned_right = np.interp(t, times[idx_right], right_stroke_volume[idx_right])
stroke_difference = aligned_left - aligned_right

afterload_left = np.interp(t, times[idx_left], hs1[idx_left])
afterload_right = np.interp(t, times[idx_right], hp1[idx_right])

preload_left = np.interp(t, times[idx_left], hs2[idx_left])
preload_right = np.interp(t, times[idx_right], hp2[idx_right])

pressure_ventricle_left = np.interp(t, times[idx_left], hvL[idx_left])
pressure_ventricle_right = np.interp(t, times[idx_right], hvR[idx_right])

v_ventricle_left = np.interp(t, times[idx_left], vvL[idx_left])
v_ventricle_right = np.interp(t, times[idx_right], vvR[idx_right])

plt.figure()
plt.plot(times[idx_left], hs1[idx_left], 'r-')
plt.plot(times[idx_right], hp1[idx_right], 'b-')
plt.plot(t, afterload_left, 'r--', label="afterload left")
plt.plot(t, afterload_right, 'b--', label="afterload right")

fig, ax = plt.subplots()
plt.title("Absolute value of state variables (one datapoint per cycle) on log scale")
plt.semilogy(t, np.abs(afterload_left - afterload_left[-1]), 'r-', label="LV afterload")
plt.semilogy(t, np.abs(afterload_right - afterload_right[-1]), 'b-', label="RV afterload")
plt.semilogy(t, np.abs(preload_left - preload_left[-1]), 'r--', label="LV preload")
plt.semilogy(t, np.abs(preload_right - preload_right[-1]), 'b--', label="RV preload")
plt.semilogy(t, np.abs(stroke_difference), 'k-', label="SV imbalance")
plt.semilogy(t, np.abs(pressure_ventricle_left - pressure_ventricle_left[-1]), 'r:', label="LV pressure")
plt.semilogy(t, np.abs(pressure_ventricle_right - pressure_ventricle_right[-1]), 'b:', label="RV pressure")
plt.semilogy(t, np.abs(v_ventricle_left - v_ventricle_left[-1]), 'r-.', label="LV volume")
plt.semilogy(t, np.abs(v_ventricle_right - v_ventricle_right[-1]), 'b-.', label="RV volume")
plt.legend()
plt.xlabel("Time (s) / TO (s)")

# idd = np.where(t > 20.0)[0][0]
# id0 = np.argmax(stroke_difference)
# exp_decay = lambda t, tau: stroke_difference[id0] * np.exp(-t/tau)
# tau, _ = curve_fit(exp_decay, t[id0:], stroke_difference[id0:], 1.0)

plt.figure()
plt.axhline(0, color='k', alpha=0.1)
plt.plot(times[idx_left], left_stroke_volume[idx_left], 'ro-', label="lsv", alpha=0.2)
plt.plot(times[idx_right], right_stroke_volume[idx_right], 'bo-', label="rsv", alpha=0.2)
plt.plot(t, aligned_left, 'r--', label="aligned left")
plt.plot(t, aligned_right, 'b--', label="aligned right")
plt.plot(t, stroke_difference, 'k-', label="stroke difference")
plt.xlabel("Time (s) / TO (s)")
plt.ylabel("Volume (mL) / SV0 (mL)")
plt.legend()

fig, ax = plt.subplots()
ax.plot(t, stroke_difference, 'k-')
ax.set_yscale('symlog', linthresh=1e-6)
plt.xlabel("Time (s) / TO (s)")
plt.ylabel("Volume (mL) / SV0 (mL)")

plt.figure()
plt.axhline(0, color='k', alpha=0.1)
plt.plot(times[idx_left] * t_ref, left_stroke_volume[idx_left] * v_ref, 'ro-', label="lsv", alpha=0.2)
plt.plot(times[idx_right] * t_ref, right_stroke_volume[idx_right] * v_ref, 'bo-', label="rsv", alpha=0.2)
plt.plot(t * t_ref, aligned_left * v_ref, 'r--', label="aligned left")
plt.plot(t * t_ref, aligned_right * v_ref, 'b--', label="aligned right")
plt.plot(t * t_ref, stroke_difference * v_ref, 'k-', label="stroke difference")
plt.xlabel("Time (s)")
plt.ylabel("Volume (mL)")
plt.legend()

fig, ax = plt.subplots()
ax.plot(t * t_ref, stroke_difference * v_ref, 'k-')
ax.set_yscale('symlog', linthresh=1e-6 * v_ref)
plt.xlabel("Time (s)")
plt.ylabel("Volume (mL)")

plt.figure()
plt.plot(vvL, hvL, 'r-', label="left ventricle PV")
plt.plot(heart.left.Vp0 + heart.left.Vv0 - vvL, haL, 'r--', label="left pouch PV", alpha=0.1)
plt.axvline(heart.left.Vp0, linestyle="dotted", color='red', label="initial left pouch volume")
plt.axvline(heart.left.Vv0, linestyle="dashed", color="red", label="initial left ventricle volume")

plt.plot(vvR, hvR, 'b-', label="right ventricle PV")
plt.plot(heart.right.Vp0 + heart.right.Vv0 - vvR, haR, 'b--', label="right pouch PV", alpha=0.1)
plt.axvline(heart.right.Vp0, linestyle="dotted", color='blue', label="initial right pouch volume")
plt.axvline(heart.right.Vv0, linestyle="dashed", color="blue", label="initial right ventricle volume")

plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.xlabel("Volume (mL) / SV0 (mL)")
plt.ylabel("Pressure (mmHg) / MAP (mmHg)")
plt.legend()

plt.figure()
plt.plot(vvL * v_ref, hvL * p_ref, 'r-', label="left ventricle PV")
plt.plot((heart.left.Vp0 + heart.left.Vv0 - vvL) * v_ref, haL * p_ref, 'r--', label="left pouch PV", alpha=0.1)
plt.axvline(heart.left.Vp0 * v_ref, linestyle="dotted", color='red', label="initial left pouch volume")
plt.axvline(heart.left.Vv0 * v_ref, linestyle="dashed", color="red", label="initial left ventricle volume")

plt.plot(vvR * v_ref, hvR * p_ref, 'b-', label="right ventricle PV")
plt.plot((heart.right.Vp0 + heart.right.Vv0 - vvR) * v_ref, haR * p_ref, 'b--', label="right pouch PV", alpha=0.1)
plt.axvline(heart.right.Vp0 * v_ref, linestyle="dotted", color='blue', label="initial right pouch volume")
plt.axvline(heart.right.Vv0 * v_ref, linestyle="dashed", color="blue", label="initial right ventricle volume")

plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.xlabel("Volume (mL)")
plt.ylabel("Pressure (mmHg)")
plt.legend()

plt.figure()
plt.plot(vvL[index:] * v_ref, hvL[index:] * p_ref, 'r-', label="left ventricle PV")
plt.plot((heart.left.Vp0 + heart.left.Vv0 - vvL[index:]) * v_ref, haL[index:] * p_ref, 'r--', label="left pouch PV", alpha=0.1)
plt.axvline(heart.left.Vp0 * v_ref, linestyle="dotted", color='red', label="initial left pouch volume")
plt.axvline(heart.left.Vv0 * v_ref, linestyle="dashed", color="red", label="initial left ventricle volume")

plt.plot(vvR[index:] * v_ref, hvR[index:] * p_ref, 'b-', label="right ventricle PV")
plt.plot((heart.right.Vp0 + heart.right.Vv0 - vvR[index:]) * v_ref, haR[index:] * p_ref, 'b--', label="right pouch PV", alpha=0.1)
plt.axvline(heart.right.Vp0 * v_ref, linestyle="dotted", color='blue', label="initial right pouch volume")
plt.axvline(heart.right.Vv0 * v_ref, linestyle="dashed", color="blue", label="initial right ventricle volume")

plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.xlabel("Volume (mL)")
plt.ylabel("Pressure (mmHg)")
plt.legend()

plt.show()