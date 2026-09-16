from math import pi

from tahs import LinearMembrane, NonlinearMembrane
from models import data0, Source, Actuator, Actuation, Circulation, Circulations, Heart, Valve, BiVenSystem, find_one_to_zero, find_zero_to_one, ejected_volume
from matplotlib import pyplot as plt

data = data0

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
aortic_valve = Valve(data["resistance aortic valve"] / r_ref)

tricuspid_valve = Valve(data["resistance tricuspid valve"] / r_ref)

pulmonary_valve = Valve(data["resistance pulmonary valve"] / r_ref)

mitral_valve = Valve(data["resistance mitral valve"] / r_ref)

# systemic and pulmonary circulation
systemic_circulation = Circulation(aortic_valve, tricuspid_valve, data["systemic arterial capacitance"] / c_ref,
                                   data["systemic arterial unstressed volume"] / v_ref, data["systemic venous capacitance"] / c_ref,
                                   data["systemic venous unstressed volume"] / v_ref, data["systemic vascular resistance"] / r_ref)

pulmonary_circulation = Circulation(pulmonary_valve, mitral_valve, data["pulmonary arterial capacitance"] / c_ref,
                                   data["pulmonary arterial unstressed volume"] / v_ref, data["pulmonary venous capacitance"] / c_ref,
                                   data["pulmonary venous unstressed volume"] / v_ref, data["pulmonary vascular resistance"] / r_ref)

circulations = Circulations(systemic_circulation, pulmonary_circulation)

# heart
# left_ventricle = LinearMembrane(data["left ventricle elastance"] / e_ref, data["ventricle left unstressed volume"] / v_ref,
#                                 data["pouch left unstressed volume"] / v_ref)
#
# right_ventricle = LinearMembrane(data["right ventricle elastance"] / e_ref, data["ventricle right unstressed volume"] / v_ref,
#                                  data["pouch right unstressed volume"] / v_ref)

left_ventricle = NonlinearMembrane(0.001, 1, data["ventricle left unstressed volume"] / v_ref,
                                data["pouch left unstressed volume"] / v_ref)
right_ventricle = NonlinearMembrane(0.001, 0.1, data["ventricle right unstressed volume"] / v_ref,
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

initial_guess_volumes = (0.0, 0.0, vl0, vr0, vs10, vs20, vp10, vp20)

no_actuation_cycles = 20

# solve dynamics
times, volumes, pressures, flows, valves = system.dynamics(initial_guess_volumes, no_actuation_cycles)

valve_times, valve_values, state_aortic_valve, state_tricuspid_valve, state_pulmonary_valve, state_mitral_valve = valves

plt.figure()

haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = pressures
hsL, hsR = system.actuation.left.source(times), system.actuation.right.source(times)

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

vaL, vaR, vvL, vvR, vs1, vs2, vp1, vp2 = volumes

[plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0]]
plt.axhline(heart.left.Vv0, linestyle='--', color='black', label="LVv0")
plt.plot(times, vvL, 'k-', label="left ventricular volume")
plt.axhline(heart.right.Vv0, linestyle='--', color='blue', label="RVv0")
plt.plot(times, vvR, 'b-', label="right ventricular volume")
plt.plot(times, vp1, 'y-', label="volume pulmonary arteries")
plt.plot(times, vs1, 'r-', label="volume systemic arteries")

plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.plot(times, vs2, 'g-', label="volume systemic veins")
plt.plot(times, vp2, 'c-', label="volume pulmonary veins")
plt.plot(times, vvL + vvR + vp1 + vp2 + vs1 + vs2)
plt.axhline(5000 / v_ref)

plt.xlabel("Time (s) / TO (s)")
plt.ylabel("Volume (mL) / SV0 (mL)")
plt.legend()

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
plt.xlabel("Time (s) / TO (s)")
plt.ylabel("Volume (mL) / SV0 (mL)")
plt.legend()

plt.figure()
plt.plot(vvL, hvL, 'r-', label="left ventricle PV")
plt.plot(heart.left.Vp0 + heart.left.Vv0 - vvL, haL, 'r--', label="left pouch PV", alpha=0.1)
plt.axvline(heart.left.Vp0, linestyle="dotted", color='red', label="initial left pouch volume")
plt.axvline(heart.left.Vv0, linestyle="dashed", color="red", label="initial left ventricle volume")

plt.plot(vvR, hvR, 'k-', label="right ventricle PV")
plt.plot(heart.right.Vp0 + heart.right.Vv0 - vvR, haR, 'k--', label="right pouch PV", alpha=0.1)
plt.axvline(heart.right.Vp0, linestyle="dotted", color='black', label="initial right pouch volume")
plt.axvline(heart.right.Vv0, linestyle="dashed", color="black", label="initial right ventricle volume")

plt.axhline(0, color='k')
plt.axvline(0, color='k')

plt.xlabel("Volume (mL) / SV0 (mL)")
plt.ylabel("Pressure (mmHg) / MAP (mmHg)")
plt.legend()

plt.figure()

qaL, qaR, qvL, qvR, qs1, qs2, qp1, qp2 = flows
qsL, qsR = system.source_flow(times, haL, haR)

[plt.axvline(x, linestyle='--', color='k', alpha=0.1) for x in valves[0]]
plt.plot(times, qsL, 'r--', label='Source left')
plt.plot(times, qsR, 'b--', label='Source right')
plt.plot(times, qvL, 'r-', label='Ventricle left')
plt.plot(times, qvR, 'b-', label='Ventricle right')
plt.axhline(0, color='k')
plt.xlabel("Time (s) / T0 (s)")
plt.ylabel("Flow rate (mL/s) / q0 (mL/s)")
plt.legend()

plt.show()