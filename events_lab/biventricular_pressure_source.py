from math import pi
import numpy as np
from sympy.physics.units import impedance

from tahs import TAH, LinearMembrane, NonlinearMembrane
from utils import Sigmoid
from scipy.signal import square
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from motors import DCM
from pumps import CP
from utils import event
from copy import deepcopy

class Valve:
    def __init__(self, Ropen:float=0.1, Rclosed:float=1e4, dhopen:float=4, dhclose:float=0.5, initial_state: int=0):
        self.Ropen = Ropen
        self.Rclosed = Rclosed
        self.dhopenref = dhopen
        self.dhcloseref = dhclose
        self.dhopen = 1.0 * dhopen
        self.dhclose = 1.0 *dhclose
        self.state: int = initial_state

    def open(self):
        self.state = 1
        self.dhopen = 1000
        self.dhclose = 1.0 * self.dhcloseref

    def close(self):
        self.state = 0
        self.dhopen = 1.0 * self.dhopenref
        self.dhclose = 1000

    def event_open(self, dh):
        return dh - self.dhopen

    def event_close(self, dh):
        return dh - self.dhclose

    def __call__(self) -> float:
        return self.Rclosed if self.state == 0 else self.Ropen

class Source:
    def __init__(self, magnitude: float = 1.0, freq: float = 1, phase: float = pi, duty: float = 0.5):
        self.magnitude = magnitude
        self.freq = freq
        self.phase = phase
        self.duty = duty

    def __call__(self, t):
        return self.magnitude * (0.5 * square(2 * pi * self.freq * t - self.phase, duty=self.duty) + 0.5)

class Circuit:
    def __init__(self,
                 SL: Source = Source(3.0, pi/2, 0.5),
                 SR: Source = Source(5.0, pi, 0.5),
                 CL: float = 0.01,
                 CR: float = 0.01,
                 RL: float = 100,
                 RR: float = 100):
        self.SL = SL
        self.SR = SR
        self.CL = CL
        self.CR = CR
        self.RL = RL
        self.RR = RR

class TCM:

    def __init__(self, valve_in: Valve = Valve(),
                 valve_out: Valve = Valve(),
                 C1: float = 0.1,
                 C2: float = 0.1,
                 R: float = 10):
        self.C1 = C1 # L / m
        self.C2 = C2
        self.R = R # m / L/min
        self.valve_in = valve_in
        self.valve_out = valve_out

class SCM:
    def __init__(self, pc: TCM = TCM(), sc: TCM = TCM()):
        self.pc = pc
        self.sc = sc

class BiVenSystem:
    def __init__(self, circuit: Circuit = Circuit(),
                 tahL: TAH = LinearMembrane(),
                 tahR: TAH = LinearMembrane(),
                 hemo: SCM = SCM()):
        self.circuit = circuit
        self.tahL = tahL
        self.tahR = tahR
        self.hemo = hemo

    def solve(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y


        hvL = self.tahL.pressure(haL, vvL)
        hvR = self.tahR.pressure(haR, vvR)

        qso = (hs2 - hvR) / self.hemo.sc.valve_out()
        qpi = (hvR - hp1) / self.hemo.pc.valve_in()
        qp = (hp1 - hp2) / self.hemo.pc.R
        qpo = (hp2 - hvL) / self.hemo.pc.valve_out()
        qsi = (hvL - hs1) / self.hemo.sc.valve_in()
        qs = (hs1 - hs2) / self.hemo.sc.R

        dhs1 = (qsi - qs) / self.hemo.sc.C1
        dhs2 = (qs - qso) / self.hemo.sc.C2
        dhp1 = (qpi - qp) / self.hemo.pc.C1
        dhp2 = (qp - qpo) / self.hemo.pc.C2

        dvvL = qpo - qsi
        qaL = -1.0 * dvvL

        dvvR = qso - qpi
        qaR = -1.0 * dvvR

        qsL = (self.circuit.SL(t) - haL) / self.circuit.RL
        dhaL = (qsL - qaL) / self.circuit.CL

        qsR = (self.circuit.SR(t) - haR) / self.circuit.RR
        dhaR = (qsR - qaR) / self.circuit.CR

        return [dhaL, dhaR, dvvL, dvvR, dhp1, dhp2, dhs1, dhs2]


    @event(direction=1)
    def event_valve_systemic_in_opening(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_in.event_open(self.tahL.pressure(haL, vvL) - hs1)

    @event(direction=-1)
    def event_valve_systemic_in_closing(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_in.event_close(self.tahL.pressure(haL, vvL) - hs1)

    @event(direction=1)
    def event_valve_systemic_out_opening(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_out.event_open(hs2 - self.tahR.pressure(haR, vvR))

    @event(direction=-1)
    def event_valve_systemic_out_closing(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_out.event_close(hs2 - self.tahR.pressure(haR, vvR))

    @event(direction=1)
    def event_valve_pulmonary_in_opening(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_in.event_open(self.tahR.pressure(haR, vvR) - hp1)

    @event(direction=-1)
    def event_valve_pulmonary_in_closing(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_in.event_close(self.tahR.pressure(haR, vvR) - hp1)

    @event(direction=1)
    def event_valve_pulmonary_out_opening(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_out.event_open(hp2 - self.tahL.pressure(haL, vvL))

    @event(direction=-1)
    def event_valve_pulmonary_out_closing(self, t, y):
        haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_out.event_close(hp2 - self.tahL.pressure(haL, vvL))

# pressure source of left ventricle has magnitude of 50 kPa (pressure head of 5 m)
# pressure source of right ventricle has magnitude of 30 kPa (pressure head of 3 m)
# duty cycle: 1/3 systole, 2/3 diastole
freq = 1.0 # frequency: 1 Hz aka 60 BPM
duty = 1/3
pressure_source_left = Source(magnitude=5.0, freq=freq, phase=0, duty=duty)
pressure_source_right = Source(magnitude=3.0, freq=freq, phase=0, duty=duty)

# capacitance and resistance between pressure source and artificial ventricle
inlet_capacitance_left = 0.05 # L/m
inlet_capacitance_right = 0.05 # L/m
inlet_resistance_left = 1 # m / L/s
inlet_resistance_right = 1 # m / L/s

circuit = Circuit(SL=pressure_source_left, SR=pressure_source_right,
                  CL=inlet_capacitance_left, CR=inlet_capacitance_right,
                  RL=inlet_resistance_left, RR=inlet_resistance_right)

mitral_valve = Valve(Ropen=1, Rclosed=1e4, dhopen=0.0, dhclose=0.0)
aortic_valve = deepcopy(mitral_valve)
tricuspid_valve = deepcopy(mitral_valve)
pulmonary_valve = deepcopy(mitral_valve)

# systemic circulation
# TPR = (aortic pressure - RAP) / CO approx 120 - 2 mmHg / 5 L/min
# TPR = 16 kpa / 5 L/min = 1.6 m / 0.083 L/s =approx 20 m / L/s

# capacitance in mL/mmHg
# aortic: 1-2
# systemic vascular: 10-200
# pulmonary arterial 2-4
# pulmonary vascular 4-6

# mL/mmHg to L/m is x 0.075

# aortic: 2 * 0.075 = 0.15
# systemic: 200 * 0.075 = 15
# pulmonary arterial: 4 * 0.075 = 0.3
# pulmonary vascular: 6 * 0.075 = 0.45

systemic_circulation = TCM(valve_in=aortic_valve, valve_out=tricuspid_valve, C1=0.15, C2=15, R=20)

# pulmonary resistance approx TPR/10
pulmonary_circulation = TCM(valve_in=pulmonary_valve, valve_out=mitral_valve, C1=0.3, C2=0.45, R=2)

hemo = SCM(pc=pulmonary_circulation, sc=systemic_circulation)

# elastance of heart approx E = dp/dv = 2m / 0.1 L = 20 m/L
tahL = NonlinearMembrane(a=1, b=0, Vv0=0.2, Vp0=0.2)
tahR = NonlinearMembrane(a=1, b=0, Vv0=0.2, Vp0=0.2)

system = BiVenSystem(circuit=circuit, tahL=tahL, tahR=tahR, hemo=hemo)

events = [system.event_valve_systemic_in_opening, system.event_valve_systemic_in_closing,
          system.event_valve_systemic_out_opening, system.event_valve_systemic_out_closing,
          system.event_valve_pulmonary_in_opening, system.event_valve_pulmonary_in_closing,
          system.event_valve_pulmonary_out_opening, system.event_valve_pulmonary_out_closing
          ]

# haL, haR, vvL, vvR, hp1, hp2, hs1, hs2
initial_state = (0.0, 0.0, tahL.Vv0, tahR.Vv0, 0.0, 0.0, 0.0, 0.0)

t_start = 0.0
t_end = 4

t_full = []
y_full = []
valve_pcin_state = []
valve_pcout_state = []
valve_scin_state = []
valve_scout_state = []
derivatives = []

event_times = []

while t_start < t_end:
    sol = solve_ivp(system.solve, [t_start, t_end], initial_state, events=events, rtol=1e-9, atol=1e-9)
    derivatives.append(system.solve(sol.t, sol.y))

    t_full.append(sol.t)
    y_full.append(sol.y)
    valve_pcin_state.append(system.hemo.pc.valve_in.state * np.ones_like(sol.t))
    valve_pcout_state.append(system.hemo.pc.valve_out.state * np.ones_like(sol.t))
    valve_scin_state.append(system.hemo.sc.valve_in.state * np.ones_like(sol.t))
    valve_scout_state.append(system.hemo.sc.valve_out.state * np.ones_like(sol.t))

    if any([i.size > 0 for i in sol.t_events]):

        event = next(i for i, j in enumerate(sol.t_events) if len(j))
        if event == 0:
            system.hemo.sc.valve_in.open()
        elif event == 1:
            system.hemo.sc.valve_in.close()
        elif event == 2:
            system.hemo.sc.valve_out.open()
        elif event == 3:
            system.hemo.sc.valve_out.close()
        elif event == 4:
            system.hemo.pc.valve_in.open()
        elif event == 5:
            system.hemo.pc.valve_in.close()
        elif event == 6:
            system.hemo.pc.valve_out.open()
        elif event == 7:
            system.hemo.pc.valve_out.close()
        else:
            print("no valid event")

        event_time = sol.t_events[event][0]
        event_times.append(event_time)
        print(event_time)
        print(event)
        t_start = event_time
        initial_state = sol.y_events[event][0]
    else:
        t_start = t_end


t_full = np.concatenate(t_full)
y_full = np.concatenate(y_full, axis=1)
valve_pcin_state = np.concatenate(valve_pcin_state)
valve_pcout_state = np.concatenate(valve_pcout_state)
valve_scin_state = np.concatenate(valve_scin_state)
valve_scout_state = np.concatenate(valve_scout_state)

derivatives = np.concatenate(derivatives, axis=1)

haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y_full
dhaL, dhaR, dvvL, dvvR, dhp1, dhp2, dhs1, dhs2 = derivatives

plt.figure()

hvL = tahL.pressure(haL, vvL)
hvR = tahR.pressure(haR, vvR)

plt.plot(t_full, circuit.SL(t_full), 'r--', label="Source left")
plt.plot(t_full, circuit.SR(t_full), 'b--', label="Source right")

plt.plot(t_full, haL, 'r-.', label='Pouch left')
plt.plot(t_full, haR, 'b-.', label='Pouch right')

plt.plot(t_full, hvL, 'ro-', label='Ventricle left')
plt.plot(t_full, hvR, 'bo-', label='Ventricle right')

plt.plot(t_full, hs1, 'r-', label='Afterload left ventricle (aorta)')
plt.plot(t_full, hp2, 'r:', label='Preload left ventricle (LAP, pulmonary)')

plt.plot(t_full, hp1, 'b-', label='Afterload right ventricle (pulmonary artery)')
plt.plot(t_full, hs2, 'b:', label='Preload right ventricle (RAP, systemic)')

plt.xlabel("Time (s)")
plt.ylabel("Pressure head (m)")
plt.legend()

plt.figure()
qaL = -1.0 * dvvL
qaR = -1.0 * dvvR

qsL = (circuit.SL(t_full) - haL) / circuit.RL
qsR = (circuit.SR(t_full) - haR) / circuit.RR

plt.plot(t_full, 60 * qsL, 'r--', label='Source left')
plt.plot(t_full, 60 * qsR, 'b--', label='Source right')
#
# plt.plot(t_full, 60 * qaL, 'r-.', label='Pouch left')
# plt.plot(t_full, 60 * qaR, 'b-.', label='Pouch right')

plt.plot(t_full, 60 * qaL, 'r-', label='Ventricle left')
plt.plot(t_full, 60 * qaR, 'b-', label='Ventricle right')

plt.axhline(0, color='k')

plt.xlabel("Time (s)")
plt.ylabel("Flow rate (L/min)")
plt.legend()

plt.figure()
plt.axhline(tahL.Vv0, linestyle='--', color='black', label="LVv0")
plt.plot(t_full, vvL, 'k-', label="left ventricular volume")
plt.axhline(tahR.Vv0, linestyle='--', color='blue', label="RVv0")
plt.plot(t_full, vvR, 'b-', label="right ventricular volume")
plt.plot(t_full, tahL.Vv0 - vvL, label="L-DV")
plt.plot(t_full, tahR.Vv0 - vvR, label="R-DV")
plt.plot(t_full, circuit.CL * haL, 'k--', label="left capacitor volume")
plt.plot(t_full, circuit.CR * haR, 'b--', label="right capacitor volume")
plt.plot(t_full, hemo.pc.C1 * hp1, 'y-', label="volume pulmonary 1")
plt.plot(t_full, hemo.pc.C2 * hp2, 'c-', label="volume pulmonary 2")
plt.plot(t_full, hemo.sc.C1 * hs1, 'r-', label="volume systemic 1")
plt.plot(t_full, hemo.sc.C2 * hs2, 'g-', label="volume systemic 2")
plt.xlabel("Time (s)")
plt.ylabel("Volume (L)")
plt.legend()


plt.figure()
plt.plot(vvL, hvL, 'r-', label="left ventricle PV")
plt.plot(tahL.Vp0 + tahL.Vv0 - vvL, haL, 'r--', label="left pouch PV")
plt.axvline(tahL.Vp0, linestyle="dotted", color='red', label="initial left pouch volume")
plt.axvline(tahL.Vv0, linestyle="dashed", color="red", label="initial left ventricle volume")

plt.plot(vvR, hvR, 'k-', label="right ventricle PV")
plt.plot(tahR.Vp0 + tahR.Vv0 - vvR, haR, 'k--', label="right pouch PV")
plt.axvline(tahR.Vp0, linestyle="dotted", color='black', label="initial right pouch volume")
plt.axvline(tahR.Vv0, linestyle="dashed", color="black", label="initial right ventricle volume")
plt.legend()

plt.figure()
plt.plot(tahL.Vv0 - vvL, haL - hvL, label="L-DH-DV")
plt.plot(tahR.Vv0 - vvR, haR - hvR, label="R-DH-DV")
plt.axis("equal")
plt.legend()

plt.show()