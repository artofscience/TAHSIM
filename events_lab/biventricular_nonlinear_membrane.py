from math import pi
import numpy as np
from sympy.physics.units import impedance

from tahs import TAH, LinearMembrane, NonlinearMembrane
from utils import Sigmoid
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


class Circuit:
    def __init__(self, hvalve: Valve = Valve(),
                 C: float = 1.0,
                 CL: float = 0.01,
                 CR: float = 0.01,
                 R: float = 100,
                 L: float = 0.01,
                 RinL: float = 0.01,
                 RinR: float = 0.01,
                 RoutL: float = 0.01,
                 RoutR: float = 0.01):
        self.L = L
        self.C = C
        self.CL = CL
        self.CR = CR
        self.R = R
        self.hvalve = hvalve
        self.RinL = RinL
        self.RinR = RinR
        self.RoutL = RoutL
        self.RoutR = RoutR

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
    def __init__(self, motor: DCM = DCM(),
                 pump: CP = CP(),
                 circuit: Circuit = Circuit(),
                 tahL: TAH = LinearMembrane(),
                 tahR: TAH = LinearMembrane(),
                 hemo: SCM = SCM()):
        self.motor = motor
        self.pump = pump
        self.circuit = circuit
        self.tahL = tahL
        self.tahR = tahR
        self.hemo = hemo

    def solve(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y

        # di(i, w, v(t))
        di = (self.motor.voltage(t) - self.motor.R * current - self.motor.kt * speed) / self.motor.L

        # tau(w, q)
        torque = self.pump.torque(speed, pump_capacity)

        # dw(i, w, tau(w, q))
        dw = (self.motor.kt * current - self.motor.mu * speed - torque) / self.motor.M

        # h(q, w)
        pump_head = self.pump.hq(speed, pump_capacity)

        # hx(hal, har)
        hx = (haL / self.circuit.RoutL + haR / self.circuit.RoutR - pump_capacity ) / (1/ self.circuit.RoutL + 1/self.circuit.RoutR)
        qoutL = (haL - hx) / self.circuit.RoutL
        qoutR = (haR - hx) / self.circuit.RoutR

        # dq(h(q,w), h1)
        hr = self.circuit.R * pump_capacity
        impedance_head = (hx - hr + pump_head) - h1
        dq_pump = impedance_head / self.circuit.L

        # dh1(q, h1, ha)
        Rhv = self.circuit.hvalve()
        ha = (h1 / Rhv + haL / self.circuit.RinL + haR / self.circuit.RinR) / (1 / Rhv + 1 / self.circuit.RinR + 1/self.circuit.RinL)

        qinL = (ha - haL) / self.circuit.RinL
        qinR = (ha - haR) / self.circuit.RinR

        qhv = (h1 - ha) / Rhv
        qc1 = pump_capacity - qhv
        dh1 = qc1 / self.circuit.C

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

        qcL = qinL - qoutL - qaL
        qcR = qinR - qoutR - qaR

        dhaL = qcL / self.circuit.CL
        dhaR = qcR / self.circuit.CR

        return [di, dw, dq_pump, dh1, dhaL, dhaR, dvvL, dvvR, dhp1, dhp2, dhs1, dhs2]

    @event(direction=1)
    def event_valve_opening(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        Rhv = self.circuit.hvalve()
        ha = (h1 / Rhv + haL / self.circuit.RinL + haR / self.circuit.RinR) / (1 / Rhv + 1 / self.circuit.RinR + 1/self.circuit.RinL)
        return self.circuit.hvalve.event_open(h1 - ha)

    @event(direction=-1)
    def event_valve_closing(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        Rhv = self.circuit.hvalve()
        ha = (h1 / Rhv + haL / self.circuit.RinL + haR / self.circuit.RinR) / (1 / Rhv + 1 / self.circuit.RinR + 1/self.circuit.RinL)
        return self.circuit.hvalve.event_close(h1 - ha)

    @event(direction=1)
    def event_valve_systemic_in_opening(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_in.event_open(self.tahL.pressure(haL, vvL) - hs1)

    @event(direction=-1)
    def event_valve_systemic_in_closing(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_in.event_close(self.tahL.pressure(haL, vvL) - hs1)

    @event(direction=1)
    def event_valve_systemic_out_opening(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_out.event_open(hs2 - self.tahR.pressure(haR, vvR))

    @event(direction=-1)
    def event_valve_systemic_out_closing(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.sc.valve_out.event_close(hs2 - self.tahR.pressure(haR, vvR))

    @event(direction=1)
    def event_valve_pulmonary_in_opening(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_in.event_open(self.tahR.pressure(haR, vvR) - hp1)

    @event(direction=-1)
    def event_valve_pulmonary_in_closing(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_in.event_close(self.tahR.pressure(haR, vvR) - hp1)

    @event(direction=1)
    def event_valve_pulmonary_out_opening(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_out.event_open(hp2 - self.tahL.pressure(haL, vvL))

    @event(direction=-1)
    def event_valve_pulmonary_out_closing(self, t, y):
        current, speed, pump_capacity, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y
        return self.hemo.pc.valve_out.event_close(hp2 - self.tahL.pressure(haL, vvL))

voltage = lambda t: Sigmoid(2, 1.0, k=5)(t)
motor = DCM(voltage, R=0.2, L=0.5, M=3.88/1e6, kt=5.9/1000, mu=12/1e7)
pump = CP(hm0=2.4, qn0=1.6, hn0=1.8, w0=1770 * (2* pi / 60), effn=0.35)
oscillator = Valve(Ropen=3, Rclosed=1e4, dhopen=4, dhclose=1.5)
circuit = Circuit(oscillator, C=0.5, CL=0.01, CR=0.01, R=10, L=1.0, RinL=0.5, RinR=1e4, RoutL=0.5, RoutR=0.5)
# tahL = LinearMembrane(E=1, Vv0=1, Vp0=0.9)
# tahR = LinearMembrane(E=2, Vv0=0.8, Vp0=0.7)
tahL = NonlinearMembrane(a=0.1, b=100, Vv0=1, Vp0=0.9)
tahR = NonlinearMembrane(a=0.5, b=10, Vv0=0.8, Vp0=0.7)
heart_valve = Valve(Ropen=1, Rclosed=1e4, dhopen=0.1, dhclose=0.0)
hemo_pc = TCM(heart_valve, deepcopy(heart_valve), C1=0.1, C2=0.5, R=5)
hemo_sc = TCM(deepcopy(heart_valve), deepcopy(heart_valve), C1=0.2, C2=1.0, R=20)
hemo = SCM(hemo_pc, hemo_sc)
system = BiVenSystem(motor=motor, pump=pump, circuit=circuit, tahL=tahL, tahR=tahR, hemo=hemo)

events = [system.event_valve_opening, system.event_valve_closing,
          system.event_valve_systemic_in_opening, system.event_valve_systemic_in_closing,
          system.event_valve_systemic_out_opening, system.event_valve_systemic_out_closing,
          system.event_valve_pulmonary_in_opening, system.event_valve_pulmonary_in_closing,
          system.event_valve_pulmonary_out_opening, system.event_valve_pulmonary_out_closing
          ]

# current, speed, pump_capacity, circuit_head, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2
initial_state = (0.0, 1e-6, 1e-6, 3.0, 3.0, 3.0, tahL.Vv0, tahR.Vv0, 0.0, 0.0, 0.0, 0.0)

t_start = 0.0
t_end = 20

t_full = []
y_full = []
hvalve_state = []
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
    hvalve_state.append(system.circuit.hvalve.state * np.ones_like(sol.t))
    valve_pcin_state.append(system.hemo.pc.valve_in.state * np.ones_like(sol.t))
    valve_pcout_state.append(system.hemo.pc.valve_out.state * np.ones_like(sol.t))
    valve_scin_state.append(system.hemo.sc.valve_in.state * np.ones_like(sol.t))
    valve_scout_state.append(system.hemo.sc.valve_out.state * np.ones_like(sol.t))

    if any([i.size > 0 for i in sol.t_events]):

        event = next(i for i, j in enumerate(sol.t_events) if len(j))
        if event == 0:
            system.circuit.hvalve.open()
        elif event == 1:
            system.circuit.hvalve.close()
        elif event == 2:
            system.hemo.sc.valve_in.open()
        elif event == 3:
            system.hemo.sc.valve_in.close()
        elif event == 4:
            system.hemo.sc.valve_out.open()
        elif event == 5:
            system.hemo.sc.valve_out.close()
        elif event == 6:
            system.hemo.pc.valve_in.open()
        elif event == 7:
            system.hemo.pc.valve_in.close()
        elif event == 8:
            system.hemo.pc.valve_out.open()
        elif event == 9:
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


length = 24
offset = 5
m = np.s_[-length-offset:-offset-1]

t_full = np.concatenate(t_full)
y_full = np.concatenate(y_full, axis=1)
hvalve_state = np.concatenate(hvalve_state)
valve_pcin_state = np.concatenate(valve_pcin_state)
valve_pcout_state = np.concatenate(valve_pcout_state)
valve_scin_state = np.concatenate(valve_scin_state)
valve_scout_state = np.concatenate(valve_scout_state)

derivatives = np.concatenate(derivatives, axis=1)

i, w, qp, h1, haL, haR, vvL, vvR, hp1, hp2, hs1, hs2 = y_full
di, dw, dqp, dh1, dhaL, dhaR, dvvL, dvvR, dhp1, dhp2, dhs1, dhs2 = derivatives

# current
plt.figure()
plt.plot(t_full, i, label='current')
plt.legend()

# voltage
plt.figure()
plt.plot(t_full, motor.voltage(t_full), label='v')
plt.plot(t_full, motor.R * i, label='v_resist')
plt.plot(t_full , motor.kt * w, label='back-emf')
plt.plot(t_full, motor.L * di, label='v_imp')
plt.legend()

plt.figure()
plt.plot(t_full, w, label='speed')
plt.legend()

# torque
plt.figure()
plt.plot(t_full, system.pump.torque(w, qp), label='tau_pump')
plt.plot(t_full, motor.mu * w, label='tau_resist')
plt.plot(t_full, motor.kt * i, label='tau_elec')
plt.plot(t_full, motor.M * dw, label='tau_imp')
plt.legend()


plt.figure()
hp = system.pump.hq(w, qp)
hr = circuit.R * qp
Rhv = np.asarray([circuit.hvalve.Ropen if i == 1 else circuit.hvalve.Rclosed for i in hvalve_state])
ha = (h1 / Rhv + haL / circuit.RinL + haR / circuit.RinR) / (
            1 / Rhv + 1 / circuit.RinR + 1 / circuit.RinL)
hx = (haL / circuit.RoutL + haR / circuit.RoutR - qp) / (
            1 / circuit.RoutL + 1 / circuit.RoutR)

hpf = ha - hr
hpa = ha - hr + hp
hvL = tahL.pressure(haL, vvL)
hvR = tahR.pressure(haR, vvR)

plt.plot(t_full, h1, 'k-', label="capacitance pressure")
plt.plot(t_full, hpf, 'g-', label="pressure before pump")
plt.plot(t_full, hpa, 'r-', label="pressure after pump")
plt.plot(t_full, ha, 'b-', label="pouch pressure")
plt.plot(t_full, haL, 'm-', label="pouch L pressure")
plt.plot(t_full, haR, 'c-', label="pouch R pressure")
plt.plot(t_full, hx, 'y-', label="pressure before resistor")
plt.plot(t_full, hvL, 'm--', label="left ventricular pressure")
plt.plot(t_full, hvR, 'c--', label="right ventricular pressure")

plt.legend()

plt.figure()
impedance_head = hx - hr + hp - h1
plt.plot(t_full, hp, 'k-', label="pump pressure head")
plt.plot(t_full, impedance_head, 'b-', label="head pressure impedance")
plt.plot(t_full, hr, 'y-', label="pressure loss resistance")
plt.plot(t_full, h1-ha, 'ro-', label='pressure drop hysteretic valve')

plt.plot(t_full, hvalve_state, 'k--', label="hysteretic valve state")
plt.axhline(oscillator.dhopenref, linestyle='--', color='black')
plt.axhline(oscillator.dhcloseref, linestyle='--', color='black')
plt.legend()


plt.figure()
plt.plot(t_full, hp1, 'k-', label='hp1')
plt.plot(t_full, hp2, 'r-', label='hp2')
plt.plot(t_full, hvL, 'g-', label='hvL')
plt.plot(t_full, hs1, 'b-', label='hs1')
plt.plot(t_full, hs2, 'c-', label='hs2')
plt.plot(t_full, hvR, 'y-', label='hvR')

plt.legend()

plt.figure()
plt.axhline(oscillator.dhopenref, linestyle='--', color='black')
plt.axhline(oscillator.dhcloseref, linestyle='--', color='black')
plt.axhline(0.1, linestyle='--', color='black')
plt.axhline(0.0, linestyle='--', color='black')

plt.plot(t_full, h1-ha, 'ko-', label='pressure drop hysteretic valve')
plt.plot(t_full, hvR - hp1, 'bo-', label='pressure drop pcin valve')
plt.plot(t_full, hp2 - hvL, 'bo--', label='pressure drop pcout valve')
plt.plot(t_full, hvL - hs1, 'ro-', label='pressure drop scin valve')
plt.plot(t_full, hs2 - hvR, 'ro--', label='pressure drop scout valve')



plt.plot(t_full, hvalve_state, 'k-', label="hysteretic valve state")
plt.plot(t_full, valve_pcin_state, 'b-', label="pc in valve state")
plt.plot(t_full, valve_pcout_state, 'b--', label="pc out valve state")
plt.plot(t_full, valve_scin_state, 'r-', label="sc in valve state")
plt.plot(t_full, valve_scout_state, 'r--', label="sc out valve state")

plt.legend()

# flows
plt.figure()
qhv = (h1 - ha) / Rhv
qc1 = qp - qhv

qinL = (ha - haL) / circuit.RinL
qinR = (ha - haR) / circuit.RinR

qoutL = (haL - hx) / circuit.RoutL
qoutR = (haR - hx) / circuit.RoutR

qaL = -1.0 * dvvL
qaR = -1.0 * dvvR

qcL = qinL - qoutL - qaL
qcR = qinR - qoutR - qaR

plt.plot(t_full, qp, 'k-', label="pump flow")
plt.plot(t_full, qc1, 'r-', label="flow capacitor")
plt.plot(t_full, qhv, 'g-', label="HV flow")
plt.plot(t_full, qinL, 'c-', label="left in")
plt.plot(t_full, qoutL, 'c--', label="left out")
plt.plot(t_full, qcL, 'y-', label="left capacitor")
plt.plot(t_full, qaL, 'y--', label="left pouch")
plt.plot(t_full, qinR, 'b-', label="right in")
plt.plot(t_full, qoutR, 'b--', label="right out")
plt.plot(t_full, qcR, 'm-', label="right capacitor")
plt.plot(t_full, qaR, 'm--', label="right pouch")
plt.legend()


plt.figure()
plt.axhline(tahL.Vv0, linestyle='--', color='black', label="LVv0")
plt.plot(t_full, vvL, 'k-', label="left ventricular volume")
plt.axhline(tahR.Vv0, linestyle='--', color='blue', label="RVv0")
plt.plot(t_full, vvR, 'b-', label="right ventricular volume")
plt.plot(t_full, tahL.Vv0 - vvL, label="L-DV")
plt.plot(t_full, tahR.Vv0 - vvR, label="R-DV")
plt.plot(t_full, circuit.C * h1, 'm-', label="capacitance volume")
plt.plot(t_full, circuit.CL * haL, 'k--', label="left capacitor volume")
plt.plot(t_full, circuit.CR * haR, 'b--', label="right capacitor volume")
plt.plot(t_full, hemo.pc.C1 * hp1, 'y-', label="volume pulmonary 1")
plt.plot(t_full, hemo.pc.C2 * hp2, 'c-', label="volume pulmonary 2")
plt.plot(t_full, hemo.sc.C1 * hs1, 'r-', label="volume systemic 1")
plt.plot(t_full, hemo.sc.C2 * hs2, 'g-', label="volume systemic 2")

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




