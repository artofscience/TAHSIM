from math import pi
import numpy as np
from tahs import LinearMembrane
from utils import Sigmoid
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from motors import DCM
from pumps import CP
from utils import event
from copy import deepcopy

"""
Setup of Luuk van Laake, closed loop with linear membrane connected to closed loop three-compartment circulation
"""

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
    def __init__(self, hvalve: Valve = Valve(), C1: float = 1.0, C2: float = 0.01,
                 R: float = 100, L: float = 0.01):
        self.L = L
        self.C1 = C1
        self.C2 = C2
        self.R = R
        self.hvalve = hvalve

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

class System:
    def __init__(self, motor: DCM = DCM(),
                 pump: CP = CP(),
                 circuit: Circuit = Circuit(),
                 tah: LinearMembrane = LinearMembrane(),
                 hemo: TCM = TCM()):
        self.motor = motor
        self.pump = pump
        self.circuit = circuit
        self.tah = tah
        self.hemo = hemo
        self.ki: float = 5.0
        self.kp: float = 0.0
        self.reference_co = lambda t: Sigmoid(0.2, 1.0, 10)(t) - Sigmoid(0.1, 10.0, 10)(t)


    def solve(self, t, y):
        current, speed, pump_capacity, h1, ha, vv, hart, hb, int_error = y

        # tau(w, q)
        torque = self.pump.torque(speed, pump_capacity)

        # dw(i, w, tau(w, q))
        dw = (self.motor.kt * current - self.motor.mu * speed - torque) / self.motor.M

        # h(q, w)
        pump_head = self.pump.hq(speed, pump_capacity)

        # dq(h(q,w), h1)
        impedance_head = pump_head - h1 + ha - self.circuit.R * pump_capacity
        dq_pump = impedance_head / self.circuit.L

        # dh1(q, h1, ha)
        hv_head = h1 - ha
        qhv = hv_head / self.circuit.hvalve()
        qc1 = pump_capacity - qhv
        dh1 = qc1 / self.circuit.C1

        hv = self.tah.pressure(ha, vv)

        # qav
        qav = (hb - hv) / self.hemo.valve_in()

        # qart
        qart = (hv - hart) / self.hemo.valve_out()

        # dvv
        dvv = qav - qart
        qa = -1.0 * dvv # qa = dva/dt = -dvv/dt

        # dh2(h1, ha, qa(dvv))
        qc2 = qhv - pump_capacity - qa
        dha = qc2 / self.circuit.C2

        qp = (hart - hb) / self.hemo.R
        dhart = (qart - qp) / self.hemo.C1
        dhb = (qp - qav) / self.hemo.C2

        #####

        error = self.reference_co(t) - qp

        # voltage based on error
        voltage = self.ki * int_error + self.kp * error

        # di(i, w, v(t))
        di = (voltage - self.motor.R * current - self.motor.kt * speed) / self.motor.L

        return [di, dw, dq_pump, dh1, dha, dvv, dhart, dhb, error]

    @event(direction=1)
    def event_valve_opening(self, t, y):
        current, speed, pump_capacity, h1, ha, vv, hart, hb, _ = y
        return self.circuit.hvalve.event_open(h1 - ha)

    @event(direction=-1)
    def event_valve_closing(self, t, y):
        current, speed, pump_capacity, h1, ha, vv, hart, hb, _ = y
        return self.circuit.hvalve.event_close(h1 - ha)

    @event(direction=1)
    def event_valve_in_opening(self, t, y):
        current, speed, pump_capacity, h1, ha, vv, hart, hb, _ = y
        return self.hemo.valve_in.event_open(hb - self.tah.pressure(ha, vv))

    @event(direction=-1)
    def event_valve_in_closing(self, t, y):
        current, speed, pump_capacity, h1, ha, vv, hart, hb, _ = y
        return self.hemo.valve_in.event_close(hb - self.tah.pressure(ha, vv))

    @event(direction=1)
    def event_valve_out_opening(self, t, y):
        current, speed, pump_capacity, h1, ha, vv, hart, hb, _ = y
        return self.hemo.valve_out.event_open(self.tah.pressure(ha, vv) - hart)

    @event(direction=-1)
    def event_valve_out_closing(self, t, y):
        current, speed, pump_capacity, h1, ha, vv, hart, hb, _ = y
        return self.hemo.valve_out.event_close(self.tah.pressure(ha, vv) - hart)

voltage = lambda t: Sigmoid(2, 0.1)(t)
motor = DCM(voltage, R=0.2, L=0.01, M=3.88/1e7, kt=5.9/1000, mu=12/1e7)
pump = CP(hm0=2.4, qn0=1.6, hn0=1.8, w0=1770 * (2* pi / 60), effn=0.35)
oscillator = Valve(Ropen=1, Rclosed=1000, dhopen=4, dhclose=1)
circuit = Circuit(oscillator, C1=1.0, C2=0.01, R=10, L=0.01)
tah = LinearMembrane(E=1, Vv0=-1)
heart_valve = Valve(Ropen=1, Rclosed=1e4, dhopen=0.1, dhclose=0.0)
hemo = TCM(heart_valve, deepcopy(heart_valve), C1=0.1, C2=0.5, R=5)
system = System(motor=motor, pump=pump, circuit=circuit, tah=tah, hemo=hemo)

events = [system.event_valve_opening, system.event_valve_closing,
          system.event_valve_in_opening, system.event_valve_in_closing,
          system.event_valve_out_opening, system.event_valve_out_closing]

# current, speed, pump_capacity, circuit_head, ha, ven_volume, hart, hb
initial_state = (0.0, 1e-6, 1e-6, 0.0, 0.0, tah.Vv0, 0.0, 0.0, 0.0) # note preloaded system

t_start = 0.0
t_end = 20

t_full = []
y_full = []
hvalve_state = []
valve_in_state = []
valve_out_state = []
derivatives = []

event_times = []

while t_start < t_end:
    sol = solve_ivp(system.solve, [t_start, t_end], initial_state, events=events, rtol=1e-9, atol=1e-9)
    t_full.append(sol.t)
    y_full.append(sol.y)
    derivatives.append(system.solve(sol.t, sol.y))
    hvalve_state.append(system.circuit.hvalve.state * np.ones_like(sol.t))
    valve_in_state.append(system.hemo.valve_in.state * np.ones_like(sol.t))
    valve_out_state.append(system.hemo.valve_out.state * np.ones_like(sol.t))

    if any([i.size > 0 for i in sol.t_events]):

        event = next(i for i, j in enumerate(sol.t_events) if len(j))
        if event == 0:
            system.circuit.hvalve.open()
        elif event == 1:
            system.circuit.hvalve.close()
        elif event == 2:
            system.hemo.valve_in.open()
        elif event == 3:
            system.hemo.valve_in.close()
        elif event == 4:
            system.hemo.valve_out.open()
        elif event == 5:
            system.hemo.valve_out.close()
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

# length = 8
# offset = 5
# m = np.s_[-8-offset:-offset-1]
# m= np.s_[0:-1]
t_full = np.concatenate(t_full)
y_full = np.concatenate(y_full, axis=1)
hvalve_state = np.concatenate(hvalve_state)
valve_in_state = np.concatenate(valve_in_state)
valve_out_state = np.concatenate(valve_out_state)
# event_times = event_times[m]
derivatives = np.concatenate(derivatives, axis=1)

i, w, qp, h1, ha, vv, hart, hb, int_error = y_full
di, dw, dqp, dh1, dha, dvv, dhart, dhb, error = derivatives

# current
plt.figure()
plt.plot(t_full, i, label='current')
[plt.axvline(i, color='black', linestyle='--') for i in event_times]
plt.legend()

# current derivative
plt.figure()
plt.plot(t_full, di, label='di')
[plt.axvline(i, color='black', linestyle='--') for i in event_times]
plt.legend()

# voltage
plt.figure()
voltage = system.ki * int_error
plt.plot(t_full, voltage, label='v')
plt.plot(t_full, motor.R * i, label='v_resist')
plt.plot(t_full , motor.kt * w, label='back-emf')
plt.plot(t_full, motor.L * di, label='v_imp')
[plt.axvline(i, color='black', linestyle='--') for i in event_times]
plt.legend()

plt.figure()
plt.plot(t_full, w, label='speed')
[plt.axvline(i, color='black', linestyle='--') for i in event_times]
plt.legend()

# torque
plt.figure()
# plt.plot(t_full, system.reference_co(t_full), label='reference_torque')
plt.plot(t_full, system.pump.torque(w, qp), label='tau_pump')
plt.plot(t_full, motor.mu * w, label='tau_resist')
plt.plot(t_full, motor.kt * i, label='tau_elec')
plt.plot(t_full, motor.M * dw, label='tau_imp')
plt.legend()


# pressure
plt.figure()
hv = system.tah.pressure(ha, vv)
hp = system.pump.hq(w, qp)
hr = circuit.R * qp
impedance_head = hp - h1 + ha - hr

plt.plot(t_full, hp, 'ro-', label="pump pressure head")
plt.plot(t_full, impedance_head, 'yo-', label="pump impedance head")
plt.plot(t_full, hr, 'co-', label="resistance pressure loss")
plt.plot(t_full, h1-ha, 'bo-', label='pressure drop hysteretic valve')
plt.plot(t_full, hvalve_state, 'm-', label="hysteretic valve state")
plt.axhline(oscillator.dhopenref, linestyle='--', color='black')
plt.axhline(oscillator.dhcloseref, linestyle='--', color='black')
plt.legend()

plt.figure()
hpf = ha - hr
hpa = ha - hr + hp
plt.plot(t_full, h1, 'ko-', label="capacitance pressure")
plt.plot(t_full, hpf, 'go-', label="pressure before pump")
plt.plot(t_full, hpa, 'ro-', label="pressure after pump")
plt.plot(t_full, ha, 'bo-', label="pouch pressure")
plt.plot(t_full, hv, 'mo-', label="ventricular pressure")
plt.legend()

plt.figure()
plt.plot(t_full, hart, 'ro-', label="pressure afterload")
plt.plot(t_full, hb, 'go-', label="pressure preload")
plt.plot(t_full, hv, 'bo-', label="ventricular pressure")
plt.legend()

plt.figure()
plt.plot(t_full, valve_in_state, 'b-', label="input valve state")
plt.plot(t_full, valve_out_state, 'k-', label="output valve state")

plt.plot(t_full, hv-hart, 'ko-', label='pressure drop output valve')
plt.plot(t_full, hb-hv, 'bo-', label='pressure drop input valve')
plt.plot(t_full, hart - hb, 'ro-', label='pressure drop resistance')

plt.axhline(0.1, linestyle='--', color='black')
plt.axhline(0.0, linestyle='--', color='black')
plt.legend()

plt.figure()
qa = -dvv
plt.plot(t_full, qp, 'ko-', label='qp')
qhv = (h1 - ha) / [system.circuit.hvalve.Rclosed if i == 0 else system.circuit.hvalve.Ropen for i in hvalve_state]
plt.plot(t_full, qhv, 'bo-', label="qhv")
plt.plot(t_full, qp - qhv, 'yo-', label="qc1")
plt.plot(t_full, qa, 'ro-', label="qa")
plt.plot(t_full, qhv - qp -qa, 'mo-', label= "qc2")
co = (hart - hb) / hemo.R
plt.plot(t_full, co, 'k--', label="co")
plt.plot(t_full, system.reference_co(t_full), 'r--', label="reference co")
[plt.axvline(i, color='black', linestyle='--') for i in event_times]
plt.legend()

plt.figure()
va = -1.0 * vv
plt.plot(t_full, system.circuit.C1 * h1, label='vc1')
plt.plot(t_full, system.circuit.C2 * ha, label='vc2')
plt.plot(t_full, va, label='va')
plt.plot(t_full, vv, label='vv')
plt.plot(t_full, system.hemo.C1 * hart, label='vc_art')
plt.plot(t_full, system.hemo.C2 * hb, label='vc_b')
plt.ylabel('Volume (L)')
plt.legend()

plt.figure()
plt.plot(vv, hv, label="ventricle PV")

plt.figure()
plt.plot(t_full, error, label="flow error")
plt.plot(t_full, int_error, label="int_error (volume)")
plt.legend()

plt.show()





