from utils import event
from scipy.signal import square
from math import pi
from scipy.integrate import solve_ivp

import numpy as np
from tahs import TAH

data0 = {"relative duration systole": 1/3, # relative duration of systole vs full cycle
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
        "systemic venous capacitance": 40.0,  # systemic venous capacitance in mL / mmHg
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


def find_zero_to_one(data):
    # Start the list with a zero
    result = [0]

    # Check if the value changed from 0 to 1
    for i in range(1, len(data)):
        if data[i - 1] == 0 and data[i] == 1:
            result.append(1)
        else:
            result.append(0)

    return result


def find_one_to_zero(data):
    # Start de lijst met een nul
    result = [0]

    # Controleer of de waarde is veranderd van 1 naar 0
    for i in range(1, len(data)):
        if data[i - 1] == 1 and data[i] == 0:
            result.append(1)
        else:
            result.append(0)

    return result


def ejected_volume(volume, valve_trigger):
    ejected = []
    edv_reference = volume[0]
    for vol, opened in zip(volume, valve_trigger):
        if opened:
            edv_reference = vol
        current_ejection = edv_reference - vol
        ejected.append(current_ejection)
    return ejected

class PhysiologicValues:
    r_valve_aortic = 0.005 # mmHg * s / mL
    r_valve_pulmonary = 0.003 # mmHg * s / mL
    r_valve_mitral = 0.01 # mmHg * s / mL
    r_valve_tricuspid = 0.008 # mmHg * s / mL
    r_systemic_circulation = 1.0 # mmHg * s / mL
    r_pulmonary_circulation = 0.12 # mmHg * s / mL
    c_systemic_arterial = 1.5 # mL / mmHg
    c_systemic_venous = 100.0 # mL / mmHg
    c_pulmonary_arterial = 3.5 # mL / mmHg
    c_pulmonary_venous = 12.0 # mL / mmHg
    e_left_ventricle = 1.5 # mmHg / mL
    e_right_ventricle = 1.5 # mmHg / mL
    v_total = 5000 # mL
    v_stroke = 100 # mL
    v_left_dead = 15 # mL
    v_right_dead = 15 # mL
    v_systemic_venous_dead = 2500 # mL
    v_pulmonary_venous_dead = 150 # mL
    v_systemic_arterial_dead = 100 # mL
    v_pulmonary_arterial_dead = 50 # mL
    p_left_ventricle_max = 375 # mmHg 120?
    p_right_ventricle_max = 225 # mmHg 25?
    p_mean_arterial = 100 # mmHg
    t_act = 1.0 # s

class FourCompartment:
    def __init__(self, sl, sr,
                 ra, rt, rp, rm,
                 el: float = 1.0, er: float = 1.0,
                 cs: float = 1.0, cp: float = 1.0,
                 vl0: float = 1.0, vr0: float = 1.0,
                 vs0: float = 1.0, vp0: float = 1.0):
        self.sl = sl
        self.sr = sr
        self.ra = ra
        self.rt = rt
        self.rp = rp
        self.rm = rm
        self.el = el
        self.er = er
        self.cs = cs
        self.cp = cp
        self.vl0 = vl0
        self.vr0 = vr0
        self.vs0 = vs0
        self.vp0 = vp0

    def solve(self, t, y):
        """
        Solves the nondimensionalized system of equations:

        left ventricle:
        Pl = Sl + El(Vl - Vl0)
        Ql = dVl/dT = Qm - Qa (mitral flow - aortic flow)
        Qa = (Pl - Ps) / Ra (resistance of aortic valve)
        Qm = (Pp - Pl) / Rm (resistance of pulmonary valve AND pulmonary vascular resistance)

        right ventricle:
        Pr = Sr + Er(Vr - Vr0)
        Qr = dVr/dT = Qt - Qp (tricuspid flow - pulmonary flow)
        Qp = (Pr - Pp) / Rp (resistance of pulmonary valve)
        Qt = (Ps - Pr) / Rt (resistance of tricuspid valve AND systemic vascular resistance)

        systemic circulation:
        Ps = Vs / Cs
        dVs/dT = Qa - Qt

        pulmonary circulation:
        Pp = Vp / Cp
        dVp/dT = Qp - Qm
        """

        vl, vs, vr, vp = y

        # ventricular pressures
        pl = self.sl(t) + self.el * (vl - self.vl0)
        ps = (vs - self.vs0) / self.cs
        pr = self.sr(t) + self.er * (vr - self.vr0)
        pp = (vp - self.vp0) / self.cp

        # flows through valves
        qav = (pl - ps) / self.ra()
        qtv = (ps - pr) / self.rt()
        qpv = (pr - pp) / self.rp()
        qmv = (pp - pl) / self.rm()

        # flows in chambers
        ql = qmv - qav
        qs = qav - qtv
        qr = qtv - qpv
        qp = qpv - qmv

        return [ql, qs, qr, qp]

    @event(direction=1)
    def aortic_valve_open(self, t, y):
        vl, vs, _, _ = y
        pl = self.sl(t) + self.el * (vl - self.vl0)
        ps = (vs - self.vs0) / self.cs
        return self.ra.event_open(pl - ps)

    @event(direction=-1)
    def aortic_valve_close(self, t, y):
        vl, vs, _, _ = y
        pl = self.sl(t) + self.el * (vl - self.vl0)
        ps = (vs - self.vs0) / self.cs
        return self.ra.event_close(pl - ps)

    @event(direction=1)
    def tricuspid_valve_open(self, t, y):
        _, vs, vr, _ = y
        ps = (vs - self.vs0) / self.cs
        pr = self.sr(t) + self.er * (vr - self.vr0)
        return self.rt.event_open(ps - pr)

    @event(direction=-1)
    def tricuspid_valve_close(self, t, y):
        _, vs, vr, _ = y
        ps = (vs - self.vs0) / self.cs
        pr = self.sr(t) + self.er * (vr - self.vr0)
        return self.rt.event_close(ps - pr)

    @event(direction=1)
    def pulmonary_valve_open(self, t, y):
        _, _, vr, vp = y
        pr = self.sr(t) + self.er * (vr - self.vr0)
        pp = (vp - self.vp0) / self.cp
        return self.rp.event_open(pr - pp)

    @event(direction=-1)
    def pulmonary_valve_close(self, t, y):
        _, _, vr, vp = y
        pr = self.sr(t) + self.er * (vr - self.vr0)
        pp = (vp - self.vp0) / self.cp
        return self.rp.event_close(pr - pp)

    @event(direction=1)
    def mitral_valve_open(self, t, y):
        vl, _, _, vp = y
        pp = (vp - self.vp0) / self.cp
        pl = self.sl(t) + self.el * (vl - self.vl0)
        return self.rm.event_open(pp - pl)

    @event(direction=-1)
    def mitral_valve_close(self, t, y):
        vl, _, _, vp = y
        pp = (vp - self.vp0) / self.cp
        pl = self.sl(t) + self.el * (vl - self.vl0)
        return self.rm.event_close(pp - pl)

class Valve:
    def __init__(self, Ropen:float=0.1, Rclosed:float=100, dhopen:float=0.0, dhclose:float=0.0, initial_state: int=0):
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
        self.magnitude0 = 1.0 * magnitude
        self.magnitude = magnitude
        self.freq = freq
        self.phase = phase
        self.duty = duty

    def __call__(self, t):
        return self.magnitude * (0.5 * square(2 * pi * self.freq * t - self.phase, duty=self.duty) + 0.5)

class Actuator:
    def __init__(self, source, capacitance, unstressed_volume, resistance):
        self.source = source
        self.capacitance = capacitance
        self.unstressed_volume = unstressed_volume
        self.resistance = resistance

class Actuation:
    def __init__(self, left: Actuator, right: Actuator):
        self.left = left
        self.right = right

class Circulation:
    def __init__(self, valve_in: Valve = Valve(),
                 valve_out: Valve = Valve(),
                 C1: float = 0.1,
                 V10: float = 1.0,
                 C2: float = 0.1,
                 V20: float = 1.0,
                 R: float = 10):
        self.C1 = C1 # L / m
        self.V10 = V10
        self.C2 = C2
        self.V20 = V20
        self.R = R # m / L/min
        self.valve_in = valve_in
        self.valve_out = valve_out

class Circulations:
    def __init__(self, systemic: Circulation, pulmonary: Circulation):
        self.systemic = systemic
        self.pulmonary = pulmonary

class Heart:
    def __init__(self, left: TAH, right: TAH):
        self.left = left
        self.right = right

class BiVenSystem:
    def __init__(self, actuation: Actuation,
                heart: Heart,
                 circulations: Circulations):
        self.actuation = actuation
        self.heart = heart
        self.circulations = circulations

    def actuation_pressure(self, vaL, vaR):
        haL = (vaL - self.actuation.left.unstressed_volume) / self.actuation.left.capacitance
        haR = (vaR - self.actuation.right.unstressed_volume) / self.actuation.right.capacitance
        return haL, haR

    def ventricular_pressure(self, vvL, vvR, haL, haR):
        hvL = self.heart.left.pressure(haL, vvL)
        hvR = self.heart.right.pressure(haR, vvR)
        return hvL, hvR

    def circulation_pressure(self, vs1, vs2, vp1, vp2):
        hs1 = (vs1 - self.circulations.systemic.V10) / self.circulations.systemic.C1
        hs2 = (vs2 - self.circulations.systemic.V20) / self.circulations.systemic.C2

        hp1 = (vp1 - self.circulations.pulmonary.V10) / self.circulations.pulmonary.C1
        hp2 = (vp2 - self.circulations.pulmonary.V20) / self.circulations.pulmonary.C2
        return hs1, hs2, hp1, hp2

    def circulation_pressure_collapsable(self, vs1, vs2, vp1, vp2):
        hs1, hs2, hp1, hp2 = self.circulation_pressure(vs1, vs2, vp1, vp2)
        return np.maximum(0.0, hs1), np.maximum(0.0, hs2), np.maximum(0.0, hp1), np.maximum(0.0, hp2)

    def pressures(self, volumes):
        vaL, vaR, vvL, vvR, vs1, vs2, vp1, vp2 = volumes

        haL, haR = self.actuation_pressure(vaL, vaR)
        hvL, hvR = self.ventricular_pressure(vvL, vvR, haL, haR)
        hs1, hs2, hp1, hp2 = self.circulation_pressure_collapsable(vs1, vs2, vp1, vp2)

        return [haL, haR, hvL, hvR, hs1, hs2, hp1, hp2]

    def source_flow(self, t, haL, haR):
        qsL = (self.actuation.left.source(t) - haL) / self.actuation.left.resistance
        qsR = (self.actuation.right.source(t) - haR) / self.actuation.right.resistance
        return qsL, qsR

    def flows(self, t, pressures):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = pressures

        # flow through systemic and pulmonary valves and resistors
        qsi = (hvL - hs1) / self.circulations.systemic.valve_in()
        qs = (hs1 - hs2) / self.circulations.systemic.R(t)
        qso = (hs2 - hvR) / self.circulations.systemic.valve_out()
        qpi = (hvR - hp1) / self.circulations.pulmonary.valve_in()
        qp = (hp1 - hp2) / self.circulations.pulmonary.R
        qpo = (hp2 - hvL) / self.circulations.pulmonary.valve_out()

        # flow into systemic and pulmonary capacitors and heart
        qs1 = qsi - qs
        qs2 = qs - qso
        qvR = qso - qpi
        qp1 = qpi - qp
        qp2 = qp - qpo
        qvL = qpo - qsi

        # flow from/to pressure source and source capacitor
        qsL, qsR = self.source_flow(t, haL, haR)

        qaL = qsL + qvL
        qaR = qsR + qvR

        return [qaL, qaR, qvL, qvR, qs1, qs2, qp1, qp2]

    def solve(self, t, volumes):
        pressures = self.pressures(volumes)
        return self.flows(t, pressures)

    def dynamics(self, initial_state, t_end, t_start: float = 0.0, rtol: float = 1e-9, atol: float = 1e-9):
        t_full = []
        y_full = []
        valve_scin_state = []
        valve_scout_state = []
        valve_pcin_state = []
        valve_pcout_state = []
        derivatives = []
        event_times = []
        event_values = []

        events = [self.event_valve_systemic_in_opening, self.event_valve_systemic_in_closing,
                  self.event_valve_systemic_out_opening, self.event_valve_systemic_out_closing,
                  self.event_valve_pulmonary_in_opening, self.event_valve_pulmonary_in_closing,
                  self.event_valve_pulmonary_out_opening, self.event_valve_pulmonary_out_closing
                  ]

        while t_start < t_end:
            sol = solve_ivp(self.solve, [t_start, t_end], initial_state, events=events, rtol=rtol, atol=atol)
            derivatives.append(self.solve(sol.t, sol.y))

            t_full.append(sol.t)
            y_full.append(sol.y)

            valve_scin_state.append(self.circulations.systemic.valve_in.state * np.ones_like(sol.t))
            valve_scout_state.append(self.circulations.systemic.valve_out.state * np.ones_like(sol.t))
            valve_pcin_state.append(self.circulations.pulmonary.valve_in.state * np.ones_like(sol.t))
            valve_pcout_state.append(self.circulations.pulmonary.valve_out.state * np.ones_like(sol.t))

            if any([i.size > 0 for i in sol.t_events]):

                event = next(i for i, j in enumerate(sol.t_events) if len(j))
                if event == 0:
                    self.circulations.systemic.valve_in.open()
                elif event == 1:
                    self.circulations.systemic.valve_in.close()
                elif event == 2:
                    self.circulations.systemic.valve_out.open()
                elif event == 3:
                    self.circulations.systemic.valve_out.close()
                elif event == 4:
                    self.circulations.pulmonary.valve_in.open()
                elif event == 5:
                    self.circulations.pulmonary.valve_in.close()
                elif event == 6:
                    self.circulations.pulmonary.valve_out.open()
                elif event == 7:
                    self.circulations.pulmonary.valve_out.close()
                else:
                    print("no valid event")

                event_time = sol.t_events[event][0]
                event_times.append(event_time)
                event_values.append(event)

                print(event_time)
                print(event)
                t_start = event_time
                initial_state = sol.y_events[event][0]
            else:
                t_start = t_end

        times = np.concatenate(t_full)
        volumes = np.concatenate(y_full, axis=1)
        pressures = self.pressures(volumes)
        flows = np.concatenate(derivatives, axis=1)

        valve_scin_state = np.concatenate(valve_scin_state)
        valve_scout_state = np.concatenate(valve_scout_state)
        valve_pcin_state = np.concatenate(valve_pcin_state)
        valve_pcout_state = np.concatenate(valve_pcout_state)
        valves = [event_times, event_values, valve_scin_state, valve_scout_state, valve_pcin_state, valve_pcout_state]

        return [times, volumes, pressures, flows, valves]

    @event(direction=1)
    def event_valve_systemic_in_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.systemic.valve_in.event_open(hvL - hs1)

    @event(direction=-1)
    def event_valve_systemic_in_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.systemic.valve_in.event_close(hvL - hs1)

    @event(direction=1)
    def event_valve_systemic_out_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.systemic.valve_out.event_open(hs2 - hvR)

    @event(direction=-1)
    def event_valve_systemic_out_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.systemic.valve_out.event_close(hs2 - hvR)

    @event(direction=1)
    def event_valve_pulmonary_in_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_in.event_open(hvR - hp1)

    @event(direction=-1)
    def event_valve_pulmonary_in_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_in.event_close(hvR - hp1)

    @event(direction=1)
    def event_valve_pulmonary_out_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_out.event_open(hp2 - hvL)

    @event(direction=-1)
    def event_valve_pulmonary_out_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_out.event_close(hp2 - hvL)

class BiVenSystemSimple(BiVenSystem):
    def pressures(self, volumes):
        vaL, vaR, vvL, vvR, vs1, vp1 = volumes

        haL = (vaL - self.actuation.left.unstressed_volume) / self.actuation.left.capacitance
        haR = (vaR - self.actuation.right.unstressed_volume) / self.actuation.right.capacitance

        hvL = self.heart.left.pressure(haL, vvL)
        hvR = self.heart.right.pressure(haR, vvR)

        hs1 = (vs1 - self.circulations.systemic.V10) / self.circulations.systemic.C1
        hp1 = (vp1 - self.circulations.pulmonary.V10) / self.circulations.pulmonary.C1

        return [haL, haR, hvL, hvR, hs1, hp1]

    def flows(self, t, pressures):
        haL, haR, hvL, hvR, hs1, hp1 = pressures

        # flow through systemic and pulmonary valves and resistors
        qsi = (hvL - hs1) / self.circulations.systemic.valve_in()
        qso = (hs1 - hvR) / self.circulations.systemic.valve_out()
        qpi = (hvR - hp1) / self.circulations.pulmonary.valve_in()
        qpo = (hp1 - hvL) / self.circulations.pulmonary.valve_out()

        # flow into systemic and pulmonary capacitors and heart
        qs1 = qsi - qso
        qvR = qso - qpi
        qp1 = qpi - qpo
        qvL = qpo - qsi

        # flow from/to pressure source and source capacitor
        qsL, qsR = self.source_flow(t, haL, haR)

        qaL = qsL + qvL
        qaR = qsR + qvR

        return [qaL, qaR, qvL, qvR, qs1, qp1]

    @event(direction=1)
    def event_valve_systemic_in_opening(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.systemic.valve_in.event_open(self.heart.left.pressure(haL, vvL) - hs1)

    @event(direction=-1)
    def event_valve_systemic_in_closing(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.systemic.valve_in.event_close(self.heart.left.pressure(haL, vvL) - hs1)

    @event(direction=1)
    def event_valve_systemic_out_opening(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.systemic.valve_out.event_open(hs1 - self.heart.right.pressure(haR, vvR))

    @event(direction=-1)
    def event_valve_systemic_out_closing(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.systemic.valve_out.event_close(hs1 - self.heart.right.pressure(haR, vvR))

    @event(direction=1)
    def event_valve_pulmonary_in_opening(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_in.event_open(self.heart.right.pressure(haR, vvR) - hp1)

    @event(direction=-1)
    def event_valve_pulmonary_in_closing(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_in.event_close(self.heart.right.pressure(haR, vvR) - hp1)

    @event(direction=1)
    def event_valve_pulmonary_out_opening(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_out.event_open(hp1 - self.heart.left.pressure(haL, vvL))

    @event(direction=-1)
    def event_valve_pulmonary_out_closing(self, t, volumes):
        vvL, vvR = volumes[2:4]
        haL, haR, hvL, hvR, hs1, hp1 = self.pressures(volumes)
        return self.circulations.pulmonary.valve_out.event_close(hp1 - self.heart.left.pressure(haL, vvL))

class BiVenSeptum(BiVenSystem):

    def septum_volume(self, vvL, vvR, haL, haR):
        sumE = self.heart.left.E + self.heart.right.E + self.heart.septum.E

        vs = self.heart.left.E * (vvL - self.heart.left.Vv0) / sumE
        vs -= self.heart.right.E * (vvR - self.heart.right.Vv0) / sumE
        vs += (haL - haR) / sumE

        return vs

    def ventricular_pressure(self, vvL, vvR, haL, haR):
        vs = self.septum_volume(vvL, vvR, haL, haR)
        hvL = self.heart.left.E * (vvL - vs - self.heart.left.Vv0) + haL
        hvR = self.heart.right.E * (vvR + vs - self.heart.right.Vv0) + haR
        return hvL, hvR

class BiVenPeri(BiVenSystem):

    def ventricular_pressure(self, vvL, vvR, haL, haR):

        hperi = self.heart.pericardium.E * (vvL + vvR - self.heart.pericardium.V0)

        hvL = self.heart.left.pressure(haL, vvL) + hperi
        hvR = self.heart.right.pressure(haR, vvR) + hperi
        return hvL, hvR

class BiVenSeptumPeri(BiVenSystem):
    def ventricular_pressure(self, vvL, vvR, haL, haR):
        hperi = self.heart.pericardium.E * (vvL + vvR - self.heart.pericardium.V0)

        vs = self.septum_volume(vvL, vvR, haL, haR)
        hvL = self.heart.left.E * (vvL - vs - self.heart.left.Vv0) + haL + hperi
        hvR = self.heart.right.E * (vvR + vs - self.heart.right.Vv0) + haR + hperi
        return hvL, hvR

class BiVenCustom(BiVenSystem):

    def ventricular_pressure(self, vvL, vvR, haL, haR):
        hvL = 1.0 * (vvL - self.heart.left.Vv0) + 0.1 * (vvR - self.heart.right.Vv0) ** 3 + haL
        hvR = 0.1 * (vvR - self.heart.right.Vv0) ** 3 * (vvL - self.heart.left.Vv0) + 0.05 * (
                    vvR - self.heart.right.Vv0) + haR

        return hvL, hvR


class BiVenControl:
    def __init__(self, actuation: Actuation,
                heart: Heart,
                 circulations: Circulations):
        self.actuation = actuation
        self.heart = heart
        self.circulations = circulations

    def actuation_pressure(self, vaL, vaR):
        haL = (vaL - self.actuation.left.unstressed_volume) / self.actuation.left.capacitance
        haR = (vaR - self.actuation.right.unstressed_volume) / self.actuation.right.capacitance
        return haL, haR

    def ventricular_pressure(self, vvL, vvR, haL, haR):
        hvL = self.heart.left.pressure(haL, vvL)
        hvR = self.heart.right.pressure(haR, vvR)
        return hvL, hvR

    def circulation_pressure(self, vs1, vs2, vp1, vp2):
        hs1 = (vs1 - self.circulations.systemic.V10) / self.circulations.systemic.C1
        hs2 = (vs2 - self.circulations.systemic.V20) / self.circulations.systemic.C2

        hp1 = (vp1 - self.circulations.pulmonary.V10) / self.circulations.pulmonary.C1
        hp2 = (vp2 - self.circulations.pulmonary.V20) / self.circulations.pulmonary.C2
        return hs1, hs2, hp1, hp2

    def circulation_pressure_collapsable(self, vs1, vs2, vp1, vp2):
        hs1, hs2, hp1, hp2 = self.circulation_pressure(vs1, vs2, vp1, vp2)
        return np.maximum(0.0, hs1), np.maximum(0.0, hs2), np.maximum(0.0, hp1), np.maximum(0.0, hp2)

    def pressures(self, volumes):
        vaL, vaR, vvL, vvR, vs1, vs2, vp1, vp2 = volumes

        haL, haR = self.actuation_pressure(vaL, vaR)
        hvL, hvR = self.ventricular_pressure(vvL, vvR, haL, haR)
        hs1, hs2, hp1, hp2 = self.circulation_pressure_collapsable(vs1, vs2, vp1, vp2)

        return [haL, haR, hvL, hvR, hs1, hs2, hp1, hp2]

    def flows(self, t, pressures):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = pressures

        # flow through systemic and pulmonary valves and resistors
        qsi = (hvL - hs1) / self.circulations.systemic.valve_in()
        qs = (hs1 - hs2) / self.circulations.systemic.R(t)
        qso = (hs2 - hvR) / self.circulations.systemic.valve_out()
        qpi = (hvR - hp1) / self.circulations.pulmonary.valve_in()
        qp = (hp1 - hp2) / self.circulations.pulmonary.R
        qpo = (hp2 - hvL) / self.circulations.pulmonary.valve_out()

        # flow into systemic and pulmonary capacitors and heart
        qs1 = qsi - qs
        qs2 = qs - qso
        qvR = qso - qpi
        qp1 = qpi - qp
        qp2 = qp - qpo
        qvL = qpo - qsi

        # flow from/to pressure source and source capacitor
        qsL = (self.actuation.left.source(t) - haL) / self.actuation.left.resistance
        qsR = (self.actuation.right.source(t) - haR) / self.actuation.right.resistance

        qaL = qsL + qvL
        qaR = qsR + qvR

        return [qaL, qaR, qvL, qvR, qs1, qs2, qp1, qp2]

    def solve(self, t, state_variables):
        volumes = state_variables[:-2]
        pressures = self.pressures(volumes)

        hs1 = pressures[4]
        xavg, int_error = state_variables[-2:]

        dxavg = (hs1 - xavg) / 6
        error = 2.8 - xavg

        self.actuation.left.source.magnitude = 3.2 + (30 * error + 1 * int_error)
        self.actuation.right.source.magnitude = 2.0 + (30 * error + 1 * int_error)

        flows = self.flows(t, pressures)
        flows.append(dxavg)
        flows.append(error)

        return flows

    def dynamics(self, initial_state, t_end, t_start: float = 0.0, rtol: float = 1e-9, atol: float = 1e-9):
        t_full = []
        y_full = []
        valve_scin_state = []
        valve_scout_state = []
        valve_pcin_state = []
        valve_pcout_state = []
        # derivatives = []
        event_times = []
        event_values = []
        control_vars = []

        events = [self.event_valve_systemic_in_opening, self.event_valve_systemic_in_closing,
                  self.event_valve_systemic_out_opening, self.event_valve_systemic_out_closing,
                  self.event_valve_pulmonary_in_opening, self.event_valve_pulmonary_in_closing,
                  self.event_valve_pulmonary_out_opening, self.event_valve_pulmonary_out_closing
                  ]

        while t_start < t_end:
            sol = solve_ivp(self.solve, [t_start, t_end], initial_state, events=events, rtol=rtol, atol=atol)
            # derivatives.append(self.solve(sol.t, sol.y))

            t_full.append(sol.t)
            y_full.append(sol.y[:-2])
            control_vars.append(sol.y[-2:])

            valve_scin_state.append(self.circulations.systemic.valve_in.state * np.ones_like(sol.t))
            valve_scout_state.append(self.circulations.systemic.valve_out.state * np.ones_like(sol.t))
            valve_pcin_state.append(self.circulations.pulmonary.valve_in.state * np.ones_like(sol.t))
            valve_pcout_state.append(self.circulations.pulmonary.valve_out.state * np.ones_like(sol.t))

            if any([i.size > 0 for i in sol.t_events]):

                event = next(i for i, j in enumerate(sol.t_events) if len(j))
                if event == 0:
                    self.circulations.systemic.valve_in.open()
                elif event == 1:
                    self.circulations.systemic.valve_in.close()
                elif event == 2:
                    self.circulations.systemic.valve_out.open()
                elif event == 3:
                    self.circulations.systemic.valve_out.close()
                elif event == 4:
                    self.circulations.pulmonary.valve_in.open()
                elif event == 5:
                    self.circulations.pulmonary.valve_in.close()
                elif event == 6:
                    self.circulations.pulmonary.valve_out.open()
                elif event == 7:
                    self.circulations.pulmonary.valve_out.close()
                else:
                    print("no valid event")

                event_time = sol.t_events[event][0]
                event_times.append(event_time)
                event_values.append(event)

                print(event_time)
                print(event)
                t_start = event_time
                initial_state = sol.y_events[event][0]
            else:
                t_start = t_end

        times = np.concatenate(t_full)
        volumes = np.concatenate(y_full, axis=1)
        pressures = self.pressures(volumes)
        control_vars = np.concatenate(control_vars, axis=1)
        # flows = np.concatenate(derivatives, axis=1)

        valve_scin_state = np.concatenate(valve_scin_state)
        valve_scout_state = np.concatenate(valve_scout_state)
        valve_pcin_state = np.concatenate(valve_pcin_state)
        valve_pcout_state = np.concatenate(valve_pcout_state)
        valves = [event_times, event_values, valve_scin_state, valve_scout_state, valve_pcin_state, valve_pcout_state]

        return [times, volumes, pressures, valves, control_vars]

    @event(direction=1)
    def event_valve_systemic_in_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.systemic.valve_in.event_open(hvL - hs1)

    @event(direction=-1)
    def event_valve_systemic_in_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.systemic.valve_in.event_close(hvL - hs1)

    @event(direction=1)
    def event_valve_systemic_out_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.systemic.valve_out.event_open(hs2 - hvR)

    @event(direction=-1)
    def event_valve_systemic_out_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.systemic.valve_out.event_close(hs2 - hvR)

    @event(direction=1)
    def event_valve_pulmonary_in_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.pulmonary.valve_in.event_open(hvR - hp1)

    @event(direction=-1)
    def event_valve_pulmonary_in_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.pulmonary.valve_in.event_close(hvR - hp1)

    @event(direction=1)
    def event_valve_pulmonary_out_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.pulmonary.valve_out.event_open(hp2 - hvL)

    @event(direction=-1)
    def event_valve_pulmonary_out_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(volumes[:-2])
        return self.circulations.pulmonary.valve_out.event_close(hp2 - hvL)

class BiVenNoControl(BiVenControl):
    def solve(self, t, state_variables):
        volumes = state_variables[:-2]
        pressures = self.pressures(volumes)

        hs1 = pressures[4]
        xavg, int_error = state_variables[-2:]

        dxavg = (hs1 - xavg) / 6
        error = 2.8 - xavg

        flows = self.flows(t, pressures)
        flows.append(dxavg)
        flows.append(error)

        return flows

class Controller:
    def __init__(self, var: int = 0, ref_val: float = 1.0, tau: float = 1.0, kp: float = 1.0, ki: float = 1.0, kd: float = 1.0):
        self.var = var
        self.ref_val = ref_val
        self.tau = tau
        self.kp = kp
        self.ki = ki
        self.kd = kd

class BiVen:
    def __init__(self, actuation: Actuation,
                heart: Heart,
                 circulations: Circulations, controller: Controller = None):
        self.actuation = actuation
        self.heart = heart
        self.circulations = circulations
        self.controller = controller

    def actuation_pressure(self, vaL, vaR):
        haL = (vaL - self.actuation.left.unstressed_volume) / self.actuation.left.capacitance
        haR = (vaR - self.actuation.right.unstressed_volume) / self.actuation.right.capacitance
        return haL, haR

    def ventricular_pressure(self, vvL, vvR, haL, haR):
        hvL = self.heart.left.pressure(haL, vvL)
        hvR = self.heart.right.pressure(haR, vvR)
        return hvL, hvR

    def circulation_pressure(self, t, vs1, vs2, vp1, vp2):
        hs1 = (vs1 - self.circulations.systemic.V10) / self.circulations.systemic.C1(t)
        hs2 = (vs2 - self.circulations.systemic.V20) / self.circulations.systemic.C2(t)

        hp1 = (vp1 - self.circulations.pulmonary.V10) / self.circulations.pulmonary.C1(t)
        hp2 = (vp2 - self.circulations.pulmonary.V20) / self.circulations.pulmonary.C2(t)
        return hs1, hs2, hp1, hp2

    def circulation_pressure_collapsable(self, t, vs1, vs2, vp1, vp2):
        hs1, hs2, hp1, hp2 = self.circulation_pressure(t, vs1, vs2, vp1, vp2)
        return np.maximum(0.0, hs1), np.maximum(0.0, hs2), np.maximum(0.0, hp1), np.maximum(0.0, hp2)

    def pressures(self, t, volumes):
        vaL, vaR, vvL, vvR, vs1, vs2, vp1, vp2 = volumes

        haL, haR = self.actuation_pressure(vaL, vaR)
        hvL, hvR = self.ventricular_pressure(vvL, vvR, haL, haR)
        hs1, hs2, hp1, hp2 = self.circulation_pressure_collapsable(t, vs1, vs2, vp1, vp2)

        return [haL, haR, hvL, hvR, hs1, hs2, hp1, hp2]

    def source_flow(self, t, haL, haR):
        # flow from/to pressure source and source capacitor
        qsL = (self.actuation.left.source(t) - haL) / self.actuation.left.resistance
        qsR = (self.actuation.right.source(t) - haR) / self.actuation.right.resistance
        return qsL, qsR

    def flows_intermediate(self, t, pressures):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = pressures

        # flow through systemic and pulmonary valves and resistors
        qav = (hvL - hs1) / self.circulations.systemic.valve_in()
        qs = (hs1 - hs2) / self.circulations.systemic.R(t)
        qtv = (hs2 - hvR) / self.circulations.systemic.valve_out()
        qpv = (hvR - hp1) / self.circulations.pulmonary.valve_in()
        qp = (hp1 - hp2) / self.circulations.pulmonary.R(t)
        qmv = (hp2 - hvL) / self.circulations.pulmonary.valve_out()

        return qav, qs, qtv, qpv, qp, qmv

    def flows_state_vars(self, flows_int):
        qav, qs, qtv, qpv, qp, qmv = flows_int
        # flow into systemic and pulmonary capacitors and heart
        qs1 = qav - qs
        qs2 = qs - qtv
        qvR = qtv - qpv
        qp1 = qpv - qp
        qp2 = qp - qmv
        qvL = qmv - qav

        return [qvL, qvR, qs1, qs2, qp1, qp2]


    def flows(self, t, pressures):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = pressures

        flows_int = self.flows_intermediate(t, pressures)
        qvL, qvR, qs1, qs2, qp1, qp2 = self.flows_state_vars(flows_int)
        qsL, qsR = self.source_flow(t, haL, haR)

        qaL = qsL + qvL
        qaR = qsR + qvR

        return [qaL, qaR, qvL, qvR, qs1, qs2, qp1, qp2]

    def set_actuation_pressures(self, control_vars, control_pressure):
        xavg, int_error = control_vars

        dxavg = (control_pressure - xavg) / self.controller.tau
        error = self.controller.ref_val - xavg

        pleft = self.actuation.left.source.magnitude0 + (self.controller.kp * error + self.controller.ki * int_error)
        pright = self.actuation.right.source.magnitude0 + (self.controller.kp * error + self.controller.ki * int_error)
        return pleft, pright, dxavg, error

    def solve(self, t, state_variables):
        volumes = state_variables[:-2]
        pressures = self.pressures(t, volumes)

        pleft, pright, dxavg, error = self.set_actuation_pressures(state_variables[-2:], pressures[self.controller.var])

        self.actuation.left.source.magnitude = 1.0 * pleft
        self.actuation.right.source.magnitude = 1.0 * pright

        flows = self.flows(t, pressures)
        flows.append(dxavg)
        flows.append(error)

        return flows

    def dynamics(self, initial_state, t_end, t_start: float = 0.0, rtol: float = 1e-9, atol: float = 1e-9):
        t_full = []
        y_full = []
        valve_scin_state = []
        valve_scout_state = []
        valve_pcin_state = []
        valve_pcout_state = []
        flows = []
        event_times = []
        event_values = []
        control_vars = []

        events = [self.event_valve_systemic_in_opening, self.event_valve_systemic_in_closing,
                  self.event_valve_systemic_out_opening, self.event_valve_systemic_out_closing,
                  self.event_valve_pulmonary_in_opening, self.event_valve_pulmonary_in_closing,
                  self.event_valve_pulmonary_out_opening, self.event_valve_pulmonary_out_closing
                  ]

        while t_start < t_end:
            sol = solve_ivp(self.solve, [t_start, t_end], initial_state, events=events, rtol=rtol, atol=atol)
            flows.append(self.solve(sol.t, sol.y)[:-2])

            t_full.append(sol.t)
            y_full.append(sol.y[:-2])
            control_vars.append(sol.y[-2:])

            valve_scin_state.append(self.circulations.systemic.valve_in.state * np.ones_like(sol.t))
            valve_scout_state.append(self.circulations.systemic.valve_out.state * np.ones_like(sol.t))
            valve_pcin_state.append(self.circulations.pulmonary.valve_in.state * np.ones_like(sol.t))
            valve_pcout_state.append(self.circulations.pulmonary.valve_out.state * np.ones_like(sol.t))

            if any([i.size > 0 for i in sol.t_events]):

                event = next(i for i, j in enumerate(sol.t_events) if len(j))
                if event == 0:
                    self.circulations.systemic.valve_in.open()
                elif event == 1:
                    self.circulations.systemic.valve_in.close()
                elif event == 2:
                    self.circulations.systemic.valve_out.open()
                elif event == 3:
                    self.circulations.systemic.valve_out.close()
                elif event == 4:
                    self.circulations.pulmonary.valve_in.open()
                elif event == 5:
                    self.circulations.pulmonary.valve_in.close()
                elif event == 6:
                    self.circulations.pulmonary.valve_out.open()
                elif event == 7:
                    self.circulations.pulmonary.valve_out.close()
                else:
                    print("no valid event")

                event_time = sol.t_events[event][0]
                event_times.append(event_time)
                event_values.append(event)

                print(event_time)
                print(event)
                t_start = event_time
                initial_state = sol.y_events[event][0]
            else:
                t_start = t_end

        times = np.concatenate(t_full)
        volumes = np.concatenate(y_full, axis=1)
        pressures = self.pressures(times, volumes)
        control_vars = np.concatenate(control_vars, axis=1)
        flows = np.concatenate(flows, axis=1)

        valve_scin_state = np.concatenate(valve_scin_state)
        valve_scout_state = np.concatenate(valve_scout_state)
        valve_pcin_state = np.concatenate(valve_pcin_state)
        valve_pcout_state = np.concatenate(valve_pcout_state)
        valves = [event_times, event_values, valve_scin_state, valve_scout_state, valve_pcin_state, valve_pcout_state]

        return [times, volumes, pressures, flows, valves, control_vars]

    @event(direction=1)
    def event_valve_systemic_in_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.systemic.valve_in.event_open(hvL - hs1)

    @event(direction=-1)
    def event_valve_systemic_in_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.systemic.valve_in.event_close(hvL - hs1)

    @event(direction=1)
    def event_valve_systemic_out_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.systemic.valve_out.event_open(hs2 - hvR)

    @event(direction=-1)
    def event_valve_systemic_out_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.systemic.valve_out.event_close(hs2 - hvR)

    @event(direction=1)
    def event_valve_pulmonary_in_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.pulmonary.valve_in.event_open(hvR - hp1)

    @event(direction=-1)
    def event_valve_pulmonary_in_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.pulmonary.valve_in.event_close(hvR - hp1)

    @event(direction=1)
    def event_valve_pulmonary_out_opening(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.pulmonary.valve_out.event_open(hp2 - hvL)

    @event(direction=-1)
    def event_valve_pulmonary_out_closing(self, t, volumes):
        haL, haR, hvL, hvR, hs1, hs2, hp1, hp2 = self.pressures(t, volumes[:-2])
        return self.circulations.pulmonary.valve_out.event_close(hp2 - hvL)
