from utils import TDP
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from typing import Protocol, Any

class Event(Protocol):
    terminal: bool = True
    direction: int = 0

    def __call__(self, t, y, *args):
        pass

def event(terminal: bool = True, direction: float = 0):
    def decorator_event(func: Any) -> Event:
        func.terminal = terminal
        func.direction = direction
        return func

    return decorator_event

class HystereticValve:
    def __init__(self, rmax:float=1000, rmin=0.1, dpmin:float=0.5, dpmax:float=4, initial_state: int=0):
        self.rmax = rmax
        self.rmin = rmin
        self.dpmin = dpmin
        self.dpmax = dpmax
        self.state: int = initial_state
    def resistance(self):
        return self.rmax if self.state == 0 else self.rmin

class System:
    def __init__(self, Pa: TDP = TDP(min=0.001, max=1), E:float = 1.0, V0: float = 0,
                 Cin:float = 0.5, Cout:float = 0.1, R:float = 10,
                 valve_in = HystereticValve(dpmax=0.1, dpmin=0.05, rmax=1000, rmin=1),
                 valve_out = HystereticValve(dpmax=0.1, dpmin=0.05, rmax=1000, rmin=1)):
        self.E = E
        self.Pa = Pa
        self.V0 = V0
        self.Cin = Cin
        self.R = R
        self.Cout = Cout
        self.valve_in = valve_in
        self.valve_out = valve_out

    def __call__(self, t, y):
        vv, pin, pout = y
        pv = self.Pa(t) + self.E * (vv - self.V0)
        qin = (pin - pv) / self.valve_in.resistance()
        qout = (pv - pout) / self.valve_out.resistance()
        qp = (pout - pin) / self.R

        dvv = qin - qout
        dpin = (qp - qin) / self.Cin
        dpout = (qout - qp) / self.Cout

        return [dvv, dpin, dpout]

    @event(direction=1)
    def event_valve_in_opening(self, t, y):
        vv, pin, pout = y
        pv = self.Pa(t) + self.E * (vv - self.V0)
        return (pin - pv) - self.valve_in.dpmax

    @event(direction=-1)
    def event_valve_in_closing(self, t, y):
        vv, pin, pout = y
        pv = self.Pa(t) + self.E * (vv - self.V0)
        return (pin - pv) - self.valve_in.dpmin

    @event(direction=1)
    def event_valve_out_opening(self, t, y):
        vv, pin, pout = y
        pv = self.Pa(t) + self.E * (vv - self.V0)
        return (pv - pout) - self.valve_out.dpmax

    @event(direction=-1)
    def event_valve_out_closing(self, t, y):
        vv, pin, pout = y
        pv = self.Pa(t) + self.E * (vv - self.V0)
        return (pv - pout) - self.valve_out.dpmin

system = System()


t_start = 0.0
t_end = 5
initial_state = [0.0, 0.0, 0.0]

t_full = []
y_full = []
valve_in_state = []
valve_out_state = []
event_times = []

events = [system.event_valve_in_opening, system.event_valve_in_closing,
          system.event_valve_out_opening, system.event_valve_out_closing]

while t_start < t_end:
    sol = solve_ivp(system, [t_start, t_end], initial_state, events=events, rtol=1e-9, atol=1e-9, max_step=0.1)
    t_full.append(sol.t)
    y_full.append(sol.y)
    valve_in_state.append(system.valve_in.state * np.ones_like(sol.t))
    valve_out_state.append(system.valve_out.state * np.ones_like(sol.t))

    if any([i.size > 0 for i in sol.t_events]):

        event = next(i for i, j in enumerate(sol.t_events) if len(j))
        if event == 0:
            system.valve_in.state = 1
        elif event == 1:
            system.valve_in.state = 0
        elif event == 2:
            system.valve_out.state = 1
        elif event == 3:
            system.valve_out.state = 0
        else:
            print("no event")

        event_time = sol.t_events[event][0]
        print(event_time)
        event_times.append(event_time)
        t_start = event_time + 1e-10
        initial_state = sol.y_events[event][0]
    else:
        t_start = t_end

t_full = np.concatenate(t_full)
y_full = np.concatenate(y_full, axis=1)
valve_in_state = np.concatenate(valve_in_state)
valve_out_state = np.concatenate(valve_out_state)

vv, pin, pout = y_full
pv = system.Pa(t_full) + system.E * (vv - system.V0)

plt.plot(t_full, system.Pa(t_full), 'k--', label='Pa')
plt.plot(t_full, pin, 'ro-', label='pin')
plt.plot(t_full, pout, 'go-', label='pout')
plt.plot(t_full, pv, 'bo-', label='pv')
plt.plot(t_full, pin-pv, 'mo-', label='pin-pv')
plt.plot(t_full, pv-pout, 'ko-', label='pv-pout')
plt.axhline(system.valve_in.dpmax)
plt.axhline(system.valve_in.dpmin)
[plt.axvline(i) for i in event_times]

plt.plot(t_full, valve_in_state, 'g--', label='valve_in')
plt.plot(t_full, valve_out_state, 'r--', label='valve_out')
plt.legend()

plt.plot(t_full, 10*vv, 'y-', label='vv')
plt.legend()

plt.legend()
plt.xlabel('time')
plt.legend()
plt.show()



