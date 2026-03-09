from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np
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
    def __init__(self, q_pump:float = 1, C1:float=1, C2:float=1, valve:HystereticValve=HystereticValve()):
        self.q_pump = q_pump
        self.C1 = C1
        self.C2 = C2
        self.valve = valve

    def __call__(self, t, y):
        p1, p2 = y
        q_valve = (p1 - p2) / self.valve.resistance()
        dp1 = (self.q_pump - q_valve) / self.C1
        dp2 = (q_valve - self.q_pump) / self.C2
        return [dp1, dp2]

    @event(direction=1)
    def event_valve_opening(self, t, y):
        p1, p2 = y
        return (p1 - p2) - self.valve.dpmax

    @event(direction=-1)
    def event_valve_closing(self, t, y):
        p1, p2 = y
        return (p1 - p2) - self.valve.dpmin

system = System()

t_start = 0.0
t_end = 10
initial_state = [1.0, 1.0]

t_full = []
y_full = []
valve_state = []
event_times = []

events = [system.event_valve_opening, system.event_valve_closing]

while t_start < t_end:
    sol = solve_ivp(system, [t_start, t_end], initial_state, events=events, rtol=1e-9, atol=1e-9, max_step=0.1)
    t_full.append(sol.t)
    y_full.append(sol.y)
    valve_state.append(system.valve.state * np.ones_like(sol.t))

    if any([i.size > 0 for i in sol.t_events]):

        event = next(i for i, j in enumerate(sol.t_events) if len(j))
        if event == 0:
            system.valve.state = 1
        elif event == 1:
            system.valve.state = 0
        else:
            print("hoi")

        event_time = sol.t_events[event][0]
        event_times.append(event_time)
        t_start = event_time
        initial_state = sol.y_events[event][0]
    else:
        t_start = t_end

t_full = np.concatenate(t_full)
y_full = np.concatenate(y_full, axis=1)
valve_state = np.concatenate(valve_state)

p1, p2 = y_full

plt.figure()
plt.plot(t_full, p1, 'ko-', label="p1")
plt.plot(t_full, p2, 'g-', label="p2")
plt.plot(t_full, valve_state, 'r-', label="state")
[plt.axvline(i, color='black', linestyle='--') for i in event_times]

plt.figure()
plt.axhline(system.q_pump, linestyle='-', label="q_pump")
qhv = (p1 - p2) / [system.valve.rmax if i == 0 else system.valve.rmin for i in valve_state]
plt.plot(t_full, qhv, 'bo-', label="qhv")
plt.plot(t_full, system.q_pump - qhv, 'k-', label="qc1")
plt.plot(t_full, qhv - system.q_pump, 'r-', label="qc2")
[plt.axvline(i, color='black', linestyle='--') for i in event_times]



plt.xlabel("Time (s)")
plt.ylabel("Pressure head (m)")
plt.legend()
plt.show()

