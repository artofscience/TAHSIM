import numpy as np

""""
Lumped-parameter models of dynamic flow circuits that inverter DC flow to AC flow, typically using hysteretic components.
"""

class Oscillator:
    """
    Resistance with pressure-dependent hysteresis.
    """
    def __init__(self,
                 Ropen: float = 1,
                 Rclosed: float = 100,
                 dhopen: float = 3,
                 dhclose: float = 1):
        self.Ropen = Ropen
        self.Rclosed = Rclosed
        self.closed = True
        self.dhopen = dhopen
        self.dhclose = dhclose

    def __call__(self, t, dh):

        if dh > self.dhopen:
            self.closed = False
            return self.Ropen
        elif dh < self.dhclose:
            self.closed = True
            return self.Rclosed
        else:
            return self.Rclosed if self.closed else self.Ropen

class Circuit:
    def solve(self, t, h_pump, param):
        pass

class NLRLCircuit(Circuit):
    """
    Pump --> Impedance --> Nonlinear resistor.
    """
    def __init__(self, resistance, impedance: float = 0.01):
        self.resistance = resistance
        self.impedance = impedance
        self.resistance_power = 1

    def h(self, t, y):
        return self.resistance(t) * np.power(y, self.resistance_power)

    def solve(self, t, h_pump, y):
        return [(h_pump - self.h(t, y[0])) / self.impedance]

class RLCCircuit(Circuit):
    """
    Parallel RC, series L circuit with resistance in parallel with capacitor, known as low-pass filter.

    Impedance L connected in series with pump with state (q, h_pump).

    System is open-loop; pump is connected to ground, resistance and capacitor are connected to ground.

    All components (R, L, C) are assumed a function of time, resistance assumed also a function of pressure difference.
    """
    def __init__(self,
                 resistance = lambda t, h: 1.0,
                 capacitance = lambda t: 1.0,
                 impedance: float = 0.01):
        self.resistance = resistance
        self.capacitance = capacitance
        self.impedance = impedance


    def solve(self, t, h_pump, y):
        """
        h_pump: pump head
        y = [q, h]
        q: pump flow rate
        h: circuit head
        """
        impedance_head = h_pump - y[1]
        dq = impedance_head / self.impedance

        qr = y[1] / self.resistance(t, y[1])
        qc = y[0] - qr
        dh = qc / self.capacitance(t)
        return [dq, dh]

class RLCRCCircuit(RLCCircuit):
    """
    Parallel RC, series L circuit with resistance in parallel with capacitor, known as low-pass filter.

    Impedance L connected in series with pump with state (q, h_pump).

    System is open-loop; pump is connected to ground, capacitor is connected to ground.
    Resistor is connected in series to a parallel RC!

    All components (R, L, C, R, C) are assumed a function of time, resistance assumed also a function of pressure difference.
    """
    def __init__(self,
                 resistance = lambda t, h: 1.0,
                 capacitance = lambda t: 1.0,
                 impedance: float = 0.01,
                 Rout = lambda t: 1.0,
                 Cac = lambda t: 1.0):
        super().__init__(resistance, capacitance, impedance)
        self.Rout = Rout
        self.Cac = Cac

    def solve(self, t, h_pump, y):
        """
        h_pump: pump head
        y = [q, h, hac]
        q: pump flow rate
        h: circuit head
        hac: circuit head actuator
        """
        impedance_head = h_pump - y[1]
        dq = impedance_head / self.impedance

        hv_head = y[1] - y[2]
        qhv = hv_head / self.resistance(t, hv_head)
        qc = y[0] - qhv
        dh = qc / self.capacitance(t)

        qr = y[2] / self.Rout(t)
        qac = qhv - qr
        dhac = qac / self.Cac(t)
        return [dq, dh, dhac]