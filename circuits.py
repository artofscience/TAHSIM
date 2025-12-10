""""
Lumped-parameter models of dynamic flow circuits that inverter DC flow to AC flow, typically using hysteretic components.
"""

class Oscillator:
    """
    Resistance with pressure-dependent hysteresis.
    """
    def __init__(self, Ropen: float = 1e4, Rclosed: float = 1e8):
        self.Ropen = Ropen
        self.Rclosed = Rclosed
        self.closed = True
        self.dhopen = 35
        self.dhclose = 10

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
    pass

class NLRLCircuit(Circuit):
    """
    Pump --> Impedance --> Nonlinear resistor.
    """
    def __init__(self, resistance, inductance):
        self.resistance = resistance
        self.inductance = inductance

    def solve(self, t, h_pump, q):
        h = self.resistance * q**2
        dq = (h_pump - h) / self.inductance
        return [dq]


class RLCCircuit(Circuit):
    """
    Parallel RC, series L circuit with resistance in parallel with capacitor, known as low-pass filter.

    Impedance L connected in series with pump with state (q, h_pump).

    System is open-loop; pump is connected to ground, resistance and capacitor are connected to ground.

    All components (R, L, C) are assumed a function of time, resistance assumed also a function of pressure difference.
    """
    def __init__(self,
                 R = lambda t, h: 1.0,
                 C = lambda t: 1.0,
                 L = lambda t: 1.0,
                 hR0 = lambda t: 0.0,
                 dhC0 = lambda t: 0.0,
                 h0pump = lambda t: 0.0):
        self.R = R
        self.C = C
        self.L = L
        self.hR0 = hR0 # pressure (head) at output of resistor
        self.dhC0 = dhC0 # rate of pressure (head) at other side of capacitor
        self.h0pump = h0pump # pressure (head) at inlet of pump

    def solve(self, t, h_pump, q, h):
        """
        h_pump: pump head
        q: pump flow rate
        h: circuit head
        """
        impedance_head = self.h0pump(t) + h_pump - h[0]
        dq = impedance_head / self.L(t)

        resistor_head = h[0] - self.hR0(t)
        qr = resistor_head / self.R(t, resistor_head)
        qc = q - qr
        dh = qc / self.C(t) + self.dhC0(t)
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
                 R = lambda t, h: 1.0,
                 C = lambda t: 1.0,
                 L = lambda t: 1.0,
                 Rout = lambda t: 1.0,
                 Cac = lambda t: 1.0,
                 hR0 = lambda t: 0.0,
                 dhC0 = lambda t: 0.0,
                 dhCac = lambda t: 0.0,
                 h0pump = lambda t: 0.0):
        super().__init__(R, L, C, hR0, dhC0, h0pump)
        self.Rout = Rout
        self.Cac = Cac
        self.dhCac = dhCac

    def solve(self, t, h_pump, q, h):
        """
        h_pump: pump head
        q: pump flow rate
        h: circuit head
        """
        impedance_head = self.h0pump(t) + h_pump - h[0]
        dq = impedance_head / self.L(t)

        hv_head = h[0] - h[1]
        qhv = hv_head / self.R(t, hv_head)
        qc = q - qhv
        dh = qc / self.C(t) + self.dhC0(t)

        qr = (h[1] - self.hR0(t)) / self.Rout(t)
        qac = qhv - qr
        dhac = qac / self.Cac(t) + self.dhCac(t)
        return [dq, dh, dhac]