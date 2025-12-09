""""
Lumped-parameter models of dynamic flow circuits that inverter DC flow to AC flow, typically using hysteretic components.
"""

class Inverter:
    pass

class RLCCircuit(Inverter):
    """
    Parallel RC, series L circuit with resistance in parallel with capacitor, known as low-pass filter.

    Corner frequency w = 1 / sqrt(LC).
    """
    def __init__(self,
                 R = lambda t, h: 1.0,
                 C = lambda t: 1.0,
                 L = lambda t: 1.0):
        self.R = R
        self.C = C
        self.L = L

    def solve(self, t, h_pump, q, h):
        """
        h_pump: pump head
        q: pump flow rate
        h: circuit head
        """
        dq = (h_pump - h) / self.L(t)
        dh = q / self.C(t) - h / (self.R(t, h) * self.C(t))
        return [dq, dh]