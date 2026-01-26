from math import sin, cos, pi

class Pouch:

    def energy(self, x, force, pressure):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta)
        :param force: normalized force Fn = FL/uLHD = F/uHD
        :param pressure: normalized actuation pressure Pn = PL^2D/uLHD = PL/uH
        :return: normalized energy En = E / uLHD
        """
        strain_energy = self.sed(x) * self.mat_volume(x)
        work_force = force * (self.width(x) - 1)
        work_pressure = pressure * self.volume(x)
        return strain_energy - work_force - work_pressure

    @staticmethod
    def width(x):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta)
        :return: normalized width wh = w / L = lh sin(theta) / theta
        """
        return x[0] * sin(x[3]) / x[3]

    @staticmethod
    def volume(x):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta)
        :return: normalized pouch volume vph = vp / DL**2 = 0.5 lh**2 dh (theta - sin(theta) cos(theta)) / theta**2
        """
        return x[0] ** 2 * x[2] * (x[3] - sin(x[3]) * cos(x[3])) / x[3] ** 2 / 2

    @staticmethod
    def mat_volume(x):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta)
        :return: normalized pouch material volume vmph = vmp / LHD = 2 lh hh dh
        """
        return 2 * x[0] * x[1] * x[2]

    @staticmethod
    def sed(x):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta)
        :return: normalized strain energy density Wh = W / u = 0.5 (x1**2 + x2**2 + x3**2 - 3)
        """
        return (x[0]**2 + x[1]**2 + x[2]**2 - 3) / 2

class PouchArray:

    def __init__(self, Lsh: float = 0.02):
        """
        :param Lsh: relative seal length Lsh = Ls / L
        """
        self.Lsh = Lsh

    def energy(self, x, force, pressure):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta, lsh = ls/Ls, th = t/H)
        :param force: normalized force Fn = FL/uLHD = F/uHD
        :param pressure: normalized actuation pressure Pn = PL^2D/uLHD = PL/uH
        :return: normalized energy En = E / NuLHD
        """
        xp = x[:4] # pouch variables xp = lh, hh, dh, theta
        xs = [x[4], x[5], x[2]] # seal variables xs = lsh, th, dh

        strain_energy_pouch = Pouch.sed(xp) * Pouch.mat_volume(xp)
        strain_energy_seal = self.Lsh * Pouch.sed(xs) * Pouch.mat_volume(xs)
        work_force = force * self.extension(x)
        work_pressure = pressure * Pouch.volume(xp)
        return strain_energy_pouch + strain_energy_seal - work_force - work_pressure

    def extension(self, x):
        """
        Normalized absolute change in array length.

        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta, lsh = ls/Ls, th = t/H)
        :return: array extension wt = (wa - Wa) / FNL = wh + (lsh - 1)Lsh - 1
        """
        return Pouch.width(x[:4]) + (x[4] - 1) * self.Lsh - 1

    def width(self, x):
        """
        Normalized array width.

        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta, lsh = ls/Ls, th = t/H)
        :return: wah = wa / Wa = (wh + lsh Lsh) / (1 + Lsh)
        """
        return (Pouch.width(x[:4]) + x[4] * self.Lsh) / (1 + self.Lsh)

class CylindricalPouchArray(PouchArray):
    def __init__(self, Lsh: float = 0.02, N: int = 10):
        self.N = N
        super().__init__(Lsh)

    def vcb(self, x):
        return x[2] * (Pouch.width(x[:4]) + x[4] * self.Lsh)**2 / 4 / pi

    def ab(self, x):
        """
        Conversion factor from cylidner pressure to hoop force.

        ab = Fh / Pc / NLD
        """
        return x[2] * (Pouch.width(x[:4]) + x[4] * self.Lsh) / 2 / pi

    def cylinder_volume(self, x):
        """
        Normalized cylinder volume.

        vch = vc / DNL**2 = N vcb - vph / 2
        with vcb = dh (wh + lsh Lsh)**2 / 4 / pi
        """
        return self.N * self.vcb(x) - Pouch.volume(x[:4]) / 2

    def energy(self, x, pressure_cylinder, pressure):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta, lsh = ls/Ls, th = t/H)
        :param force: normalized cylinder pressure Pcn = NL Pc / uH
        :param pressure: normalized actuation pressure Pn = PL^2D/uLHD = PL/uH
        :return: normalized energy En = E / NuLHD
        """
        xp = x[:4] # pouch variables xp = lh, hh, dh, theta
        xs = [x[4], x[5], x[2]] # seal variables xs = lsh, th, dh

        strain_energy_pouch = Pouch.sed(xp) * Pouch.mat_volume(xp)
        strain_energy_seal = self.Lsh * Pouch.sed(xs) * Pouch.mat_volume(xs)
        work_force = pressure_cylinder * self.ab(x) * self.extension(x)
        work_pressure = pressure * Pouch.volume(xp)
        return strain_energy_pouch + strain_energy_seal - work_force - work_pressure




class PouchArray2(PouchArray):

    def __init__(self, Lwh: float = 0.02, N = 10):
        """
        :param Lwh: relative seal length Lwh = Ls / Wa
        :param N: number of pouches
        """
        self.Lwh = Lwh
        self.N = N

        """Relative seal length to pouch length from 
        relative seal length to array length and number of pouches."""

        lwh2lsh = 1 / (1 / (self.N * self.Lwh) - 1)
        super().__init__(lwh2lsh)

        """Conversion factor for different normalization of array pouch volume.
                pa2p = ph / pah, with pah = P Wa / uNH"""
        self.pa2p = 1 - self.N * self.Lwh

        """Conversion factor for different normalization of array pouch volume.
        lwb = vab / vah, with vab = va N / (D Wa**2)"""
        self.vh2vb = 1 - 2 * self.N * self.Lwh + self.N**2 * self.Lwh**2


    def volume(self, x):
        """
        Normalized array pouch volume with normalization via
        vab = va N / D Wa**2

        :param x:
        :return:
        """
        return self.vh2vb * Pouch.volume(x[:4])

    def energy(self, x, force, pressure):
        """
        :param x: design variables x = (lh = l/L, hh = h/H, dh = d/D, theta, lsh = ls/Ls, th = t/H)
        :param force: normalized force Fn = FL/uLHD = F/uHD
        :param pressure: normalized actuation pressure Pa = P Wa / uNH
        :return: normalized energy En = E / NuLHD
        """

        return super().energy(x, force, self.pa2p * pressure)