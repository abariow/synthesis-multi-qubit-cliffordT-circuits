import numpy as np
import sympy as sp

class Zw:

    def __init__(self, a=0, b=0, c=0, d=0):
        self.cff = np.array([d, c, b, a])

    def __str__(self):
        return ""

    def __eq__(self, other):

        if not isinstance(other, Zw):
            return False

        return (self.cff == other.cff).all()

    def __add__(self, other):

        t = Zw()
        u = self.cff

        if isinstance(other, Zw):
            v = other.cff
            t.cff = u + v

        elif isinstance(other, int):
            v = np.array([other, 0, 0, 0])
            t.cff = u + v

        else:
            raise TypeError("Wrong type")

        return t

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return other + (-1 * self)

    def __mul__(self, other):

        u = self.cff

        if isinstance(other, Zw):

            v = other.cff
            a = u[3] * v[0] + u[2] * v[1] + u[1] * v[2] + u[0] * v[3]
            b = u[2] * v[0] + u[1] * v[1] + u[0] * v[2] - u[3] * v[3]
            c = u[0] * v[1] + u[1] * v[0] - u[2] * v[3] - u[3] * v[2]
            d = u[0] * v[0] - u[3] * v[1] - u[2] * v[2] - u[1] * v[3]
            t = Zw(a, b, c, d)

        elif isinstance(other, int):

            t = Zw()
            t.cff = other * u

        elif other == 0.5:

            t = Zw()
            t.cff = u // 2

        else:
            raise TypeError("Wrong type")

        return t

    def __rmul__(self, other):
        return self * other

    def to_latex(self):
        omega = sp.Symbol(r'\omega')
        return self.cff[3]*omega**3 + self.cff[2]*omega**2 + self.cff[1]*omega**1 + self.cff[0]*omega**0

    def conjug(self):
        return Zw(-self.cff[1], -self.cff[2], -self.cff[3], self.cff[0])

    def norm(self):
        return self.conjug() * self

    def residue(self):
        R = Zw(self.cff[3] % 2, self.cff[2] % 2, self.cff[1] % 2, self.cff[0] % 2)
        return R

    def reduce(self):

        k = 0
        temp = self

        x = temp.residue().norm().residue().cff_str()

        while x == "0000":
            temp = (temp * Zw.root2()) * 0.5
            x = temp.residue().norm().residue().cff_str()
            k += 1

        self.cff = temp.cff
        return k

    def is_reducible(self):
        return self.residue().norm().residue().cff_str() == "0000"


    def cff_str(self):
        return "".join(str(c) for c in self.cff[::-1])

    def circular_left_shift(self):
        self.cff = np.roll(self.cff, 1)

    def align(self, other):

        m = 0
        while self.cff_str() != other.cff_str():
            other.circular_left_shift()
            m += 1
            if m >= 4:
                return -1

        return m

    def complement(self):
        r = self.residue()
        r.cff += 1
        return r.residue()

    def one():
        return Zw(0, 0, 0, 1)

    def omega():
        return Zw(0, 0, 1, 0)

    def root2():
        return Zw(-1, 0, 1, 0)

    def root2_power(k):

        if k % 2 == 0:
            return 2 ** (k // 2) * Zw.one()
        else:
            return 2 ** ((k - 1) // 2) * Zw.root2()

    def num(self):
        rr2 = np.sqrt(2) ** -1
        omega = np.complex(rr2, rr2)
        omega2 = omega ** 2
        omega3 = omega ** 3
        return self.cff[0] + self.cff[1] * omega + self.cff[2] * omega2 + self.cff[3] * omega3


if __name__ == "__main__":

    x = Zw(1,0,0,1)



