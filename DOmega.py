import numpy as np
import sympy as sp
from ZOmega import Zw

class Dw:

    def __init__(self, zw, n=0):

        self.zw = zw
        self.n = n

    def __str__(self):
        return "power(1/root2, {}) * ({})".format(self.n, self.zw)

    def __eq__(self, other):

        if not isinstance(other, Dw):
            return False

        self.reduce()
        other.reduce()

        return self.zw == other.zw and self.n == other.n

    def __add__(self, other):

        zw1 = self.zw

        if isinstance(other, Zw):
            zw2 = other
            othern = 0

        elif isinstance(other, int):
            zw2 = Zw(0, 0, 0, other)
            othern = 0

        elif isinstance(other, Dw):
            zw2 = other.zw
            othern = other.n

        dn = self.n - othern
        p = Zw.root2_power(abs(dn))

        if dn >= 0:
            zw2 = zw2 * p
            n = self.n
        else:
            zw1 = zw1 * p
            n = other.n

        result = Dw(zw1 + zw2, n)
        result.reduce()

        return result


    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return  self + (-1 * other)

    def __rsub__(self, other):
        return  other + (-1 * self)

    def __mul__(self, other):

        dw = None

        if isinstance(other, Dw):
            dw = Dw(self.zw * other.zw, self.n + other.n)

        elif isinstance(other, (Zw, int, float)):
            dw = Dw(self.zw * other, self.n)

        else:
            raise TypeError("Wrong type")

        if dw.is_zero():
            dw.n = 0

        return dw

    def __rmul__(self, other):
        return self * other

    def to_latex(self):
        return self.zw.to_latex() / (sp.sqrt(2) ** self.n)

    def conjug(self):
        return Dw(self.zw.conjug(), self.n)

    def norm(self):
        return Dw(self.zw.norm(), 2 * self.n)

    def k_residue(self, k=0):

        if k == 0:
            k = self.n

        if k >= self.n:
            return (Zw.root2_power(k - self.n) * self.zw).residue()

    def reduce(self):

        if self.zw.cff_str() == "0000":
            self.n = 0
            return

        k = self.zw.reduce()
        self.n -= k

    def is_zero(self):
        return self.zw.cff_str() == "0000"

    def num(self):
        return self.zw.num() / (np.sqrt(2) ** self.n)


if __name__ == "__main__":

    dw1 = Dw(Zw(0,0,0,1),1)
    dw2 = Dw(Zw(0,0,1,1),0)

    print(dw1.num())


