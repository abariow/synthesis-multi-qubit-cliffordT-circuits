import numpy as np
import sympy as sp
from ZOmega import Zw

class Dw:

    """
        This class describes D-omega numbers
        D-omega is a set, pretty much like the set Z-omega. A D-omega number is
        a Z-omega number divided n times by sqrt(2) i.e Dw = Zw / sqrt(2) ^ n.
        It means that D-omega numbers can also be represented in the form of 3th
        degree polynomial of omega but the coefficients are not integers,
        the coefficients are in the form of c / sqrt(2) ^ n, where c is an
        integer. ( D-omega equals the ring Z[1/sqrt(2), i] )

        An object of this class represents a memeber of D-omega set

        Args:
            zw (Zw): Z-omega object
            n (int): exponent of sqrt(2), sould be greater than 0

        Attributes:
            zw (Zw): To store Z-oemga object
            n (int): To store exponent of sqrt(2)

    """
    # Constructor
    def __init__(self, zw, n=0):
        # Initializing the attributes
        self.zw = zw
        self.n = n

###############################################################################
# Magic methods

    def __str__(self):
        return "power(1/root2, {}) * ({})".format(self.n, self.zw)

    def __eq__(self, other):

        """ This method checks equality of two D-omega objects """

        if not isinstance(other, Dw):
            return False

        self.reduce()
        other.reduce()

        return self.zw == other.zw and self.n == other.n

    def __add__(self, other):

        """ This method Adds two D-omega objects """

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

        """ This method multiplies two D-omega objects"""

        dw = None

        if isinstance(other, Dw): # if other is a Dw object
            dw = Dw(self.zw * other.zw, self.n + other.n)

        elif isinstance(other, (Zw, int, float)): # if other is int or float
            dw = Dw(self.zw * other, self.n)

        else:
            raise TypeError("Wrong type")

        if dw.is_zero():
            dw.n = 0

        return dw

    def __rmul__(self, other):
        return self * other

###############################################################################
# Normal methods:

    def to_latex(self):

        """
            This method makes a latex expression(using sympy library)
            to show the object perfectly
        """

        return self.zw.to_latex() / (sp.sqrt(2) ** self.n)

    def conjug(self):

        """ This method returns complex conjugate of object """

        return Dw(self.zw.conjug(), self.n)

    def norm(self):

        """ This method returns norm of object """

        return Dw(self.zw.norm(), 2 * self.n)

    def k_residue(self, k=0):

        """
            This method, in case k >= Dw.n, multiplies D-omega object by
            sqrt(2) k times, then returns the residue of that
            i.e if k >= Dw.n then returns residue(Dw * sqrt(2) ^ k)
        """

        if k == 0:
            # In this case(defualt case), k considered self.n
            k = self.n

        if k >= self.n:
            return (Zw.root2_power(k - self.n) * self.zw).residue()

    def reduce(self):

        """
            This method reduces the D-omega object. As you know, the D-omega object
            can be considered as a fraction, this method simplyfies(reduces) the fraction,
            by removing common sqrt(2) factors in numerator and denominator
            (decreases the denominator exponent).
        """

        if self.zw.cff_str() == "0000":
            self.n = 0
            return

        # Remove common sqrt(2) factors in self.zw( sqrt(2) factors in numerator )
        k = self.zw.reduce()

        # Remove common sqrt(2) factors in denominator
        self.n -= k

    def is_zero(self):

        """ Returns true if the Dw-object equals zero, o.w returns false """

        return self.zw.cff_str() == "0000"

    def num(self):

        """
            This methods return numberical form of D-omega object i.e (x + yi)
            For example let self.zw = [0, 3, 1, 0] and self.n = 1 then the method
            returns 1/sqrt(2) + 3i/sqrt(2)
        """

        return self.zw.num() / (np.sqrt(2) ** self.n)

