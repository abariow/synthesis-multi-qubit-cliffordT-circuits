import numpy as np
import sympy as sp


class Zw:

    """
        This class describes Z-omega numbers
        Z-omega is a set of special complex numbers, These numbers can be
        represented in the form of 3th degree polynomial of omega(= (1 + i)/sqrt(2))
        with integer coefficients.
        ( let w = omega then a Z-omega object is like: c3 * w ^ 3 + c2 * w ^ 2 + c1 * w + c0
            where c0, c1, c2, c3 are integers )

        An object of this class represents a memeber of Z-omega set

        Args:
            a (int): 1st coefficient(c0)
            b (int): 2nd coefficient(c1)
            c (int): 3rd coefficient(c2)
            d (int): leading(4th) coefficient(c3)

        Attributes:
            cff (array): to store coefficients
                         cff[0] = a
                         cff[1] = b
                         cff[2] = c
                         cff[3] = d
    """

    # Constructor
    def __init__(self, a=0, b=0, c=0, d=0):
        # create a numpy array to store coefficients
        self.cff = np.array([d, c, b, a])


###############################################################################
# Magic methods

    def __str__(self):
        return ""

    def __eq__(self, other):

        """ This method checks equality of two Z-omega objects """

        if not isinstance(other, Zw):
            return False

        return (self.cff == other.cff).all() # compare coefficients

    def __add__(self, other):

        """ This method Adds two Z-omega objects """

        t = Zw()
        u = self.cff

        if isinstance(other, Zw): # if other is Z-omega object
            # Add 2 ploynomial(add the coefficients)
            v = other.cff
            t.cff = u + v

        elif isinstance(other, int): # if other is an integer
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

        """ This method multiplies two Z-omega objects  """

        u = self.cff

        if isinstance(other, Zw): # if other is a Z-omega object

            # multiply 2 polynomial
            v = other.cff
            a = u[3] * v[0] + u[2] * v[1] + u[1] * v[2] + u[0] * v[3]
            b = u[2] * v[0] + u[1] * v[1] + u[0] * v[2] - u[3] * v[3]
            c = u[0] * v[1] + u[1] * v[0] - u[2] * v[3] - u[3] * v[2]
            d = u[0] * v[0] - u[3] * v[1] - u[2] * v[2] - u[1] * v[3]
            t = Zw(a, b, c, d)

        elif isinstance(other, int): # if other is an integer

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


###############################################################################
# Normal methods:

    def to_latex(self):

        """
            This method makes a latex expression(using sympy library)
            to show the object perfectly
        """

        omega = sp.Symbol(r'\omega') # using sympy library
        return self.cff[3]*omega**3 + self.cff[2]*omega**2 + self.cff[1]*omega**1 + self.cff[0]*omega**0

    def conjug(self):

        """ This method returns complex conjugate of object """

        return Zw(-self.cff[1], -self.cff[2], -self.cff[3], self.cff[0])

    def norm(self):

        """ This method returns norm of object """

        return self.conjug() * self

    def residue(self):

        """
            This method returns residue of object
            Residue of a Z-omega number is a Z2-omega number, which its
            coefficeints are either 0 or 1 (coefficents % 2)
        """

        R = Zw(self.cff[3] % 2, self.cff[2] % 2, self.cff[1] % 2, self.cff[0] % 2)
        return R

    def is_reducible(self):

        """
            This method returns True if the Z-omega object is reducible
            (a Z-omega object is reducible if it has at least one sqrt(2) factor
            i.e self = sqrt(2) * other, so that can be divided by sart(2))

            If residue(norm(residue(self))) = [0,0,0,0] we can conclude self is reducible
        """

        return self.residue().norm().residue().cff_str() == "0000"


    def reduce(self):

        """
            This method divides self by sqrt(2) as long as it's divisible(until
            there's no sqrt(2) factor in self). In other words, this method removes
            sqrt(2) factors in the Z-omega object.
            This method also alculates how many sqrt(2) factors there are in the object
            In other words, if self = (sqrt(2) ^ k) * other(=another Z-omega
            object having no sqrt(2) factors) then this method divides self by sqrt(2)
            k times and updates: self <- [self / (sqrt(2) ^ k)]
            and as a result, self = other. then finally returns k
        """

        k = 0 # variable ot store number of sqrt(2) factors
        temp = self

        x = temp.residue().norm().residue().cff_str()

        # loop till x is reducible
        while x == "0000": # check reducibility
            temp = (temp * Zw.root2()) * 0.5 # division by sqrt(2)
            x = temp.residue().norm().residue().cff_str() # update x
            k += 1 # count number of sqrt(2) factors

        self.cff = temp.cff
        return k


    def cff_str(self):

        """
            This methods return the coefficients together in string format. i.e
            'dcba'. for example "1101"
        """

        return "".join(str(c) for c in self.cff[::-1])

    def circular_left_shift(self):

        """
            This method cicular shift the coefficients of plolynomial to left
            [d, c, b, a] ----after the method is called-------> [c, b, a, d]
        """

        self.cff = np.roll(self.cff, 1)

    def align(self, other):

        """
            This method ,in case of possibility, align coefficients of other
            with coefficiets of self, using circular shifting coefficients of
            other, then returns how many times the coefficients of other have
            been shifted, if alignment isn't possible returns -1.
            For example let self be [0,1,1,1] and other be [1,1,0,1], if we
            shift other 2 times to left, they would be aligned.
        """

        m = 0 # number of shifting

        # check if the coefficients of other and self are aligned
        while self.cff_str() != other.cff_str():
            other.circular_left_shift() # circular shift to left
            m += 1 # count number of shifts
            if m >= 4: # returns -1 if alignment is not possible
                return -1

        return m

    def complement(self):

        """
            This methods returns 1's complement of coefficients, in case of
            the coefficients are either 0 or 1
            For example if self be [1,0,1,1] the this method returns [0,1,0,0]
        """

        r = self.residue()
        r.cff += 1
        return r.residue()

    def num(self):

        """
            This methods return numberical form of Z-omega object i.e (x + yi)
            For example let self be [0,3,1,0] then the method returns (1 + 3i)
        """

        rr2 = np.sqrt(2) ** -1
        omega = np.complex(rr2, rr2)
        omega2 = omega ** 2
        omega3 = omega ** 3
        return self.cff[0] + self.cff[1] * omega + self.cff[2] * omega2 + self.cff[3] * omega3


###############################################################################
# Statics function:

    def one():

        """ This function returns an Zw object representing 1 """

        return Zw(0, 0, 0, 1)

    def omega():

        """
            This function returns an Zw object representing omega
            ( omega = (1 + i) / sqrt(2) )
        """

        return Zw(0, 0, 1, 0)

    def root2():

        """ This function returns an Zw object representing sqrt(2) """

        return Zw(-1, 0, 1, 0)

    def root2_power(k):

        """ This function returns an Zw object representing (sqrt(2) ^ k) """

        if k % 2 == 0:
            return 2 ** (k // 2) * Zw.one()
        else:
            return 2 ** ((k - 1) // 2) * Zw.root2()

