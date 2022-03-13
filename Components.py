import numpy as np
import sympy as sp

from DOmega import Dw
from ZOmega import Zw
from Gates import gates

from qiskit import QuantumCircuit
from qiskit import Aer,execute



def getIdentityMatrix(n):

    """ This function returns an n*n identity matrix """

    I = np.full((n, n), Dw(Zw()), dtype=object)
    for i in range(n):
        I[i,i] = Dw(Zw.one())
    return np.matrix(I)

def makeTLMatrix(A, n, i, j):

    """
        This function makes an n*n two-level matrix, i.e A[i,j]

        Args:
            A (numpy matrix): a 2*2 matrix
            n (int): size of matrix (n > 1)
            i (int): indicates first component
            j (int): indicates second component

        Returns:
            An n*n numpy matrix
    """

    TLM = getIdentityMatrix(n)
    TLM[i,i] = A[0,0]
    TLM[i,j] = A[0,1]
    TLM[j,i] = A[1,0]
    TLM[j,j] = A[1,1]

    return TLM

def makeOLMatrix(a, n, i):

    """
        This function makes an n*n one-level matrix, i.e a[i]

        Args:
            a (Complex): a scalar
            n (int): size of matrix (n > 1)
            i (int): indicates the component matrix trivially act on

        Returns:
            An n*n numpy matrix
    """

    OLM = getIdentityMatrix(n)
    OLM[i,i] = a

    return OLM

class HighLevelComponent:

    """
        This class describes two-level(and also one-level) matrices.
        (Recall that a two-level matrix is an n * n-matrix that acts
        non-trivially on at most two vector components)

        An object of this class represents a two-level(or one-level) matrix)

        Args:
            name (str): name of gate. for example 'H' or 'T'
            power (int): exponent of matrix (power > 0)
            n (int): size of matrix (n * n), (n > 1)
            i (int): indicates first component matrix trivially act on (i >= 0)
            j (int): indicates second component matrix trivially act on (j > i)

        For example, by calling HighLevelComponent(name='T', power=3, n=8, i=2, j=5)
        the two-level matrix T[2,5] ^ 3 with size 8 * 8 will be created


        Attributes:
            name (str): to store name of unitary gate. for example 'H' or 'T'
            power (int): to store exponent of matrix
            N (int): to store size of matrix (n)
            matrix (numpy.matrix): to store the gate matrix (N*N-matirx)
            idx (list): to store i and j (= [i, j])

    """

    # Constructor
    def __init__(self, name, power, n, i, j=0):

        # Initializing the attributes
        self.name = name # name can be 'T', 'H', 'X' or 'w'
        self.power = power
        self.N = n

        gate = gates[name] # Find uniatry matrix by the name of gate

        if name == 'w': # w = omega ( w ^ 2 = omega ^ 2 = sqrt(-1) )
            # In this case there is no gate but a one-level matrix of type omega is created

            # create one-level matrix of type omega (omega[i] or w[i])
            self.matrix = makeOLMatrix(gate, n, i)

            self.idx = [i]

        else:
            # In this case, a two-level matrix
            self.matrix = makeTLMatrix(gate, n, i, j)
            self.idx = [i,j]

    def powered_matrix(self):

        """ This method returns the matrix(power is considered) """

        return self.matrix ** self.power # a numpy matrix is returned

    def TC(self):

        """
            This method replace the matrix with its inverse
            ( Inverse of a unitary matrix equals to its transpose conjugate )
        """

        # inverse(H) = H ^ -1 = H ---> H[i,j] ^ -1 = H[i,j]
        # inverse(X) = X ^ -1 = X ---> X[i,j] ^ -1 = X[i,j]
        # inverse(T) = T ^ -1 = T ^ 7 ---> T[i,j] ^ -1 = T[i,j] ^ 7
        # inverse(T ^ m) = T ^ -m  = T ^ (8 - m) ---> T[i,j] ^ -m = T[i,j] ^ (8 - m)
        # inverse(w[i] ^ m) = w[i] ^ -m  = w[i] ^ (8 - m)

        if self.name == 'T' or self.name == 'w':
            # to inverse the matrix only change the power
            self.power = 8 - (self.power % 8)

        # If name is X or H, the inverse of matrix equals the matrix itself and
        # change nothing

    def to_latex(self):

        """
            This method makes a latex expression(using sympy library)
            to show the matrix perfectly
        """

        if self.name == 'w':
            look = '\omega_' + '{[' + str(self.idx[0] + 1) + ']}'
        elif self.name == 'XTX':
            look_x = 'X_[' + str(self.idx[0] + 1) + ',' + str(self.idx[1] + 1) + ']'
            x = sp.UnevaluatedExpr(sp.Symbol(look_x))
            look_t = 'T_[' + str(self.idx[0] + 1) + ',' + str(self.idx[1] + 1) + ']'
            t = sp.UnevaluatedExpr(sp.Symbol(look_t))
            return x * (t ** self.power) * x
        else:
            look = self.name + '_[' + str(self.idx[0] + 1) + ',' + str(self.idx[1] + 1) + ']'

        return sp.Symbol(look) ** self.power


    def __str__(self):

        if self.name == "w":
            return "{}[{}]^{}".format(self.name, self.idx[0], self.power)
        return "{}[{},{}]^{}".format(self.name, self.idx[0], self.idx[1], self.power)



class MidLevelComponent:

    """
        This class describes qunatum controlled gates (Cn(U))
        For example a 3 controlled H gate (CCCH) or toffoli gate(CCNOT)
        (Control qubits can be negated)

        An object of this class represents a quantum controlled gate

        Args:
            name (str): name of gate, for example 'H' or 'T'
            q_array (array): this array contains qubits type
            count (int): count of gate

        Attributes:
            name (str): to store name of gate
            q_array (array): to store q_array
            count (int): to store count of gate (count > 0)
            i (int): indicates target qubit

        For example let CG = MidLevelComponent('T', [0,-1,1], 3)
        then CG represents following gate:

                     negated contorl(0)   ---------o--------
                                                   |
                                                   |
                                                   |
                                                -------
                             target(-1)   ------| T^3 |-----
                                                -------
                                                   |
                                                   |
                                                   |
                      normal control(1)   ---------*--------


        or let CG1 = MidLevelComponent('H', [0,0,1,-1], 1)
        then CG1 represents following gate:

                     negated contorl(0)   ---------o--------
                                                   |
                                                   |
                                                   |
                     negated contorl(0)   ---------o--------
                                                   |
                                                   |
                                                   |
                      normal contorl(1)   ---------*--------
                                                   |
                                                   |
                                                   |
                                                 -----
                             target(-1)    ------| H |-----
                                                 -----

    """

    # Contructor
    def __init__(self, name, q_array, count):

        # Initializing the attributes
        self.name = name
        self.q_array = np.array(q_array)
        self.i = np.where(self.q_array == -1)[0][0] # Find target qubit
        self.count = count



class LowLevelComponent:

    """
        This class describes Clifford+T gates in circuits
        Cliffor+T gates are H,X,T, T^-1, S, S^-1, CNOT

        An object of this class represents a Clifford+T gate

        Args:
            name (str): name of gate (for example 'T' or 'H')
            idx (array of int): indicate to the qubits where gate acts on

        Attributes:
            name (str): to store name of gate
            idx (int): to store idx
                       ( for CNOT gate, idx contains 2 number,
                         idx[0] indicates control qubit,
                         idx[1] indicates target qubit.
                         for other gates, idx contains only 1 number )

        For example let G = LowLevelComponent('H',[1])
        then G represents a H gate actting on qubit 1
        following shows the gate in circuit:


                     qubit 0     -----------------------
                                          -----
                     qubit 1     ---------| H |---------
                                          -----
                     qubit 2     -----------------------
                        .                   .
                        .                   .
                        .                   .

                     qubit n-1   -----------------------


        For example let G1 = LowLevelComponent('CNOT',[2,0])
        then G1 represents a CNOT gate actting on qubit 2 and 0:
        following shows the gate in circuit:

                                           -----
                     qubit 0     ----------| X |-----------
                                           -----
                                             |
                                             |
                                             |
                     qubit 1      -----------|-----------
                                             |
                     qubit 2      -----------*-----------

                     qubit 3      ------------------------
                        .                    .
                        .                    .
                        .                    .

                     qubit n-1    -----------------------

    """

    # Constructor
    def __init__(self, name, idx):

        # Initializing the attributes
        self.name = name.upper()
        self.idx = idx


    def inverse(self):

        """ This method returns the inverse of gate """

        if self.name == 'T':
            self.name = 'TDG'
        elif self.name == 'TDG':
            self.name = 'T'
        elif self.name == 'S':
            self.name = 'SDG'
        elif self.name == 'SDG':
            self.name = 'S'

        # If name is H or CNOT or X, the inverse of gate is the gate itself
        # ( H ^ -1 = H, X ^ -1 = X, CNOT ^ -1 = CNOT )
        # so in these cases, there is nothing to change

        return self

    def to_matrix(self, nq=1):

        """
            This method returns the unitary matrix of gate in an nq-qubit
            circuit (Returns a unitary matrix with size 2 ^ nq * 2 ^ nq )
        """

        if nq == 1:
            # Returns a 2*2 unitary matrix
            return gates[self.name]

        if (nq - 1) < max(self.idx):
            raise ValueError('err')

        if self.name == 'CX':

            # If the gate is CNOT, get matrix using qiskit library

            circ = QuantumCircuit(nq)
            circ.cx(self.idx[0], self.idx[1])
            circ1 = QuantumCircuit(nq)

            circ1.compose(circ, list(range(nq - 1,-1,-1)), inplace=True)

            back = Aer.get_backend('unitary_simulator')
            result = execute(circ1, back).result()
            unitary = result.get_unitary(circ1)
            return unitary.astype(int)


        k1 = 2 ** self.idx[0]
        k2 = 2 ** (nq - self.idx[0] - 1)
        I1 = np.matrix(np.identity(k1, dtype=int))
        I2 = np.matrix(np.identity(k2, dtype=int))

        # Calculate the matrix using tensor product
        return np.kron(np.kron(I1, gates[self.name]), I2)

