import numpy as np
import sympy as sp
import random
from Gates import gates
from DOmega import Dw
from ZOmega import Zw
import Components as MX

from qiskit import QuantumCircuit
from qiskit import Aer,execute
from qiskit_textbook.tools import array_to_latex


def makeDenomsCommon(U):

    """
        This function makes the denominators of the entries in U, which are
        Dw-numbers, the same.

        Args:
            U (numpy matrix): 2^n * 2^n Unitary matrix of D-omega objects

        Returns a numpy.matrix object
    """

    shape = U.shape
    u = U.A.flatten()
    max_n = np.max([dw.n for dw in u])
    u = np.array(list(map(lambda dw: dw * Zw.root2_power(max_n - dw.n), u)))

    for dw in u:
        dw.n = max_n

    return np.matrix(u.reshape(shape))

def checkUnitarity(U):

    """
        This function checks whether matrix U is unitary or not.

        Args:
            U (numpy matrix): 2^n * 2^n Unitary matrix of D-omega objects

        Returns True if U is unitary.
    """

    N = U.shape[0]
    I = MX.getIdentityMatrix(N)
    Uct = (np.vectorize(Dw.conjug)(U)).T

    return ((U @ Uct) == I).all()

def matrix_to_latex(U):
   UL = np.vectorize(Dw.to_latex)(U)
   return sp.Matrix(UL)


def generateRandomU(nq, nc=0):

    """
        This function randomly generates a 2^n * 2^n unitary matrix of D-omega objects
        To generate the matrix, the function first randomly generates some 2-level
        matrices of type H,X,T then multiply them togther. ُThe more 2-level
        matrices is generated, the more complex the entries of matrix are.

        Args:
            nq (int): to determine size of matrix (2^nq * 2^nq)
            nc (int): number of 2-level matrices generated (to determine complexity of entries in matrix)

        Returns:
            U (numpy matrix): 2^n * 2^n Unitary matrix of D-omega objects
    """

    if nc == 0:
        nc = random.randint(1, 100)

    if nq < 2:
        raise ValueError('error')

    N = 2 ** nq

    RU = MX.getIdentityMatrix(N)

    # Generate nc random 2-level matrices and mutilpy them all togehter
    for c in range(nc):
        ij = random.sample(range(N), 2)
        ij.sort()
        i, j = ij
        gate = random.choice(list(gates.keys()))
        e = random.randint(0, 7)

        if gate == 'T' or gate == 'H':
            # Generate a random (T[i,j] ^ k) * H[i,j]
            HLC1 = MX.HighLevelComponent('H', 1, N, i, j)
            HLC2 = MX.HighLevelComponent('T', e, N, i, j)
            RU = HLC2.powered_matrix() @ HLC1.powered_matrix() @ RU

        elif gate == 'w':
            # Generate random 1-level marix of type omega (w[j] ^ k)
            HLC = MX.HighLevelComponent(gate, e, N, i, j)
            RU = HLC.powered_matrix() @ RU

        elif gate == 'X':
            # Generate random 2-level marix of type X
            HLC = MX.HighLevelComponent(gate, 1, N, i, j)
            RU = HLC.powered_matrix() @ RU

    return RU


def assess(U, circ):

    """
        This function checks if the circuit is synthesized correctly and
        represents exactly the unitary operator(U)

        Args:
            U (numpy matrix): 2^n * 2^n Unitary matrix of D-omega objects (the input)
            circ (qiskit.QunatumCircuit): the synthesized circuit (the output)

        Returns True if the circuit is synthesized correctly
    """

    N = U.shape[0]
    nq = int(np.log2(N))

    if circ.num_qubits != nq:
        nq +=  1

    circ1 = QuantumCircuit(nq)
    circ1.compose(circ, list(range(nq - 1,-1,-1)), inplace=True)

    back = Aer.get_backend('unitary_simulator')
    result = execute(circ1, back).result()
    CU = result.get_unitary(circ1)[:N,:N] # get unitary matrix of synthesized circuit

    roundC = lambda C : round(C.real,10) + round(C.imag,10) * 1j
    U = np.vectorize(Dw.num)(U)
    U = np.vectorize(roundC)(U)
    CU = np.vectorize(roundC)(CU)

    # compare U to unitary matrix of synthesized circuit
    return (U == CU).all(),U,CU

def assess1(U, components):

    N = U.shape[0]
    nq = int(np.log2(N)) + 1
    RU = np.identity(2 ** nq, dtype=int)

    for c in components:
        RU = c.to_matrix(nq) @ RU

    U = makeDenomsCommon(U)
    RU = makeDenomsCommon(RU)[:N,:N]
    return (U == RU).all(), U, RU

