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

    shape = U.shape
    u = U.A.flatten()
    max_n = np.max([dw.n for dw in u])
    u = np.array(list(map(lambda dw: dw * Zw.root2_power(max_n - dw.n), u)))

    for dw in u:
        dw.n = max_n

    return np.matrix(u.reshape(shape))

def checkUnitarity(U):

    N = U.shape[0]
    I = MX.getIdentityMatrix(N)
    Uct = (np.vectorize(Dw.conjug)(U)).T

    return ((U @ Uct) == I).all()

def matrix_to_latex(U):
   UL = np.vectorize(Dw.to_latex)(U)
   return sp.Matrix(UL)


def generateRandomU(nq, nc=0):

    if nc == 0:
        nc = random.randint(1, 100)

    if nq < 2:
        raise ValueError('error')

    N = 2 ** nq

    RU = MX.getIdentityMatrix(N)

    for c in range(nc):
        ij = random.sample(range(N), 2)
        ij.sort()
        i, j = ij
        gate = random.choice(list(gates.keys()))
        power = random.randint(0, 7)

        if gate == 'T' or gate == 'H':
            HLC1 = MX.HighLevelComponent('H', 1, N, i, j)
            HLC2 = MX.HighLevelComponent('T', power, N, i, j)
            RU = HLC2.powered_matrix() @ HLC1.powered_matrix() @ RU

        elif gate == 'w':
            HLC = MX.HighLevelComponent(gate, power, N, i, j)
            RU = HLC.powered_matrix() @ RU

        elif gate == 'X':
            HLC = MX.HighLevelComponent(gate, 1, N, i, j)
            RU = HLC.powered_matrix() @ RU

    return RU


def assess(U, circ):

    N = U.shape[0]
    nq = int(np.log2(N))

    if circ.num_qubits != nq:
        nq +=  1

    circ1 = QuantumCircuit(nq)
    circ1.compose(circ, list(range(nq - 1,-1,-1)), inplace=True)

    back = Aer.get_backend('unitary_simulator')
    result = execute(circ1, back).result()
    CU = result.get_unitary(circ1)[:N,:N]
    roundC = lambda C : round(C.real,10) + round(C.imag,10) * 1j
    U = np.vectorize(Dw.num)(U)
    U = np.vectorize(roundC)(U)
    CU = np.vectorize(roundC)(CU)

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

