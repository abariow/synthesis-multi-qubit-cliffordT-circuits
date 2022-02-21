import numpy as np
import sympy as sp

from DOmega import Dw
from ZOmega import Zw
from Gates import gates

from qiskit import QuantumCircuit
from qiskit import Aer,execute



def getIdentityMatrix(n):

    I = np.full((n, n), Dw(Zw()), dtype=object)
    for i in range(n):
        I[i,i] = Dw(Zw.one())
    return np.matrix(I)

def makeTLMatrix(A, n, i, j):

    TLM = getIdentityMatrix(n)
    TLM[i,i] = A[0,0]
    TLM[i,j] = A[0,1]
    TLM[j,i] = A[1,0]
    TLM[j,j] = A[1,1]

    return TLM

def makeOLMatrix(a, n, i):

    OLM = getIdentityMatrix(n)
    OLM[i,i] = a

    return OLM

class HighLevelComponent:

    def __init__(self, name, power, n, i, j=0):

        self.name = name
        self.power = power
        self.N = n

        gate = gates[name]

        if name == 'w':
            self.matrix = makeOLMatrix(gate, n, i)
            self.idx = [i]

        else:
            self.matrix = makeTLMatrix(gate, n, i, j)
            self.idx = [i,j]

    def powered_matrix(self):
        return self.matrix ** self.power

    def TC(self):
        if self.name == 'T' or self.name == 'w':
            self.power = 8 - (self.power % 8)

    def to_latex(self):
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

    def __init__(self, name, q_array, count):

        self.name = name
        self.q_array = np.array(q_array)
        self.i = np.where(self.q_array == -1)[0][0]
        self.count = count

class LowLevelComponent:

    def __init__(self, name, idx):

        self.name = name.upper()
        self.idx = idx

    def __str__(self):
        pass

    def inverse(self):

        if self.name == 'T':
            self.name = 'TDG'
        elif self.name == 'TDG':
            self.name = 'T'
        elif self.name == 'S':
            self.name = 'SDG'
        elif self.name == 'SDG':
            self.name = 'S'

        return self

    def to_matrix(self, nq=1):

        print(self.name)
        print(self.idx)

        if nq == 1:
            return gates[self.name]

        if (nq - 1) < max(self.idx):
            raise ValueError('err')

        if self.name == 'CX':
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

        return np.kron(np.kron(I1, gates[self.name]), I2)


if __name__ == "__main__":

    LLC = LowLevelComponent('cx', [0,1])
    m = LLC.to_matrix(4)
    print(LLC.name)
    print(LLC.idx)
    for i in range(16):
        for j in range(16):
            print(m[i,j])
