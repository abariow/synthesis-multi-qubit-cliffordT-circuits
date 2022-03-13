import math
import numpy as np
import sympy as sp
from copy import deepcopy
from DOmega import Dw
from ZOmega import Zw
from copy import deepcopy
import Components as MX
import Utils as utils

from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import AncillaRegister
from qiskit.extensions import MCXGate, CCXGate, CXGate, CHGate
from qiskit.extensions import HGate, TGate, XGate, SGate
from qiskit.circuit.library import MCMT


# Implementation of exact synthesis of multiqubit Clifford+T circuit
# arXiv:1212.0506


def rowStep(U, i, j, xi, xj, xx, col):

    """
        This function performs a single row operation (as in Lemma4 in arXiv:1212.0506)
        , applied to rows i and j of col-th column of matrix U.
        A precondition is that the entries at row i and row j are of the same
        residue type i.e norm(xi) = norm(xj)
        The function returns a list of two-level matrices that decreases the denominator exponent.

        Args:
            U (numpy matrix): 2^n * 2^n Unitary matrix of D-omega objects
            i (int): indicates row i of matrix
            j (int): indicates row j of matrix
            xi (Zw): residue of the entry at row i
            xj (Zw): residue of the entry at row j
            xx (Zw): norm of xi and xj (xx = xi.norm() = xj.norm())
            col (int): indicates a column of matrix U

    """

    N = U.shape[1]
    components = [] # a list to store 2-level matrices

    def makeComponents(U, i, j, xi, xj, components):

        m = deepcopy(xi).align(deepcopy(xj)) # shift and align xj with xi to find m
        C1 = MX.HighLevelComponent('T', m, N, i, j) # make T[i,j] ^ m
        C2 = MX.HighLevelComponent('H', 1, N, i, j) # make H[i,j]
        U[:] = C2.powered_matrix() @ C1.powered_matrix() @ U # multipy matrices
        np.vectorize(Dw.reduce)(U)
        components += [C1, C2] # add matrcies to list

    if xx.cff_str() == "0000": # case 1 (in lemma 4)
        pass

    elif xx.cff_str() == "1010": # case 2 (in lemma 4)
        makeComponents(U, i, j, xi, xj, components)

    elif xx.cff_str() == "0001": # case 3 (in lemma 4)

        if deepcopy(xi).align(deepcopy(xj)) == -1:
            makeComponents(U, i, j, xi, xj.complement(), components)
            xi = U[i, col].k_residue(U[i, col].n)
            xj = U[j, col].k_residue(U[j, col].n)

        makeComponents(U, i, j, xi, xj, components)

    return components


def reduceColumn(U, W, col):

    """
        This function finds the 2-level matrices of type X, T and H, so that
        if all multiplied by matrix U, the ith column of U would be
        converted to ei (as in Lemma5 in arXiv:1212.0506)
        In other words let U1,U2,...Uk be the 2-level matrices of type X, T and H
        and let u be ith column of matrix U, so that: (Uk * .... * U2 * U1) * u = ei,
        therefore this function first finds U1, U2, ... Uk then multipies them all
        by U:  U <- (Uk * ... U2 * U1) * U
        After the function is done, ith column of U is ei and the 2-level
        matrices are stored in a list(W = [U1, U2, ... Uk])

             _   _
            |  0  | ----> 1st component
            |  :  |
            |  0  |
       ei = |  1  | ----> ith component
            |  0  |
            |  :  |
            |_ 0 _| ----> nth component

        Args:
            U (numpy matrix): 2^n * 2^n Unitary matrix of D-omega objects
            W (list): a list to store 2-level matrices
            col (int): indicates to the column going to ei ( col = i - 1 )
    """

    N = U.shape[1]
    U[col:, col] = utils.makeDenomsCommon(U[col:, col])
    x = np.vectorize(Dw.k_residue)(U[col:, col]) # x = residue of col-th column of U
    xx = np.vectorize(Zw.residue)(np.vectorize(Zw.norm)(x)) # xx_i = x_i.norm()
    xx_str = np.vectorize(Zw.cff_str)(xx)

    u_str = np.vectorize(lambda dw: dw.zw.cff_str())(U[col:, col])
    n0rows = np.where(u_str != "0000")[0] # Indicates non-zero elements in column

    # Check if there's only 1 non-zero element left in the column
    if n0rows.size == 1:

        U[col, col].reduce()

        # Check whether the non-zero element is at first row or not
        if n0rows[0] != 0:
            C = MX.HighLevelComponent('X', 1, N, col, n0rows[0] + col) # make 2-level matrix of type X
            U[:] = C.powered_matrix() @ U # multiply by U
            W += [C] # add matrix to list

        a = U[col, col].zw.residue().align(Zw.one())
        m = 8 - a

        if (U[col, col].zw + U[col, col].zw.residue()).cff_str() == "0000":
            m += 4

        if m != 8: # if the non-zero element doesn't equal 1
            C = MX.HighLevelComponent('w', m, N, col) # make 1-level matrix of type omega
            U[:] = C.powered_matrix() @ U # multipy by U
            W += [C] # add matrix to list

        return

    # lemma 4 in arXiv:1212.0506
    for case in ["1010", "0001"]:
        idx = np.where(xx_str == case)
        for i, j in idx[0].reshape(idx[0].size // 2, 2):
            W += rowStep(U, i + col, j + col, x[i, 0], x[j, 0], xx[i, 0], col)

    # Continue recursively as long as the coulmn isn't converted to ei yet
    reduceColumn(U, W, col)


def decomposMatrix(U):

    """
        This function decomposes input matrix into two-level matrices of type
        X,T and H.(as in Lemma6 in arXiv:1212.0506)


        Args:
            U (numpy matrix): a 2^n * 2^n matrix of D-omega(Dw) objects

        Return:
            A list of HighLevelComponent objects
    """

    N = U.shape[1] # Matrix size (N * N)
    Components = [] # A list to store 2-level matrices

    # Loop through columns of matrix
    for column in range(N):
        # Convert i-th column into ei
        reduceColumn(U, Components, column)

    # After loop U is converted to I

    # Now Components contains 2-level matrices
    # let's suppose Components = [ U1, U2 , U3, ... Un ]
    # and suppose V = input matrix ( V = U before is changed )
    # then we can say (Un * Un-1 * ..... U2 * U1) * V = I
    # and we can conclude V = (U1 ^ -1) * (U2 ^ -1) * ... (Un ^ -1)

    # Loop through the components
    for c in Components:
        if c.power == 0:
            # Remove the matrices with power 0, which equal I
            Components.remove(c)
        c.TC() # Transpose conjugate the matrix (inverse it)

    # Now Components = [(U1 ^ -1), (U2 ^ -1), ... (Un ^ -1)]

    # Reverse the list and return it
    return Components[::-1]


def decompos2LMatrix(HC):

    """
        This function decomposes a two-level matrix into controlled gates,
        using the algorithm in the book "Quantum Computation and Quantum
        Information", section 4.5.2

        Args:
            HC (HighLevelComponent): 2-level matrix of type H or T or X
                                     1-level matrix of type omega

        Return:
            A list of MidLevelComponent objects : list of controlled gates that are of type:
                                                   (Multi) controlled H gate
                                                   (Multi) controlled T gate
                                                   (Multi) controlled X gate
    """

    nq = int(np.log2(HC.N))

    if HC.name == 'w': # check if the matrix is one-level

    # decompose the one-level matrix into 2-level matricies

        if HC.idx[0] == 0:
            # w[1] ->  X[1,2] * T[1,2] * X[1,2]
            HC.idx += [1]
            HC.name = 'XTX'
        else:
            # for i != 1, w[i] -> T[1,i]
            HC.idx.insert(0, 0)
            HC.name = 'T'

    i = HC.idx[0]
    j = HC.idx[1]

    bi = bin(i)[2:]
    bi = '0' * (nq - len(bi)) + bi # i in binary with size nq

    bj = bin(j)[2:]
    bj = '0' * (nq - len(bj)) + bj # j in binar with size nq

    s = np.array(list(map(int, list(bi))))
    t = np.array(list(map(int, list(bj))))

    diff = np.where(((s + t) % 2) == 1)[0]

    def makeComponents(HC, s, di, components):

        # This function creates gates

        sc = np.array(s)
        sc[di[0]] = -1

        if di.size == 1:
            if HC.name == 'XTX':
                components += [MX.MidLevelComponent('X', sc, 1)]
                components += [MX.MidLevelComponent('T', sc, HC.power)]
                components += [MX.MidLevelComponent('X', sc, 1)]
            else:
                components += [MX.MidLevelComponent(HC.name, sc, HC.power)]
            return

        s[di[0]] = (s[di[0]] + 1) % 2

        components += [MX.MidLevelComponent('X', sc, 1)]
        makeComponents(HC, s, di[1:], components)
        components += [MX.MidLevelComponent('X', sc, 1)]

    components = [] # a list to store gates

    makeComponents(HC, s, diff[::-1], components)

    return components


def decomposCH(q0, q1):

    """
        This function decomposes Controlled-H gate (CH) into Clifford+T gates

        Args:
            q0 (int): indicates to control qubit
            q1 (int): indicates to target qubit

        Returns a list of LowLevelComponent objects
    """

    CH = []
    CH += [MX.LowLevelComponent('s', [q1])]
    CH += [MX.LowLevelComponent('h', [q1])]
    CH += [MX.LowLevelComponent('t', [q1])]
    CH += [MX.LowLevelComponent('cx', [q0, q1])]
    CH += [MX.LowLevelComponent('tdg', [q1])]
    CH += [MX.LowLevelComponent('h', [q1])]
    CH += [MX.LowLevelComponent('sdg', [q1])]

    return CH


def decomposCCX(q0, q1, q2):

    """
        This function decomposes toffoli gate (CCNOT) into Clifford+T gates

        Args:
            q0 (int): indicates to first control qubit (q0 >= 0)
            q1 (int): indicates to second control qubit (q1 >= 0)
            q2 (int): indicates to target qubit (q2 >= 0)

        Returns a list of LowLevelComponent objects
    """

    CCX = []
    CCX += [MX.LowLevelComponent('h', [q2])]
    CCX += [MX.LowLevelComponent('cx', [q1, q2])]
    CCX += [MX.LowLevelComponent('tdg', [q2])]
    CCX += [MX.LowLevelComponent('cx', [q0, q2])]
    CCX += [MX.LowLevelComponent('t', [q2])]
    CCX += [MX.LowLevelComponent('cx', [q1, q2])]
    CCX += [MX.LowLevelComponent('t', [q1])]
    CCX += [MX.LowLevelComponent('tdg', [q2])]
    CCX += [MX.LowLevelComponent('cx', [q0, q2])]
    CCX += [MX.LowLevelComponent('cx', [q0, q1])]
    CCX += [MX.LowLevelComponent('t', [q2])]
    CCX += [MX.LowLevelComponent('t', [q0])]
    CCX += [MX.LowLevelComponent('tdg', [q1])]
    CCX += [MX.LowLevelComponent('h', [q2])]
    CCX += [MX.LowLevelComponent('cx', [q0, q1])]

    return CCX


def decomposMCX(nq, nctrls):

    """
        This function decomposes multi-controlled-not gate (CC...CNOT), which is in
        nq-qubit circuit, into Clifford+T gates (as in arXiv:quant-ph/9503016, Lemma 7.2)

        Args:
            nq (int): number of qubits in circuit (nq >= 2)
            nctrls (int): number of control qubits ( nctrls < nq / 2 )

        Returns a list of LowLevelComponent objects
    """

    if nq < 2:
        raise ValueError("err")

    if nctrls == 0:
        raise ValueError("err")

    if nctrls > math.ceil(nq / 2):
        raise ValueError("err")

    if nctrls == 1:
        return [MX.LowLevelComponent('cx', [0, nq - 1])]

    if nctrls == 2:
        return decomposCCX(0, 1, nq - 1)

    def _MCX_(nq, nctrls):

        if nctrls == 2:
            return decomposCCX(0, 1, nq - 1)

        _CCX_ = decomposCCX(nctrls - 1, nq - 2, nq - 1)
        return _CCX_ + _MCX_(nq - 1, nctrls - 1) + _CCX_


    return _MCX_(nq, nctrls) + _MCX_(nq - 1, nctrls - 1)


def rearrangeQ(LLCs, cqp, nqp):
    # rearrange gates in a different order of qubits
    d = dict(zip(cqp, nqp))
    RQ = lambda llc : MX.LowLevelComponent(llc.name, [d[i] for i in llc.idx])
    return list(map(RQ, LLCs))

def inverse(LLCs):
    return [llc.inverse() for llc in LLCs[::-1]]

def decomposMCiX(nq):

    """
        This function decomposes multi-controlled-iX gate (CC...CiX) into Clifford+T gates.
        (as in arXiv:1212.0506, section 5.2)

        ( iX = sqrt(-1) * X )

        Args:
            nq (int): number of qubits in circuit (nq >= 2)

        Returns a list of LowLevelComponent objects
    """

    if nq == 1:
        raise ValueError("err")

    if nq == 2:
        cix = []
        cix += [MX.LowLevelComponent('s', [0])]
        cix += [MX.LowLevelComponent('cx', [0, 1])]
        return cix

    nc = nq - 1
    nc1 = nc // 2
    nc2 = nc - nc1

    cqp = list(range(nq))
    nqp = list(range(nc2, nq - 1)) +  list(range(nc2)) + [nq - 1]

    mcix = []
    mcix += [MX.LowLevelComponent('h', [nq - 1])]
    mcix += [MX.LowLevelComponent('tdg', [nq - 1])]
    mcix += rearrangeQ(decomposMCX(nq, nc1), cqp, nqp)
    mcix += [MX.LowLevelComponent('t', [nq - 1])]
    mcix += decomposMCX(nq, nc2)
    mcix += [MX.LowLevelComponent('tdg', [nq - 1])]
    mcix += rearrangeQ(decomposMCX(nq, nc1), cqp, nqp)
    mcix += [MX.LowLevelComponent('t', [nq - 1])]
    mcix += decomposMCX(nq, nc2)
    mcix += [MX.LowLevelComponent('h', [nq - 1])]

    return mcix


def decomposMCH(nq):

    """
        This function decomposes multi-controlled-H gate (CC...CH) into Clifford+T gates.
        (as in arXiv:1212.0506, section 5.2)

        Args:
            nq (int): number of qubits in circuit (nq >= 2)

        Returns a list of LowLevelComponent objects
    """

    if nq <= 1:
        raise ValueError("err")

    nc = nq - 1
    nq = nq + 1 # consider ancilla qubit

    if nq == 3:
        return decomposCH(1,2)

    cqp = list(range(nq))
    nqp = list(range(nq - 1))[::-1] + [nq - 1]

    mch = []
    mch += rearrangeQ(decomposMCiX(nc + 1), cqp, nqp)
    mch += decomposCH(0, nq - 1)
    mch += rearrangeQ(inverse(decomposMCiX(nc + 1)), cqp, nqp)

    return mch


def decomposMCT(nq, k=1):

    """
        This function decomposes multi-controlled-T gate (CC...CT) into Clifford+T gates.
        (as in arXiv:1212.0506, section 5.2)

        Args:
            nq (int): number of qubits in circuit (nq >= 2)
            k (int): count of T gate (T ^ k)

        Returns a list of LowLevelComponent objects
    """

    if nq <= 1:
        raise ValueError("err")

    k = k % 8
    if k == 0:
        raise ValueError("err")

    nc = nq - 1
    nq = nq + 1 # consider ancilla qubit

    cqp = list(range(nq))
    nqp = list(range(nq))[::-1]

    mct = []
    mct += rearrangeQ(decomposMCiX(nq), cqp, nqp)

    if k == 1:
        mct += [MX.LowLevelComponent('t', [0])]
    elif k == 2:
        mct += [MX.LowLevelComponent('s', [0])]
    elif k == 3:
        mct += [MX.LowLevelComponent('t', [0])]
        mct += [MX.LowLevelComponent('s', [0])]
    elif k == 4:
        mct += [MX.LowLevelComponent('s', [0])]
        mct += [MX.LowLevelComponent('s', [0])]
    elif k == 5:
        mct += [MX.LowLevelComponent('sdg', [0])]
        mct += [MX.LowLevelComponent('tdg', [0])]
    elif k == 6:
        mct += [MX.LowLevelComponent('sdg', [0])]
    elif k == 7:
        mct += [MX.LowLevelComponent('tdg', [0])]

    mct += rearrangeQ(inverse(decomposMCiX(nq)), cqp, nqp)

    return mct


def decomposMCXp(nq):

    """
        This function decomposes multi-controlled-X gate (CC...CX) into Clifford+T gates.
        (as in arXiv:1212.0506, section 5.2)

        Args:
            nq (int): number of qubits in circuit (nq >= 2)

        Returns a list of LowLevelComponent objects
    """

    if nq <= 1:
        raise ValueError("err")

    nc = nq - 1
    nq = nq + 1 # consider ancilla qubit

    if nq == 3:
        return [MX.LowLevelComponent('cx', [1,2])]

    if nq == 4:
        return decomposCCX(1, 2, 3)

    cqp = list(range(nq))
    nqp = list(range(nq - 1))[::-1] + [nq - 1]

    mcx = []
    mcx += rearrangeQ(decomposMCiX(nc + 1), cqp, nqp)
    mcx += [MX.LowLevelComponent('cx', [0, nq - 1])]
    mcx += rearrangeQ(inverse(decomposMCiX(nc + 1)), cqp, nqp)

    return mcx


def decomposMCGate(mlc):

    """
        This function decomposes a (multi)controlled gate into clifford+t gates.

        Args:
            mlc (MidLevelComponent): a (multi)controlled gate of type H or X or T

        Returns a list of LowLevelComponent objects
    """

    nq = mlc.q_array.size

    cqp = list(range(nq + 1))
    nqp = list(range(nq + 1))

    neg_ctrls = [i + 1 for i in list(np.where(mlc.q_array == 0)[0])]
    result = []

    for i in neg_ctrls:
        result += [MX.LowLevelComponent('x', [i])]

    if mlc.name == 'H':
        nqp.append(nqp.pop(mlc.i + 1))
        result += rearrangeQ(decomposMCH(nq), cqp, nqp)

    elif mlc.name == 'T':
        result += decomposMCT(nq, mlc.count)

    elif mlc.name == 'X':
        nqp.append(nqp.pop(mlc.i + 1))
        result += rearrangeQ(decomposMCXp(nq), cqp, nqp)

    for i in neg_ctrls:
        result += [MX.LowLevelComponent('x', [i])]

    return result


def decompos(U):

    """
        This function decomposes a unitary matrix of D-omega numbers into clifford+t gates.

        Args:
            U (numpy.matrix): 2^n * 2^n Unitary matrix of D-omega objects

        Returns:
            LLCs (list of LowLevelComponent objects) : containing Clifford+T gates that matrix U decomposed into
            MLCs (list of LowLevelComponent objects) : containing (multi)controlled gates
            HLCs (list of LowLevelComponent objects) : containing 2-level matrices
            HLCs1 (list of LowLevelComponent objects) : containing 2-level and 1-level matrices
    """

    N = U.shape[1]
    nq = int(np.log2(N))

    if U.shape[0] != U.shape[1] or 2 ** nq !=  N:
        raise ValueError("Invalid matrix size")

    if not utils.checkUnitarity(U):
        raise ValueError("Input matrix is not unitary")


    # decompose matrix U into 2-level matriceis of type X,H,T
    HLCs = decomposMatrix(U)
    HLCs1 = deepcopy(HLCs)

    # decompose 2-level matrices into controlled gates of type MCT,MCH,MCX
    MLCs = sum(list(map(decompos2LMatrix, HLCs)), [])

    # decompose controlled gates into Clifford+T gates
    LLCs = sum(list(map(decomposMCGate, MLCs)), [])

    return LLCs, MLCs, HLCs, HLCs1

def compose_ll(LLCs, nq):

    """
        This function puts the clifford+T gates together in a nq-qubit quantum circuit

        Args:
            LLCs (list of LowLevelComponent objects): Clifford+T gates
            nq (int): number of qubits in circuit

        Returns:
            circ (Qiskit.QuantumCircuit) : a quantum circuit made of only clifford+T gates
    """

    qr = QuantumRegister(nq - 1, 'q')
    anc = QuantumRegister(1, 'ancilla')
    circ = QuantumCircuit(anc, qr)

    for llc in LLCs: # loop through gates and add them to circuit
        getattr(circ, llc.name.lower())(*llc.idx)

    return circ


def compose_ml(MLCs, nq):

    """
        This function puts the (multi)controlled gates together in a nq-qubit circuit

        Args:
            MLCs (list of MidLevelComponent objects): (multi)controlled gates with the same number of controls
            nq (int): number of qubits in circuit

        Returns:
            circ (Qiskit.QuantumCircuit) : a quantum circuit made of only controlled gates
    """

    if nq < 2:
        raise ValueError("err")

    circ = QuantumCircuit(nq)

    if not MLCs:
        circ

    for MLC in MLCs:

        p = list(range(nq))
        p.append(p.pop(MLC.i))

        ctrl_state = ''.join(list(map(lambda x : str(x), np.delete(MLC.q_array, MLC.i)))[::-1])

        if MLC.name == 'T':
            MLC.count = MLC.count % 8

            if MLC.count == 1:
                mct = TGate().control(nq - 1, ctrl_state=ctrl_state)
                circ.compose(mct, p, inplace=True)

            if MLC.count == 2:
                mcs = SGate().control(nq - 1, ctrl_state=ctrl_state)
                circ.compose(mcs, p, inplace=True)

            if MLC.count == 3:
                mct = TGate().control(nq - 1, ctrl_state=ctrl_state)
                mcs = SGate().control(nq - 1, ctrl_state=ctrl_state)
                circ.compose(mct, p, inplace=True)
                circ.compose(mcs, p, inplace=True)

            if MLC.count == 4:
                mcs = SGate().control(nq - 1, ctrl_state=ctrl_state)
                circ.compose(mcs, p, inplace=True)
                circ.compose(mcs, p, inplace=True)

            if MLC.count == 5:
                mcsdg = SGate().inverse().control(nq - 1, ctrl_state=ctrl_state)
                mctdg = TGate().inverse().control(nq - 1, ctrl_state=ctrl_state)
                circ.compose(mcsdg, p, inplace=True)
                circ.compose(mctdg, p, inplace=True)

            if MLC.count == 6:
                mcsdg = SGate().inverse().control(nq - 1, ctrl_state=ctrl_state)
                circ.compose(mcsdg, p, inplace=True)

            if MLC.count == 7:
                mctdg = TGate().inverse().control(nq - 1, ctrl_state=ctrl_state)
                circ.compose(mctdg, p, inplace=True)


        if MLC.name == 'X':
            mcx = XGate().control(nq - 1, ctrl_state=ctrl_state)
            circ.compose(mcx, p, inplace=True)

        if MLC.name == 'H':
            mch = HGate().control(nq - 1, ctrl_state=ctrl_state)
            circ.compose(mch, p, inplace=True)

    return circ

def compose_hl(HLCs):

    """
        This function put 2-level matrices together in latex style

        Args:
            HLCs (list of HighLevelComponent): 2-level matrices
    """

    return np.prod([sp.UnevaluatedExpr(HLC.to_latex()) for HLC in HLCs[::-1]])


def syntCliffordTCircuit(U):

    """
        This function runs the synthesis algorithm introduced in arXiv:1212.0506

        Args:
            U (numpy.matrix): 2^n * 2^n Unitary matrix of D-omega objects

        Returns:
            **main result**
            cliffordTCircuit (Qiskit.QuantumCircuit) : the synthesised circuit
                                                       (quantum circuit made of only Clifford+T gates)

            mcgCircuit (Qiskit.QuantumCircuit): a higher level circuit of U, made of (multi) controlled gates
    """

    N = U.shape[1]
    nq = int(np.log2(N))

    # decompose the matrix into clifford+T gates
    LLCs, MLCs, HLCs, HLCs1 = decompos(U)

    cliffordTCiruit = compose_ll(LLCs, nq + 1) # make clifford+T circuit
    mcgCircuit = compose_ml(MLCs, nq) # make higher level circuit

    return cliffordTCiruit, mcgCircuit, compose_hl(HLCs), compose_hl(HLCs1)

############################################################
############################################################

