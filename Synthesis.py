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


def rowStep(U, i, j, xi, xj, xx, col):

    N = U.shape[1]
    components = []

    def makeComponents(U, i, j, xi, xj, components):

        m = deepcopy(xi).align(deepcopy(xj))
        C1 = MX.HighLevelComponent('T', m, N, i, j)
        C2 = MX.HighLevelComponent('H', 1, N, i, j)
        U[:] = C2.powered_matrix() @ C1.powered_matrix() @ U
        np.vectorize(Dw.reduce)(U)
        components += [C1, C2]

    if xx.cff_str() == "0000":
        pass

    elif xx.cff_str() == "1010":
        makeComponents(U, i, j, xi, xj, components)

    elif xx.cff_str() == "0001":

        if deepcopy(xi).align(deepcopy(xj)) == -1:
            makeComponents(U, i, j, xi, xj.complement(), components)
            xi = U[i, col].k_residue(U[i, col].n)
            xj = U[j, col].k_residue(U[j, col].n)

        makeComponents(U, i, j, xi, xj, components)

    return components


def reduceColumn(U, W, col):

    N = U.shape[1]
    U[col:, col] = utils.makeDenomsCommon(U[col:, col])
    x = np.vectorize(Dw.k_residue)(U[col:, col])
    xx = np.vectorize(Zw.residue)(np.vectorize(Zw.norm)(x))
    xx_str = np.vectorize(Zw.cff_str)(xx)

    u_str = np.vectorize(lambda dw: dw.zw.cff_str())(U[col:, col])
    n0rows = np.where(u_str != "0000")[0]

    if n0rows.size == 1:

        U[col, col].reduce()

        if n0rows[0] != 0:
            C = MX.HighLevelComponent('X', 1, N, col, n0rows[0] + col)
            U[:] = C.powered_matrix() @ U
            W += [C]

        a = U[col, col].zw.residue().align(Zw.one())
        m = 8 - a

        if (U[col, col].zw + U[col, col].zw.residue()).cff_str() == "0000":
            m += 4

        if m != 8:
            C = MX.HighLevelComponent('w', m, N, col)
            U[:] = C.powered_matrix() @ U
            W += [C]

        return

    for case in ["1010", "0001"]:
        idx = np.where(xx_str == case)
        for i, j in idx[0].reshape(idx[0].size // 2, 2):
            W += rowStep(U, i + col, j + col, x[i, 0], x[j, 0], xx[i, 0], col)

    reduceColumn(U, W, col)


def decomposMatrix(U): # decompMatrix

    N = U.shape[1]
    Components = []

    for column in range(N):
        reduceColumn(U, Components, column)

    for c in Components:
        if c.power == 0:
            Components.remove(c)
        c.TC()

    return Components[::-1]


def decompos2LMatrix(HC): # twoLevelMatrixToCircuit

    nq = int(np.log2(HC.N))

    if HC.name == 'w':
        if HC.idx[0] == 0:
            HC.idx += [1]
            HC.name = 'XTX'
        else:
            HC.idx.insert(0, 0)
            HC.name = 'T'

    i = HC.idx[0]
    j = HC.idx[1]

    bi = bin(i)[2:]
    bi = '0' * (nq - len(bi)) + bi

    bj = bin(j)[2:]
    bj = '0' * (nq - len(bj)) + bj

    s = np.array(list(map(int, list(bi))))
    t = np.array(list(map(int, list(bj))))

    diff = np.where(((s + t) % 2) == 1)[0]

    def makeComponents(HC, s, di, components):

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

    components = []

    makeComponents(HC, s, diff[::-1], components)

    return components


def CH(q0, q1):

    CH = []
    CH += [MX.LowLevelComponent('s', [q1])]
    CH += [MX.LowLevelComponent('h', [q1])]
    CH += [MX.LowLevelComponent('t', [q1])]
    CH += [MX.LowLevelComponent('cx', [q0, q1])]
    CH += [MX.LowLevelComponent('tdg', [q1])]
    CH += [MX.LowLevelComponent('h', [q1])]
    CH += [MX.LowLevelComponent('sdg', [q1])]

    return CH


def CCX(q0, q1, q2):

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


def MCX(nq, nctrls):

    if nq < 2:
        raise ValueError("err")

    if nctrls == 0:
        raise ValueError("err")

    if nctrls > math.ceil(nq / 2):
        raise ValueError("err")

    if nctrls == 1:
        return [MX.LowLevelComponent('cx', [0, nq - 1])]

    if nctrls == 2:
        return CCX(0, 1, nq - 1)

    def _MCX_(nq, nctrls):

        if nctrls == 2:
            return CCX(0, 1, nq - 1)

        _CCX_ = CCX(nctrls - 1, nq - 2, nq - 1)
        return _CCX_ + _MCX_(nq - 1, nctrls - 1) + _CCX_


    return _MCX_(nq, nctrls) + _MCX_(nq - 1, nctrls - 1)


def rearrangeQ(LLCs, cqp, nqp):

    d = dict(zip(cqp, nqp))
    RQ = lambda llc : MX.LowLevelComponent(llc.name, [d[i] for i in llc.idx])
    return list(map(RQ, LLCs))

def inverse(LLCs):
    return [llc.inverse() for llc in LLCs[::-1]]

def MCiX(nq):

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
    mcix += rearrangeQ(MCX(nq, nc1), cqp, nqp)
    mcix += [MX.LowLevelComponent('t', [nq - 1])]
    mcix += MCX(nq, nc2)
    mcix += [MX.LowLevelComponent('tdg', [nq - 1])]
    mcix += rearrangeQ(MCX(nq, nc1), cqp, nqp)
    mcix += [MX.LowLevelComponent('t', [nq - 1])]
    mcix += MCX(nq, nc2)
    mcix += [MX.LowLevelComponent('h', [nq - 1])]

    return mcix


def MCH(nq):

    if nq <= 1:
        raise ValueError("err")

    nc = nq - 1
    nq = nq + 1

    if nq == 3:
        return CH(1,2)

    cqp = list(range(nq))
    nqp = list(range(nq - 1))[::-1] + [nq - 1]

    mch = []
    mch += rearrangeQ(MCiX(nc + 1), cqp, nqp)
    mch += CH(0, nq - 1)
    mch += rearrangeQ(inverse(MCiX(nc + 1)), cqp, nqp)

    return mch


def MCT(nq, k=1):

    if nq <= 1:
        raise ValueError("err")

    k = k % 8
    if k == 0:
        raise ValueError("err")

    nc = nq - 1
    nq = nq + 1

    cqp = list(range(nq))
    nqp = list(range(nq))[::-1]

    mct = []
    mct += rearrangeQ(MCiX(nq), cqp, nqp)

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

    mct += rearrangeQ(inverse(MCiX(nq)), cqp, nqp)

    return mct


def MCXp(nq):

    if nq <= 1:
        raise ValueError("err")

    nc = nq - 1
    nq = nq + 1

    if nq == 3:
        return [MX.LowLevelComponent('cx', [1,2])]

    if nq == 4:
        return CCX(1, 2, 3)

    cqp = list(range(nq))
    nqp = list(range(nq - 1))[::-1] + [nq - 1]

    mcx = []
    mcx += rearrangeQ(MCiX(nc + 1), cqp, nqp)
    mcx += [MX.LowLevelComponent('cx', [0, nq - 1])]
    mcx += rearrangeQ(inverse(MCiX(nc + 1)), cqp, nqp)

    return mcx


def decomposMCGate(mlc):

    nq = mlc.q_array.size

    cqp = list(range(nq + 1))
    nqp = list(range(nq + 1))

    neg_ctrls = [i + 1 for i in list(np.where(mlc.q_array == 0)[0])]
    result = []

    for i in neg_ctrls:
        result += [MX.LowLevelComponent('x', [i])]

    if mlc.name == 'H':
        nqp.append(nqp.pop(mlc.i + 1))
        result += rearrangeQ(MCH(nq), cqp, nqp)

    elif mlc.name == 'T':
        result += MCT(nq, mlc.count)

    elif mlc.name == 'X':
        nqp.append(nqp.pop(mlc.i + 1))
        result += rearrangeQ(MCXp(nq), cqp, nqp)

    for i in neg_ctrls:
        result += [MX.LowLevelComponent('x', [i])]

    return result


def decompos(U):

    N = U.shape[1]
    nq = int(np.log2(N))

    if U.shape[0] != U.shape[1] or 2 ** nq !=  N:
        raise ValueError("Invalid matrix size")

    if not utils.checkUnitarity(U):
        raise ValueError("Input matrix is not unitary")

    HLCs = decomposMatrix(U)
    HLCs1 = deepcopy(HLCs)
    MLCs = sum(list(map(decompos2LMatrix, HLCs)), [])
    LLCs = sum(list(map(decomposMCGate, MLCs)), [])

    return LLCs, MLCs, HLCs, HLCs1

def compose_ll(LLCs, nq):

    qr = QuantumRegister(nq - 1, 'q')
    anc = QuantumRegister(1, 'ancilla')
    circ = QuantumCircuit(anc, qr)
    for llc in LLCs:
        getattr(circ, llc.name.lower())(*llc.idx)
    return circ

def compose_ml(MLCs):

    nq = MLCs[0].q_array.size
    if nq < 2:
        raise ValueError("err")

    circ = QuantumCircuit(nq)

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
    return np.prod([sp.UnevaluatedExpr(HLC.to_latex()) for HLC in HLCs[::-1]])


def syntCliffordTCircuit(U):

    N = U.shape[1]
    nq = int(np.log2(N))
    LLCs, MLCs, HLCs, HLCs1 = decompos(U)
    cliffordTCiruit = compose_ll(LLCs, nq + 1)
    mcgCircuit = compose_ml(MLCs)

    return cliffordTCiruit, mcgCircuit, compose_hl(HLCs), compose_hl(HLCs1)

############################################################
############################################################

