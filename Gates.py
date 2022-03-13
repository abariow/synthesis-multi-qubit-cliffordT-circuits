import numpy as np

from DOmega import Dw
from ZOmega import Zw

# A dictionary to store clifford+T gates
# It maps gates name to unitary matrices
gates = {}

# Hadamard Gate
gates['H'] = np.matrix([[Dw(Zw.one(),1), Dw(Zw.one(),1)      ],
                        [Dw(Zw.one(),1), Dw(-1 * Zw.one(),1)]])

# T Gate
gates['T'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                        [Dw(Zw(),0),     Dw(Zw.omega(),0)]])
# X Gate
gates['X'] = np.matrix([[Dw(Zw(),0)    , Dw(Zw.one(),0)],
                        [Dw(Zw.one(),0), Dw(Zw(),0)   ]])
# S Gate
gates['S'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                        [Dw(Zw(),0),     Dw(Zw(0,1,0,0),0)]])
# Inverse of T Gate
gates['TDG'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                          [Dw(Zw(),0),     Dw(Zw(-1,0,0,0),0)]])
# Inverse of S Gate
gates['SDG'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                          [Dw(Zw(),0),     Dw(Zw(0,-1,0,0),0)]])

# Omega(= (1 + i) / root(2)), It's actually a scalar
# (Omega ^ 2 = i)
gates['w'] = Dw(Zw.omega(), 0)

