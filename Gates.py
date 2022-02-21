import numpy as np

from DOmega import Dw
from ZOmega import Zw


gates = {}
gates['H'] = np.matrix([[Dw(Zw.one(),1), Dw(Zw.one(),1)      ],
                        [Dw(Zw.one(),1), Dw(-1 * Zw.one(),1)]])

gates['T'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                        [Dw(Zw(),0),     Dw(Zw.omega(),0)]])

gates['X'] = np.matrix([[Dw(Zw(),0)    , Dw(Zw.one(),0)],
                        [Dw(Zw.one(),0), Dw(Zw(),0)   ]])

gates['S'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                        [Dw(Zw(),0),     Dw(Zw(0,1,0,0),0)]])

gates['TDG'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                          [Dw(Zw(),0),     Dw(Zw(-1,0,0,0),0)]])

gates['SDG'] = np.matrix([[Dw(Zw.one(),0), Dw(Zw(),0)      ],
                          [Dw(Zw(),0),     Dw(Zw(0,-1,0,0),0)]])

gates['w'] = Dw(Zw.omega(), 0)

