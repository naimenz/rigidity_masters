from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
import scipy 

import sys

# initialise the seed for reproducibility 
np.random.seed(100)

fw = create_framework([0,1,2,3], [(0,1), (0,2), (1,2), (1,3), (2,3)], [(0,0), (1,0), (0.3,0.4), (0.8,0.5)])
fw = make_nice_fw(40,1)

# draw_framework(fw)

# a = np.array([[1,1,1],[0,1,0],[1,1,0]])
# b = (8,2,-1)

A = Hbar_mat(fw)
Ainv = Hbar_inv_mat(fw)
# np pinv works
ok_(np.allclose(A, A@Ainv@A))
edge = (0,5)
Au, Auinv = check_update_Hinv(fw, edge, A, Ainv)
fw.edges[edge]["lam"] = 0
An = Hbar_mat(fw)
# update matrix works
ok_(np.allclose(An, Au))

Aupinv = np.linalg.pinv(Au)
ok_(np.allclose(Au, Au @ Aupinv @ Au))
# inverses close
ok_(np.allclose(Auinv, Aupinv))
# update inverse works
ok_(np.allclose(Au, Au @ Auinv @ Au))


