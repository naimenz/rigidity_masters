from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
import scipy 

import sys

# initialise the seed for reproducibility 
np.random.seed(100)

Hfw = create_framework([0,1,2,3], [(0,1), (0,2), (1,2), (1,3), (2,3)], [(0,0), (1,0), (0.3,0.4), (0.8,0.5)])
fw = make_nice_fw(50,1)
# draw_framework(fw)
SSS, SCS = subbases(fw)

G = G_f(fw, SCS)
edge_dict = get_edge_dict(fw)
index = edge_dict[(36,39)]
k = fw.graph["k"]
edge_vec = np.zeros(len(G))
edge_vec[index] = 1
Ci = k * G.dot(edge_vec)
# print(np.linalg.norm(Ci))

b1 = np.array([1,0,0])
b2 = np.array([0,1,0])
b3 = np.array([0,0,1])
# print(b1.dot(b2))

# get the rotation matrix required to rotate a to point in the direction of b
# NOTE: assumes a and b are the same length
def get_rot_mat(a,b):
    if len(a) != len(b):
        print("UHOH vectors aren't same length")
    n = len(a)
    a_hat = a/np.linalg.norm(a)
    b_hat = b/np.linalg.norm(b)
    basis = scipy.linalg.orth(np.stack((a_hat,b_hat),axis=1))
    u = basis[:,0]
    v = basis[:,1]
    cost= a_hat.dot(b_hat)
    sint = np.sin(np.arccos(cost))
    if cost == 1:
        print("already parallel")
    if cost == -1:
        print("anti parallel")
    # sint = np.linalg.norm(np.cross(u,v))
    # from stack overflow: https://math.stackexchange.com/questions/197772/generalized-rotation-matrix-in-n-dimensional-space-around-n-2-unit-vector#comment453048_197778
    A = np.eye(n) + sint*(np.outer(v,u) - np.outer(u,v)) + (cost -1)*(np.outer(u,u) + np.outer(v,v))
    return A

A = Hbar_mat(Hfw)
Ainv = Hbar_inv_mat(Hfw)
ok_(np.allclose(A, A@Ainv@A))
edge = (0,2)
Auinv = update_Hinv(Hfw, edge, Ainv)
Hfw.edges[edge]["lam"] = 0
Ar = Hbar_mat(Hfw)
Arinv = Hbar_inv_mat(Hfw)
print(np.allclose(Auinv, Arinv))

c = Ci/np.linalg.norm(Ci)
c0 = SCS[0]
A = get_rot_mat(c0, c)
ok_(np.allclose(c, A.dot(c0)))
SCS_rot =( A @ SCS.T).T

ki = 0.5
Ci2 = Ci.dot(Ci)



