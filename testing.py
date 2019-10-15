from framework import *
from pebble_game import *
from nose.tools import ok_
import numpy as np

# initialise the seed for reproducibility
np.random.seed(102)


fw_2d = create_framework([0,1,2,3], [(0,1), (0,3), (1,2), (1,3), (2,3)], [(2,3), (4,4), (5,2), (1,1)])
# a 3d fw constricted to 2d
fw_3d = create_framework([0,1,2,3], [(0,1), (0,3), (1,2), (1,3), (2,3)], [(2,3, 0), (4,4, 0), (5,2, 0), (1,1, 0)])
R = create_rigidity_matrix(fw_3d, 3)

fig_39_nodes = [0,1,2,3]
fig_39_edges = [(0,1), (0,2), (0,3), (1,2), (2,3)]
fig_39_pos = [(0,0), (3,0), (3,2), (0,2)]

fig_39_fw = create_framework(fig_39_nodes, fig_39_edges, fig_39_pos)
R39 = create_rigidity_matrix(fig_39_fw, 2)

def_node = [0,1,2,3]
def_edge = [(0,1), (0,3), (1,2), (2,3)]
def_pos = [(0,0), (4,0), (4,2), (0,2)]

deformable_fw = create_framework(def_node, def_edge, def_pos)
R = create_rigidity_matrix(deformable_fw, 2)

rigid_3d = create_framework([0,1,2,3,4],
            [(0,1), (0,3), (1,2), (1,3), (2,3), (0,2), (0,4), (1,4), (2,4)],
            [(2,3, 0), (4,4, 5), (5,2, 0), (1,1, 0), (10,10,10)])

fw_1d = create_framework([0,1,2],
            [(0,1), (1,2), (0,2)],
            [1,6,20])

ok_(is_inf_rigid(fw_2d, 2))
ok_(not is_inf_rigid(fw_3d, 3))
ok_(is_inf_rigid(fw_1d, 1))
ok_(not is_inf_rigid(deformable_fw, 2))
# draw_framework(deformable_fw)

R = create_rigidity_matrix(deformable_fw, 2)
# creating a force to apply
# as an example, move points 0 and 2 towards each other
# f is a d*n length vector
f = [0] * len(R[0])
f[0] = 1
f[1] = -1
f[4] = -1
f[5] = 1
f = np.array(f)
print(R)
print(f)

print(R.dot(f))

rand_fw = create_random_fw(2,2,0.2)
# draw_framework(rand_fw)
reduced_fw = create_reduced_fw(2,2,0.2)

p = pebble_game(reduced_fw, 2, 3)
print(p[1])
draw_framework(reduced_fw)
draw_comps(reduced_fw, p[1])
