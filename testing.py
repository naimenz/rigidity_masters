from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

# initialise the seed for reproducibility np.random.seed(102)

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

reduced_fw = create_reduced_fw(2,2,0.2)

# p = pebble_game(reduced_fw, 2, 3)
# print(p[1])
# draw_framework(reduced_fw)
# draw_comps(reduced_fw, p[1])
# experimenting with reducing a framework gradually and tracking the number of components
rand_fw = create_random_fw(2,2,0.2)
num_comps = constructive_pebble_game(rand_fw, 2, 3)
fig = plt.figure(figsize=(20,10))
# plotting the number of comps(reversed to show removal)
plt.plot(num_comps)
fig.savefig("comp_numbers.pdf")
plt.show()
# draw_framework(rand_fw, "before.pdf")
# num_comps = []
# counter = 0
# while len(rand_fw.edges) > 2*len(rand_fw.nodes):
#         index = np.random.choice(len(rand_fw.edges))
#         edge = list(rand_fw.edges)[index]
#         if rand_fw.degree(edge[0]) > 2 and rand_fw.degree(edge[1]) > 2:
#             counter += 1
#             rand_fw.remove_edge(edge[0], edge[1])
#             comps = pebble_game(rand_fw, 2, 3)[1]
#             num_comps.append(len(comps))
#             draw_comps(rand_fw, comps, filename="after"+str(counter)+".pdf", show=False)
#             plt.close("all")

# draw_comps(rand_fw, comps, "after.pdf")

