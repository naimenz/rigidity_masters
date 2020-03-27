from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
from scipy.optimize import minimize

# initialise the seed for reproducibility 
np.random.seed(102)

triangle = create_framework([0,1,2], [(0,1), (0,2), (1,2)], [(1,0), (3,0), (2,2)])
triangle.edges[(0,1)]["lam"] = 0
draw_framework(triangle, ghost=True)

tensions = [1,0,0]
# strains = exts_to_strains(triangle, extensions(triangle, tensions))
# draw_strains(triangle, strains, ghost=True)

print(energy(np.array([1,1,1,1,1,0]), triangle))
u0 = np.array([0,0,0,0,0,0])
edge = (0,1)
strain = 1
constraints = {"type":"eq", "fun":source_strain, "args":(triangle, edge, strain)}
minimised = minimize(energy, u0, args=(triangle), constraints=constraints)
print(minimised)

fwc = update_pos(triangle, minimised.x)
draw_framework(fwc, ghost=True)

