from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

np.random.seed(123)
fw = create_framework([0,1,2], [(0,1), (1,2), (0,2)], [(0,0), (1,0), (0.5, np.sqrt(3)/2)])

draw_framework(fw)

forces = [1,1,-1,-1,0,0]
draw_strains_from_forces(fw, forces, "stresses.png")

