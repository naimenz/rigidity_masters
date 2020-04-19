from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np
import scipy 

fw = create_framework([0,1,2,3], [(0,1), (0,2), (1,2), (1,3), (2,3)], [(0,0), (1,0), (0.3,0.4), (0.8,0.5)])

fw = create_framework([0,1,2,3], [(0,1), (1,2), (2,3), (0,3)], [(0,0), (1,0), (1,1), (0,1)])
draw_framework(fw)
