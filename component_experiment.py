from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

rand_fw = create_random_fw(100,0.1, 2) 
print(len(rand_fw.nodes))
draw_framework(rand_fw)
num_comps = constructive_pebble_game(rand_fw, 2, 3, save=True) 
fig = plt.figure(figsize=(20,10))
# plotting the number of comps(reversed to show removal)
plt.plot(num_comps)
fig.savefig("component_images/comp_numbers.pdf")
plt.show()
