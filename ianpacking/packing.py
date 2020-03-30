import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(2)

# drag is a force proportional to velocity squared
# scale is a parameter to determine how much drag takes place
def drag(vels, scale=0.01):
    norms = np.linalg.norm(vels, axis=1)
    return -scale * (vels.T * norms).T

# defining the overlap parameter 
# c1,c2 are sphere centres, R is radius
# NOTE: a bit more complicated than i previously thought because
# of periodic boundary conds,
# but this is just a scalar so not too bad
def del12(c1, c2, R, l=1):
    # centre to centre x distance 
    c2cx = abs(c1[0] - c2[0])
    x_dist = min(c2cx, abs(c2cx - l))
    c2cy = abs(c1[1] - c2[1])
    y_dist = min(c2cy, abs(c2cy - l))
    r12 = np.sqrt(x_dist**2 + y_dist**2)
    return 1 - r12/(2*R)

# defining the overlap parameter 
# c1,c2 are sphere centres, R is radius
def delold(c1, c2, R):
    # centre to centre distance
    r12 = np.linalg.norm(c1-c2)
    return 1 - r12/(2*R)

# defining the interaction potential (eij is spring constant of contacts)
# alph=2 for harmonic interactions
def V(c1, c2, R, eij=1, alph=2):
    d = del12(c1,c2,R)
    if d > 0:
        return eij * d**alph
    else:
        return 0

# cheaty function to get the force
# negative of the gradient so we just numerically estimate it
# by perturbing slightly in direction of motion (vector of centres)
# NOTE: This is force ON c1 due to c2
# Force ON c2 due to c1 is negative of this
def F(c1, c2, R, stepsize, l=1, eij=1, alph=2):
    # unit vector in direction of changing potential
    # NOTE: now this is tricky because of periodicity
    # basically there are four cases
    circle1 = c1.copy()
    circle2 = c2.copy()
    c2cx = abs(circle1[0] - circle2[0])
    # if the distance is smaller "wrapping around", then put the further circle there for calc
    if c2cx > abs(c2cx - l):
        if circle1[0] >= circle2[0]:
            circle1[0] -= l
        else:
            circle2[0] -= l

    c2cy = abs(circle1[1] - circle2[1])
    # if the distance is smaller "wrapping around", then put the further circle there for calc
    if c2cy > abs(c2cy - l):
        if circle1[1] >= circle2[1]:
            circle1[1] -= l
        else:
            circle2[1] -= l

    d = (c1 - c2) / np.linalg.norm(c1-c2)
    # vector with magnitude stepsize
    h = d * stepsize

    Vh = V(c1 + h, c2, R, eij, alph)
    Vflat = V(c1, c2, R, eij, alph)
    # need to subtract Vh from Vflat because d is already in direction of force
    return (Vflat - Vh)*d / stepsize
    
    
def draw_circles(centres, R, xl=1, yl=1, filename=None):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlim([0, xl])
    ax.set_ylim([0, yl])
    for c in centres:
        circle = plt.Circle(c, R, fill=False)
        ax.add_artist(circle)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def frame_circles(centres, R, xl=1, yl=1):
    for c in centres:
        im = []
        im.append(plt.Circle(c, R, fill=False))
    return im

N = 10      # number of spheres
l = 1      # dimensions of box (ASSUMING SQUARE)
xl = l
yl = l
r0 = 0.1      # initial radius of spheres
mass = 1    # mass of the particles (for working out acceleration)
e = 1

timestep = 0.05   # timestep for running the MD simulation
rstep = 0 # r0 * 4*10**-2     # step for increasing the radius (analogous to compressing box)
diff_step = 0.0001
T = 200         # number of timesteps to run
R = r0           # initial radius is r0

# generating random starting sphere centres
# centres = np.random.random((N, 2))
# # scaling coordinates to uniformly distribute in the box
# centres[:,0] *= xl
# centres[:,1] *= yl

# # particles all start out with balanced forces
# velocities = np.random.random((N,2)) / 2
# # testing particles initially at rest 
# velocities = np.zeros((N,2))
# accelerations = np.zeros((N,2))

# TEST WITH JUST TWO SPHERES
centres = np.array([[0.05,0.05], [0.95, 0.95]])
velocities = np.zeros_like(centres)
acceleration = np.zeros_like(centres)

# running the simulation for T timesteps
ims = []
for t in range(T):
    # starting accelerations are those due to drag
    accelerations = drag(velocities) / mass
    # very inefficient algorithm looping through every node summing forces
    for i in range(len(centres)):
        # we also calculate the acceleration due to drag
        for j in range(i+1, len(centres)):
            force = F(centres[i], centres[j], R, diff_step)
            accelerations[i] += force / mass
            accelerations[j] -= force / mass

    centres = (centres + velocities*timestep) % l 
    velocities += accelerations*timestep
    R += rstep

    ims.append(frame_circles(centres, R, l, ))
    draw_circles(centres, R, l, l, filename="images/c_"+str(t)+".png") 
    print("t=",t)

