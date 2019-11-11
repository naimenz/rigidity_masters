import numpy as np

# calculate the euclidean distance between two points
def dist(p1, p2):
    return np.sqrt( (p1[0] - p2[0])**2 + (p1[1]-p2[1])**2 )

# function to calculate the grid indices of a point
# grids are defined as below, shape 2*n x 2*n representing step size r
def grid_indices(point, a, step):
    x,y = point
    # grid runs from -step*n to step*n so we divide by r, take the floor, and subtract n
    i = (np.floor(x/step)) + a
    j = (np.floor(y/step)) + a
    # returning as integers to be used for indexing arrays
    return int(i), int(j)

# get the neighbouring points of a point in the grid
# also sneakily contains a check that the input point is actually in the domain
# NOTE: This defines a circle, so some of the grid isn't used
def find_neighbours(point, grid, a, step):
    i, j = grid_indices(point, a, step)
    if (i-a)**2 + (j-a)**2 > (a)**2:
        return None, False

    # values of neighbouring cells gives the indices of their points
    rv = []
    for s in range(-5, 6):
        for t in range(-5, 6):
            if i+s >= 0 and i+s < 2*a and j+t >= 0 and j+t < 2*a:
                rv.append(grid[i+s, j+t])

    return [index for index in rv if index >= 0], True


# get the neighbouring points of a point in the grid
# also sneakily contains a check that the input point is actually in the domain
# def find_neighbours(point, grid, a, step):
#     i, j = grid_indices(point, a, step)
#     if i < 0 or i > 2*a or j < 0 or j > 2*a:
#         return None, False
#     # values of neighbouring cells gives the indices of their points
#     rv = []
#     for s in range(-5, 6):
#         for t in range(-5, 6):
#             if i+s >= 0 and i+s < 2*a and j+t >= 0 and j+t < 2*a:
#                 rv.append(grid[i+s, j+t])

#     return [index for index in rv if index >= 0], True

# generate a set of random points using the Poisson disc sampling 
# algorithm from Nick Bostock's blog
# Inputs:
# n - number of points to be distributed
# k - number of candidate points generated at each step
# r - minimum distance between points
# centre - the location of the starting point
# Outputs:
# rv - length n list of (x,y) tuples
def poisson_disc_sample(n, k=30, r=1, centre=(0,0), seed=None):
    if seed:
        np.random.seed(seed)
    # making a really large grid so that we never reach the edge of it
    # the grid will be centred on (0,0) and extend n*r in each direction (in steps of r/root(2))
    # (creating a 2*r*n x 2*r*n domain)
    # because we are going up in steps of r/root(2), we need 2*root(2) * n steps
    a = int(np.ceil(np.sqrt(2) * n))
    step = r / np.sqrt(2)
    grid = np.full((2*a, 2*a), -1)
    # initialising the active list and chosen samples with (0,0)
    rv = [centre]
    active_list = [0]
    grid[grid_indices(rv[0], a, step)] = 0

    # loop while the active list still has points in it and we havent placed all points
    while len(active_list) > 0 and len(rv) < n:
        # plt.scatter(*zip(*rv))
        # plt.scatter([rv[0][0]], [rv[0][1]], c='r')
        # plt.show()
        # picking a random active point is the normal thing to do,
        # but i want to experiment with different point selection
        # index = np.random.randint(0, len(active_list))
        i = active_list[0]
        xi, yi = rv[i]
        success = False
        c = 0
        while not success and c < k:
            c += 1
            r_new = np.sqrt(np.random.uniform(r**2, (2*r)**2))
            phi = np.random.uniform(0, 2*np.pi)
            x, y = xi + r_new*np.cos(phi), yi + r_new*np.sin(phi)
            neighbours = find_neighbours((x,y), grid, a, step)
            # if we are in the grid, then check the neighbours
            if neighbours[1]:
                close = False
                for neighbour in neighbours[0]:
                    xn, yn = rv[neighbour]
                    # if a neighbour is too close, we continue to the next point
                    if dist((x, y), (xn, yn)) < r:
                        # print(neighbour,(xn, yn),"TOO CLOSE to", x, y)
                        close = True
                        break

                if close:
                    continue
                # if no points were too close, we succeeded! Add the point and update the active list
                # Don't forget to add it to the grid
                else:
                    success = True
                    rv.append((x,y))
                    active_list.append(len(rv)-1)
                    grid[grid_indices((x,y), a, step)] = len(rv)-1
        
        # next step depends on the termination condition of the loop
        # if not successful, we remove the current point from the active list
        if not success:
            del active_list[0]
    return rv
