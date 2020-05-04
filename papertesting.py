from PIL import Image
from framework import *
from pebble_game import *
from constructive_pebble_game import *
from nose.tools import ok_
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

im = load_image("bluedottedgraph.png")

blue = np.array([0,0,255])
ys, xs = np.array(np.where(np.all(im == np.array(blue), axis=-1)))
ys = 1000 - ys

positions = np.array([xs/1000, ys/1000]).T
# np.savetxt("nodes.csv",positions,delimiter=",")

nodes = list(range(len(positions)))
edges = np.loadtxt("edges.csv",dtype=np.int, delimiter=',')
# fw.add_edges_from(edges)
fw = create_framework(nodes, edges, positions)
draw_framework(fw, ghost=True)


# returns a tuned network
source = (187,188)
target = (0,1)
nstars = [1.0]
fw.add_edges_from([source, target])
fw = add_lengths_and_stiffs(fw)
fw.edges[source]["lam"] = 0
fw.edges[target]["lam"] = 0
# borrowing heavily from draw_tensions, trying to draw the strains on the bonds
def draw_strains(fw, strs, source=None, target=None, ghost=False, filename=None):
    # works on a numpy array
    strs = np.array(strs)
    # drawing the nodes of the graph
    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_aspect('equal')
    nodeview = fw.nodes

    pos = {node: nodeview[node]["position"] for node in nodeview}
    # SETTING AXIS LIMITS
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf
    for val in pos.values():
        minx = min(minx, val[0])
        miny = min(miny, val[1])
        maxx = max(maxx, val[0])
        maxy = max(maxy, val[1])
    xbuf = (maxx - minx) / 10
    ybuf = (maxy - miny) / 10
    plt.xlim(minx - xbuf, maxx + xbuf)
    plt.ylim(miny - ybuf, maxy + ybuf)
    # END SETTING AXIS LIMITS

    e_labels=dict()
    for i, edge in enumerate(fw.edges):
        e_labels[edge] = np.round(strs[i], 1)

    cmap = plt.cm.coolwarm
    nx.draw_networkx_edges(fw, pos, edgelist=fw.edges, edge_color=strs,
            width=4, edge_cmap=cmap, edge_vmin=-max(abs(strs)), edge_vmax=max(abs(strs)), with_labels=True)
    if ghost:
        ghost_es = set([edge for edge in fw.edges if fw.edges[edge]["lam"]==0])
        nx.draw_networkx_edges(fw, pos, edgelist=ghost_es, edge_color="grey",style="dashed")
    if source:
        nx.draw_networkx_nodes(fw, pos, nodelist=source, node_color="g")
    if target:
        nx.draw_networkx_nodes(fw, pos, nodelist=target, node_color="r")
       
    # bbox = dict(boxstyle="round", alpha=0.0)
    # nx.draw_networkx_edge_labels(fw, pos, e_labels, font_size=4)#, bbox=bbox)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
    plt.show()


# import time
# tic = time.perf_counter()
# fw = SM_tune_network(fw, source, target, tension=1, nstars=nstars)
# toc = time.perf_counter()
# print(f"SM took {toc - tic:0.4f} seconds")

# import time
# tic = time.perf_counter()
# fwc = BF_tune_network(fw, source, target, tension=1, nstars=nstars)
# toc = time.perf_counter()
# print(f"BF took {toc - tic:0.4f} seconds")

# import time
# tic = time.perf_counter()
# fwc = GF_tune_network(fw, source, target, tension=1, nstars=nstars)
# toc = time.perf_counter()
# print(f"GF took {toc - tic:0.4f} seconds")

# red_edges = [(1,5),(9,18),(37,46),(54,69),(77,84),(174,173), (0,1), (187,188)]
# edges removed by the SM alg. (written to save doing it again)
SM_red_edges = [(1,5), (37,46), (9,18), (62,69), (2,8), (45,59)]
for edge in SM_red_edges:
    fw.edges[edge]["lam"] = 0 
draw_framework(fw, ghost=True)
tensions = [0]*len(fw.edges)
edge_dict = get_edge_dict(fw)
tensions[edge_dict[source]] = 1
strs = strains(fw, tensions)
draw_strains(fw, strs, source, target, ghost=True, filename="TEST_STRAINS.png")
print("target, source strains:",strs[edge_dict[target]], strs[edge_dict[source]])

# ratios = animate(fw, source, target, nstars, fileroot="images/ROCKS_EDGES/", s_max=1, tensions=1)
