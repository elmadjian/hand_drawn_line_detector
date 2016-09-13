import sys, cv2, bisect
import numpy as np
import skimage
from scipy import stats
from skimage import filters, morphology, util
from datetime import datetime



#===============================================================================
class Graph():
    def __init__(self, img=None):
        self.seed = None
        self.node = {}
        self.img = img
        self.visited = set()

    def set_img(self, img):
        self.img = img

    def set_arch(self, node_A, node_B):
        node_A.set_out_arch(node_B)
        node_B.set_in_arch(node_A)
        self.node[node_A.pixel] = node_A
        self.node[node_B.pixel] = node_B
        #self._mark_node(node_B.pixel)

    def has_arch(self, pixel_A, pixel_B):
        node_B = self.node[pixel_B]
        if self.node[pixel_A].has_out_arch(node_B):
            return True
        return False

    def add_node(self, node):
        self.node[node.pixel] = node

    def get_node(self, pixel):
        return self.node[pixel]

    def has_node(self, pixel):
        if pixel in self.node:
            return True
        return False

    def add_visited(self, pixel):
        self.visited.add(pixel)

    def is_visited(self, pixel):
        if pixel in self.visited:
            return True
        return False

    def has_8_neighbor(self, pixel):
        adj_list = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        for i in adj_list:
            p = (pixel[0] + i[0], pixel[1] + i[1])
            if p in self.node:
                return self.node[p]
        return False

    def has_4_neighbor(self, pixel):
        adj_list = [(-1,0), (0,1), (1,0), (0,-1)]
        for i in adj_list:
            p = (pixel[0] + i[0], pixel[1] + i[1])
            if p in self.node:
                return self.node[p]
        return False

    def _mark_node(self, pixel):
        self.img[pixel] = (0,100,0)



#===============================================================================
class Node():
    def __init__(self, pixel, cost=sys.maxsize):
        self.in_arch = set()
        self.out_arch = set()
        self.pixel = pixel
        self.cost = cost

    def has_in_arch(self):
        if self.in_arch:
            return True
        return False

    def has_out_arch(self):
        if self.out_arch:
            return True
        return False

    def get_in_arch(self):
        return list(self.in_arch)[0]

    def get_out_arch(self):
        return list(self.out_arch)[0]

    def get_in_degree(self):
        return len(self.in_arch)

    def get_out_degree(self):
        return len(self.out_arch)

    def set_in_arch(self, node):
        self.in_arch.add(node)

    def set_out_arch(self, node):
        self.out_arch.add(node)

    def y(self):
        return self.pixel[0]

    def x(self):
        return self.pixel[1]

    def __lt__(self, other):
        self.cost < other.cost

#===============================================================================

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.nodes = {}

    def put(self, item, priority):
        if item in self.nodes:
            pos = bisect.bisect_right(self.queue, [self.nodes[item], item])
            del self.queue[pos-1]
        bisect.insort_right(self.queue, [priority, item])
        self.nodes[item] = priority

    def pop(self):
        if self.queue:
            item = self.queue.pop(0)[1]
            #print("queue:", len(self.queue))
            #print("nodes:", len(self.nodes))
            if item in self.nodes:
                del self.nodes[item]
            return item
        raise KeyError('pop from an empty priority queue')

#===============================================================================

MAX_INT = sys.maxsize
G = Graph()
H = Graph()
Q = PriorityQueue()



#Main program
#~~~~~~~~~~~~
def main():

    seeds = [(99,148), (165,139), (205,129), (246,123), (315,111), (379,103), (450,90)]
    sinks = [(88,483), (160,466), (228,441), (291,427), (352,408), (403,390), (462,372)]
    color = [(0,128,255),(0,255,128),(0,255,255),(0,0,255),(128,0,255),(255,255,0),(255,51,153)]
    # seeds = [(47,16)] #teste.png
    # sinks = [(10,24), (45,88), (62,91), (92,84), (94,49), (89,18)] #teste.png
    # seeds = [(14,10)] #teste2.png
    # sinks = [(68,65)] #teste2.png
    #seeds = [(12,2)] #teste3.png
    #sinks = [(12,22)] #teste3.png

    img   = open_img(sys.argv)
    img   = cv2.GaussianBlur(img, (3,3), 5)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = filters.sobel(gray)
    norm  = skimage.img_as_ubyte(sobel)
    norm *= 5

    # cv2.imshow("teste", norm)
    # cv2.waitKey(0)

    thresh = norm > 35
    thresh = np.uint8(thresh)
    # cv2.imshow("teste", thresh*255)
    # cv2.waitKey(0)

    paths = morphology.binary_closing(thresh)
    paths = skimage.img_as_ubyte(paths)
    paths = 255-paths
    #
    # cv2.imshow("teste", paths)
    # cv2.waitKey(0)


    #cost  = np.full(gray.shape, sys.maxsize)


    #for i in range(len(seeds)):
    pos = seeds[6]
    G.set_img(img)
    n = Node(pos, 0)
    G.add_node(n)
    Q.put(n, 0)
    neighborhood = get_neighborhood(4)
    sink_count = len(sinks)

    while sink_count != 0:
    #for i in range(60000):
        lowest = Q.pop()
        G.add_visited(lowest.pixel)
        #print("visitei:", lowest.pixel)
        ift(paths, lowest, neighborhood)

        if lowest.pixel in sinks:
            sink_count -= 1

    build_path_graph(img, sinks)
    find_correct_path(img, seeds[6], sinks, color[1])


    cv2.imshow("teste", img)
    k = cv2.waitKey(0)
    if k & 0xFF == ord('q'):
        sys.exit()
    cv2.destroyAllWindows()

# Get a window centered on pixel p of size (2*k + 1)^2
#-----------------------------------------------------
def get_window(img, p, k):
    lin = p[0]
    col = p[1]
    return img[lin-k:lin+k+1, col-k:col+k+1]


# Get local adjacency
#---------------------
def get_neighborhood(neighborhood):
    if neighborhood == 4:
        return [(-1,0), (0,1), (1,0), (0,-1)]
    if neighborhood == 8:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]


#Image-Forest Transform
#---------------------------------
def ift(img, node, neighborhood):
    for i in neighborhood:
        pixel = (node.y() + i[0], node.x() + i[1])
        if not is_valid_pixel(img, pixel):
            continue
        if not G.is_visited(pixel):
            n = Node(pixel) if not G.has_node(pixel) else G.get_node(pixel)
            cost = abs(img[pixel] - img[node.pixel]) + img[pixel]
            #print("node:", node.pixel, "V:", n.pixel, "custo:", cost, "custo_V:", n.cost)
            if cost < n.cost:
                n.cost = cost
                G.set_arch(n, node)
                Q.put(n, cost)
    #print("=====================")

#-------------------------
def is_valid_pixel(img, pixel):
    y = pixel[0]
    x = pixel[1]
    if y >= 0 and y < img.shape[0]:
        if x >= 0 and x < img.shape[1]:
            return True
    return False


#Shows a path backwards, stating from sink
#-----------------------------------------
def view_path(img, sink):
    x_points = []
    y_points = []
    node = G.get_node(sink)
    while (node.has_out_arch()):
        x_points.append(node.x())
        y_points.append(node.y())
        img[node.pixel] = (0,255,255)
        node = node.get_out_arch()
    return x_points, y_points

#---------------------------------------
def build_path_graph(img, sinks):
    for s in sinks:
        node = G.get_node(s)
        while (node.has_out_arch()):
            p1 = node.pixel
            p2 = node.get_out_arch().pixel
            #img[p1] = (0,255,255)
            n1 = Node(p1) if not H.has_node(p1) else H.get_node(p1)
            n2 = Node(p2) if not H.has_node(p2) else H.get_node(p2)
            H.set_arch(n1, n2)
            node = node.get_out_arch()

#--------------------------------------
def find_correct_path(img, seed, sinks, color):
    node = H.get_node(seed)
    crossing = 1
    while (node.pixel not in sinks):
        best_node = node.get_in_arch()
        if node.get_in_degree() > 1:
            best_cost = 0
            crossing += 1
            for n in node.in_arch:
                vec1 = get_vector(node, 15, "out")
                vec2 = get_vector(n, 15, "in")
                cost = np.dot(vec1, vec2)
                if abs(cost) > best_cost:
                    best_cost  = abs(cost)
                    best_node = n

        img[node.pixel] = color
        node = best_node


#-------------------
def get_pixel_list(node, length, direction):
    l = 0
    x_list = []
    y_list = []
    while l < length:
        x_list.append(node.x())
        y_list.append(node.y())
        node = node.get_in_arch() if direction == "in" else node.get_out_arch()
        l += 1
    return x_list, y_list

#----------------
def get_vector(node, length, direction):
    l = 0
    first_pixel = node.pixel
    last_pixel = None
    while l < length:
        last_pixel = node.pixel
        node = node.get_in_arch() if direction == "in" else node.get_out_arch()
        l += 1
    vector = [last_pixel[1]-first_pixel[1], last_pixel[0]-first_pixel[0]]
    norm   = np.sqrt(vector[0]**2 + vector[1]**2)
    return [vector[0]/norm, vector[1]/norm]


                    # n = H.get_node(node.pixel)
                    # print(n.get_degree())
                    # if n.get_degree() > 1:
                    #     cv2.circle(img, (n.pixel[1], n.pixel[0]), 5, (0,255,0))


#--------------------------
def line(x, a, b):
    return a * x + b

# Open an image
#--------------
def open_img(argv):
    if len(argv) > 2:
        print("Usage: <this_program> <your_image>")
        sys.exit()
    elif len(argv) == 1:
        img_name = input("Type in the name of the image: ")
        return cv2.imread(img_name)
    else:
        return cv2.imread(argv[1])





#===============================================================================
if __name__ == "__main__":
    main()
