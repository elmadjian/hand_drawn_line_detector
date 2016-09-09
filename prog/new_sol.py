import sys, cv2, heapq, itertools
import numpy as np
from datetime import datetime


#===============================================================================
class Graph():
    def __init__(self, img=None):
        self.V = 0
        self.N = 0
        self.seed = None
        self.node = {}
        self.img = img

    def set_img(self, img):
        self.img = img

    # def set_seed(self, seed):
    #     self.seed = Node(seed)
    #     self.__mark_node(seed)
    #     self.V += 1

    def set_arch(self, node_A, node_B):
        node_A.set_arch(node_B)
        self.node[node_A.pixel] = node_A
        self.node[node_B.pixel] = node_B
        self._mark_node(node_B.pixel)
        self.V += 2
        self.N += 1

    def has_arch(self, pixel_A, pixel_B):
        node_B = self.node[pixel_B]
        if self.node[pixel_A].has_arch(node_B):
            return True
        return False

    def add_node(self, node):
        self.node[node.pixel] = node
        self.V += 1

    def get_node(self, pixel):
        return self.node[pixel]

    def has_node(self, pixel):
        if pixel in self.node:
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
        self.arch = set()
        self.pixel = pixel
        self.cost = cost

    def has_arch(self):
        if len(self.arch) > 0:
            return True
        return False

    def get_arch(self):
        return self.arch.pop()

    def set_arch(self, node):
        self.arch.add(node)

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
        self.entry_finder = {}
        self.counter = itertools.count()

    def put(self, item, priority):
        if item in self.entry_finder:
            self.remove(item)
        entry = [priority, next(self.counter), item]
        self.entry_finder[item] = entry
        heapq.heappush(self.queue, entry)

    def remove(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = "removed"

    def pop(self):
        while self.queue:
            priority, count, item = heapq.heappop(self.queue)
            if item != "removed":
                del self.entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

#===============================================================================

MAX_INT = sys.maxsize
G = Graph()
Q = PriorityQueue()



#Main program
#~~~~~~~~~~~~
def main():

    #seeds = [(148,99), (139,165), (129,205), (123,246), (111,315), (103,379), (90,450)]
    #sinks = [(483,88), (466,160), (441,228), (427,291), (408,352), (390,403), (372,462)]
    seeds = [(99,148), (165,139), (205,129), (246,123), (315,111), (379,103), (450,90)]
    sinks = [(88,483), (160,466), (228,441), (291,427), (352,408), (403,390), (462,372)]

    img   = open_img(sys.argv)
    img   = cv2.GaussianBlur(img, (3,3), 5)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = np.ones(gray.shape)
    #cost  = np.full(gray.shape, sys.maxsize)

    for s in seeds:
        label[s] = 0
    for s in sinks:
        label[s] = 0

    pos = seeds[0]
    G.set_img(img)
    Q.put(Node(pos, 0), 0)
    neighborhood = get_neighborhood(4)
    sink_count = len(sinks)

    while sink_count != 0:
    #for i in range(30000):
        lowest = Q.pop()
        G.add_node(lowest)
        ift(gray, lowest, neighborhood)

        if lowest.pixel in sinks:
            sink_count -= 1

        # cv2.imshow("teste", img)
        # k = cv2.waitKey(0)
        # if k & 0xFF == ord('q'):
        #     sys.exit()

    for s in sinks:
        view_path(img, s)
    #view_path(img, sinks[6])

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


# Get local minimum of adjacency
#--------------------------------
def get_neighborhood(neighborhood):
    if neighborhood == 4:
        return [(-1,0), (0,1), (1,0), (0,-1)]
    if neighborhood == 8:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]


#---------------------------------
def ift(img, node, neighborhood):
    for i in neighborhood:
        pixel = (node.y() + i[0], node.x() + i[1])
        if not is_valid_pixel(img, pixel):
            continue
        if not G.has_node(pixel):
            n = Node(pixel)
            cost = abs(img[pixel] - img[node.pixel]) + img[pixel]
            if cost < n.cost:
                if n.cost != sys.maxsize:
                    Q.remove(n)
                n.cost = cost
                G.set_arch(n, node)
                Q.put(n, cost)

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
    node = G.get_node(sink)
    while (node.has_arch()):
        img[node.pixel] = (0,255,255)
        node = node.get_arch()



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
