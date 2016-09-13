import sys, cv2, bisect
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
        self.visited = set()

    def set_img(self, img):
        self.img = img

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
        self._mark_node(node.pixel)
        self.V += 1

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
        self.arch = None
        self.pixel = pixel
        self.cost = cost

    def has_arch(self):
        if self.arch:
            return True
        return False

    def get_arch(self):
        return self.arch

    def pop_arch(self):
        arch = self.arch
        self.arch = None
        return arch

    def set_arch(self, node):
        self.arch = node

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

class Line:
    def __init__(self, theta, points):
        self.theta = theta
        self.list = points

#===============================================================================

MAX_INT = sys.maxsize
G = Graph()
Q = PriorityQueue()



#Main program
#~~~~~~~~~~~~
def main():

    seeds = [(99,148), (165,139), (205,129), (246,123), (315,111), (379,103), (450,90)]
    sinks = [(88,483), (160,466), (228,441), (291,427), (352,408), (403,390), (462,372)]
    # seeds = [(47,16)] #teste.png
    # sinks = [(10,24), (45,88), (62,91), (92,84), (94,49), (89,18)] #teste.png
    # seeds = [(14,10)] #teste2.png
    # sinks = [(68,65)] #teste2.png
    #seeds = [(12,2)] #teste3.png
    #sinks = [(12,22)] #teste3.png

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
    node = Node(pos, 0)
    G.add_node(node)
    #Q.put(n, 0)
    neighborhood = get_neighborhood("asterisk")
    sink_count = len(sinks)

    star_burst(gray, node, 20, 60)

    #while sink_count != 0:
    # for i in range(2000):
    #     node = get_line(gray, node, neighborhood)
    # #     ift(gray, lowest, neighborhood)
    # #
    # #     if lowest.pixel in sinks:
    # #         sink_count -= 1
    # #
    cv2.imshow("teste", img)
    k = cv2.waitKey(0)
    if k & 0xFF == ord('q'):
        sys.exit()
    #
    # # for s in sinks:
    # #     view_path(img, s)
    # view_path(img, sinks[6])
    #
    # cv2.imshow("teste", img)
    # k = cv2.waitKey(0)
    # if k & 0xFF == ord('q'):
    #     sys.exit()
    # cv2.destroyAllWindows()

# Get a window centered on pixel p of size (2*k + 1)^2
#-----------------------------------------------------
def get_window(img, p, k):
    lin = p[0]
    col = p[1]
    return img[lin-k:lin+k+1, col-k:col+k+1]


# Get local adjacency
#---------------------
def get_neighborhood(neighborhood, length=5):
    if neighborhood == 4:
        return [(-1,0), (0,1), (1,0), (0,-1)]
    if neighborhood == 8:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    if neighborhood == "asterisk":
        lines = [[],[],[],[],[],[],[],[]]
        for i in range(1, length):
            lines[0].append((-i,-i))
            lines[1].append((-i, 0))
            lines[2].append((-i, i))
            lines[3].append(( 0, i))
            lines[4].append((i , i))
            lines[5].append((i , 0))
            lines[6].append((i, -i))
            lines[7].append((0, -i))
        return lines
#minimum line cost
#---------------------------------
def get_line(img, node, neighborhood):
    min_cost  = sys.maxsize
    curr_line = 0
    for l in range(len(neighborhood)):
        line_cost = 0
        for i in neighborhood[l]:
            pixel = (node.y() + i[0], node.x() + i[1])
            if not is_valid_pixel(img, pixel):
                continue
            line_cost += abs(img[pixel] - img[node.pixel]) + img[pixel]
            if G.has_node(pixel):
                line_cost += 255
            if line_cost > min_cost:
                break
        if line_cost < min_cost:
            min_cost = line_cost
            curr_line = l

    for i in neighborhood[curr_line]:
        pixel = (node.y() + i[0], node.x() + i[1])
        if not is_valid_pixel(img, pixel):
            continue
        G.add_node(Node(pixel))

    y = neighborhood[curr_line][3][0]
    x = neighborhood[curr_line][3][1]
    # #print(x, y)
    return Node((node.y() + y, node.x() + x))

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
        node = node.pop_arch()

#---------------------------------------
def star_burst(img, root, n, r):
    '''
    root -> point of origin
    n -> number of lines to draw (must be even)
    r -> line length (in pixels)
    '''
    theta = (2 * np.pi) / n
    for i in range(n):
        t = theta*i
        pixels = draw_line(root, t, r)
        print("theta:", np.degrees(t), "custo:", get_cost(img, pixels))


def draw_line(root, theta, length):
    x = root.x()
    y = root.y()
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    list_x = np.linspace(x, x + dx, length)
    list_y = np.linspace(y, y + dy, length)
    return list(zip(list_y, list_x))

def get_cost(img, pixels):
    cost = 0
    for i in range(len(pixels)-1):
        cost += abs(img[pixels[i]] - img[pixels[i+1]]) + img[pixels[i+1]]
        #cost += img[pixels[i]]**2
    return cost




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
