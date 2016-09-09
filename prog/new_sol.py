import sys, cv2, queue
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

    def set_seed(self, seed):
        self.seed = Node(seed)
        self.__mark_node(seed)
        self.V += 1

    def set_arch(self, node_A, node_B):
        node_A.set_arch(node_B)
        self.node[node_A.pixel] = node_A
        self.node[node_B.pixel] = node_B
        self.__mark_node(node_B.pixel)
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

    def __mark_node(self, pixel):
        self.img[pixel] = (0,255,0)



#===============================================================================
class Node():
    def __init__(self, pixel):
        self.arch = set()
        self.pixel = pixel
        self.time = datetime.now()

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
        self.time < other.time

#===============================================================================

MAX_INT = sys.maxsize
G = Graph()
Q = queue.PriorityQueue()



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
    G.set_seed(pos)
    Q.put((-1, G.seed))
    sink_count = len(sinks)

    #while sink_count != 0:
    for i in range(10):
        lowest = Q.get()[1]
        print("lowest:", lowest.pixel)
        neighbor = G.has_4_neighbor(lowest.pixel)
        if neighbor:
            G.set_arch(lowest, neighbor)

        neighborhood_4_cost(gray, lowest)

        #TODO
        #Escrever um codigo para mostrar o grafo, com as setinhas e tudo o mais
        print("arco:", G.get_node(lowest).get_arch())

        if lowest.pixel in sinks:
            sink_count -= 1

        cv2.imshow("teste", img)
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            sys.exit()

    view_paths(img, sinks)
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


# Get local minimum of adjacency
#--------------------------------
def neighborhood_4_cost(img, node):
    adj_list = [(-1,0), (0,1), (1,0), (0,-1)]
    for i in adj_list:
        pixel = (node.y() + i[0], node.x() + i[1])
        if not G.has_node(pixel):
            n = Node(pixel)
            G.add_node(n)
            cost = abs(img[pixel] - img[node.pixel]) + img[pixel]
            Q.put((cost, n))

#---------------------------------
def neighborhood_8_cost(img, node):
    adj_list = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for i in adj_list:
        pixel = (node.y() + i[0], node.x() + i[1])
        if not G.has_node(pixel):
            n = Node(pixel)
            G.add_node(n)
            cost = abs(img[pixel] - img[node.pixel]) + img[pixel]
            Q.put((cost, n))

#-------------------------
# def count():
#     if not hasattr(count, "counter"):
#         count.counter = 0
#     count.counter += 1
#     return count.counter

#------------------------------
def view_paths(img, sinks):
    for s in sinks:
        node = G.get_node(s)
        while (node.has_arch()):
            img[node.pixel] = (0,255,255)
            #print(len(node.arch))
            node = node.get_arch()

    cv2.imshow("teste", img)
    k = cv2.waitKey(0)
    if k & 0xFF == ord('q'):
        sys.exit()


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
