import sys, cv2, queue
import numpy as np


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
        self.seed = seed
        self.__mark_node(seed)

    def set_arch(self, pixel_A, pixel_B):
        node_A = Node(pixel_A)
        node_B = Node(pixel_B)
        node_A.set_arch(node_B)
        node_B.set_arch(node_A)
        self.node[pixel_A] = node_A
        self.node[pixel_B] = node_B
        self.__mark_node(pixel_B)

    def has_arch(self, pixel_A, pixel_B):
        node_B = self.node[pixel_B]
        if self.node[pixel_A].has_arch(node_B):
            return True
        return False

    def get_node(self, pixel):
        return self.node[pixel]

    def __mark_node(self, pixel):
        self.img[pixel] = (0,255,0)



#===============================================================================
class Node():
    def __init__(self, pixel):
        self.arch = set()
        self.node = pixel

    def has_arch(self, node):
        if node in self.arch:
            return True
        return False

    def set_arch(self, node):
        self.arch.add(node)

    def y(self):
        self.node[0]

    def x(self):
        self.node[1]

#===============================================================================

MAX_INT = sys.maxsize
queue = queue.PriorityQueue()
G = Graph()



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
    cost  = np.full(gray.shape, sys.maxsize)

    for s in seeds:
        label[s] = 0
    for s in sinks:
        label[s] = 0

    # for s in seeds:
    #     img = cv2.circle(img, s, 10, (0, 0, 255))
    #
    # for s in sinks:
    #     img = cv2.circle(img, s, 10, (0, 255, 0))

    #print(get_minimum(gray, seeds[0], 3))

    pos = seeds[0]
    G.set_img(img)
    G.set_seed(pos)
    for i in range(600):
        next_pos = get_neighbor(label, gray, pos, 3)
        pos = next_pos
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
def get_neighbors(label, img, p, k):
    adj_list = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    average_list = [0 for i in range(len(adj_list))]
    for i in range(len(adj_list)):
        pixel = (p[0] + adj_list[i][0], p[1] + adj_list[i][1])
        if label[pixel]:
            window = get_window(img, pixel, k)
            gradient = np.gradient(window)
            avg = np.mean(gradient)
            cost = abs(avg)
            average_list[i] = cost
            label[pixel] = 0

    for i in average_list:
        print(i)
    print("---------------------")
    pos = adj_list[np.argmax(average_list)]
    return p[0] + pos[0], p[1] + pos[1]


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
