import sys, cv2, bisect, skimage, graph, seed_reader
import numpy as np

#global parameters
#~~~~~~~~~~~~~~~~~
alpha   = 2       #power associated with intensity cost
beta    = 300000  #factor associated with curvature cost
radius  = 36      #radius of the circle
bfactor = 25      #how much predecessors are necessary to build a vector

#Main program
#~~~~~~~~~~~~
def main():
    img   = open_img(sys.argv)
    setup = seed_reader.Setup(sys.argv)
    seeds = setup.seeds
    sinks = setup.sinks
    color = setup.colors

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p1, p2 = np.percentile(img, (2, 98))
    norm = skimage.exposure.rescale_intensity(gray, in_range=(p1, p2))
    paths = skimage.img_as_ubyte(norm)
    cv2.imshow("teste", norm)
    cv2.waitKey(0)

    neighborhood = get_neighborhood(8)
    counting = 0


    for i in range(len(seeds)):
        infogrid = np.zeros(paths.shape, dtype="uint8")
        pos = seeds[i]
        cv2.circle(infogrid, (pos[1],pos[0]), radius, 100)
        predecessor = {}
        cost = initialize_costs(paths)
        cost[pos] = 0
        Q = graph.PriorityQueue()
        Q.put(pos, 0)
        sink_count = len(sinks)
        borders = []
        direction = None
        lowest = (0,0)
        radial = True

        while lowest not in sinks:
            lowest = Q.pop()
            infogrid[lowest] = 255
            border_pixel = ift(paths, lowest, neighborhood, infogrid, Q, cost, predecessor, direction, radial)
            if border_pixel:
                direction = get_vector(border_pixel, predecessor)
                Q.empty()
                cost[border_pixel] = 0
                Q.put(border_pixel, 0)
                x,y = border_pixel[1], border_pixel[0]
                infogrid[infogrid==100] = 0
                cv2.circle(infogrid, (x,y), radius, 100)
                if sink_in_circle(sinks, border_pixel, radius):
                    radial = False

        view_path(img, lowest, predecessor, color[i])


# Initialize all pixels with cost infinity
#------------------------------------------
def initialize_costs(img):
    cost = {}
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            cost[(y,x)] = sys.maxsize
    return cost


# Get local adjacency
#---------------------
def get_neighborhood(neighborhood):
    if neighborhood == 4:
        return [(-1,0), (0,1), (1,0), (0,-1)]
    if neighborhood == 8:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]


# Image-Foresting Transform
#------------------------------
def ift(img, current, neighborhood, infogrid, Q, cost_dic,
        predecessor, direction, radial):
    for i in neighborhood:
        pixel = (current[0] + i[0], current[1] + i[1])
        if not is_valid_pixel(img, pixel):
            continue
        if radial:
            if infogrid[pixel] == 100:
                predecessor[pixel] = current
                cost_dic[pixel] = cost_dic[current]
                return pixel
        if infogrid[pixel] != 255:
            cost = cost_dic[current] + ((int(img[current]) + int(img[pixel]))/2)**alpha
            if direction is not None:
                vec = get_vector(current, predecessor)
                curvature = np.dot(vec, direction)
                cost += (1-curvature)*beta
            if cost < cost_dic[pixel]:
                cost_dic[pixel] = cost
                predecessor[pixel] = current
                Q.put(pixel, cost)


# Build a a vector from a current point and some predecessors
#------------------------------------------------------------
def get_vector(current, predecessor):
    pixel = current
    plist = []
    for i in range(bfactor):
        pixel = predecessor[pixel]
        plist.append(pixel)
    x = plist[0][1] - plist[-1][1]
    y = plist[0][0] - plist[-1][0]
    norm = np.sqrt(x**2 + y**2)
    return [x/norm, y/norm]


# Check whether a pixel is inside the image
#-------------------------------------------
def is_valid_pixel(img, pixel):
    y = pixel[0]
    x = pixel[1]
    if y >= 0 and y < img.shape[0]:
        if x >= 0 and x < img.shape[1]:
            return True
    return False


# Shows a path backwards, stating from the sink
#----------------------------------------------
def view_path(img, sink, predecessor, color):
    pred = predecessor[sink]
    while True:
        if pred in predecessor.keys():
            img[pred] = color
            pred = predecessor[pred]
        else:
            break


# Check whether a sink is inside the current circle
#--------------------------------------------------
def sink_in_circle(sinks, center, radius):
    for s in sinks:
        if (s[0]-center[0])**2 + (s[1]-center[1])**2 <= radius**2:
            return True
    return False


# Open an image
#--------------
def open_img(argv):
    if len(argv) != 3:
        print("Usage: <this_program> <your_image> <seeds>")
        sys.exit()
    else:
        return cv2.imread(argv[1])



#===============================================================================
if __name__ == "__main__":
    main()
