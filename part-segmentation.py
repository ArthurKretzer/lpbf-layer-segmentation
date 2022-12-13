import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

from matplotlib import rcParams
from matplotlib.path import Path
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.spatial import distance as dist
from PIL import ImageFile
from skimage import img_as_float, img_as_ubyte, io
from skimage.morphology import opening, closing
from skimage.measure import label, regionprops
from skimage.filters import threshold_yen

ImageFile.LOAD_TRUNCATED_IMAGES = True

from time import time

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def warp(argv, filename, bx, by):
    image = filename
    
    #Chose the vertices of the polygon
    tl = (bx[0], by[0])
    bl = (bx[1], by[1])
    tr = (bx[2], by[2])
    br = (bx[3], by[3])

    # load the image, clone it, and initialize the 4 points
    # that correspond to the 4 corners of the chessboard
    pts = np.array([tl, tr, bl, br])

    # loop over the points and draw them on the image
    for (x, y) in pts:
       cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # apply the four point tranform to obtain a "birds eye view" of
    # the chessboard
    warped = four_point_transform(image, pts)
    #warped = cv2.resize(warped, (800, 800))

    height, width = warped.shape[:2]
    warped = cv2.resize(warped, (width, height))

    return warped


start = time()

# Creates csv file to store each layer data
csvFile = open("vision/TrabalhoFinal/22-11-03_11-12-42/intersection/data.csv", 'w+', encoding='UTF-8')
csv_writer = csv.writer(csvFile)
# File header
csv_writer.writerow(['id', 'intersection_percentage'])

mask = io.imread("vision/TrabalhoFinal/mask.jpg")
mask = cv2.bitwise_not(mask)

kernel_opening = np.ones((3,3),np.uint8)
kernel_closing = np.ones((40,40),np.uint8)

#quadrado desenhado tirado manualmente da imagem Pic_2022_11_28_110903_4.bmp
bx = [2277, 2395, 4827, 4890]
by = [281, 3482, 407, 3107]

for i in range(100, 300, 25):
    print("\n\n #=====================================================")
    print(f"Camada {i}")

    print("Abrindo arquivo da camada impressa...")
    printed = io.imread(f"vision/TrabalhoFinal/22-11-03_11-12-42/printed_{i}.jpg")

    print("Aplicando máscara da área de fabricação...")
    masked = cv2.bitwise_and(printed, printed, mask=mask)

    print("Aplicando threshold yen...")
    binary = masked > threshold_yen(masked, 20)
    cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/pre_process/{i}_yen.jpg', img_as_ubyte(binary[:,1990:5140]))

    print("Aplicando operação de abertura...")
    opened = opening(binary, kernel_opening)
    cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/pre_process/{i}_opened.jpg', img_as_ubyte(opened[:,1990:5140]))

    print("Aplicando operação de fechamento...")
    closed = closing(opened, kernel_closing)
    cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/pre_process/{i}_closed.jpg', img_as_ubyte(closed[:,1990:5140]))

    print("Aplicando warp na imagem real...")
    cv_image = img_as_ubyte(closed)
    warped_image = warp(sys.argv[1:],cv_image, bx, by)

    print("Rotacionando imagem real...")
    rotated_image = rotate_image(warped_image, 180.0)
    cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/pre_process/{i}_rot_warped.jpg', rotated_image)

    print("Rotulando a imagem real e obtendo propriedades das regiões conexas...")
    img = img_as_float(rotated_image)
    # Podemos primeiro identificar as componentes conexas com label()
    imagem_rotulada = label(img)
    # Extraímos componentes individuais com regionprops()
    regions = regionprops(imagem_rotulada)
    try:
        # Tomamos a única região
        props = regions[0]
        # Coordenadas da bounding box
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        print("Redimensionando imagem real ao seu tamanho da bounding box...")
        warped_rotated_image = warp(sys.argv[1:], rotated_image, bx, by)
        ret, warped_rotated_image = cv2.threshold(warped_rotated_image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/pre_process/{i}_warped_rotated_image.jpg', warped_rotated_image)

        print("Preparando imagem da camada...")
        print("Abrindo arquivo CLI...")
        arquivo_cli = open("vision/TrabalhoFinal/arquivo.cli", 'r')

        camada = i
        layer = 0.045
        layer_found = False
        stop_search = False
        #List of contours
        polyline = []

        print("Pegando vetores da camada...")
        #The file is composed of lines separated by a return carriage.
        #Readlines returns a list cotaining all those lines as strings.
        #This loop sweeps thorugh all lines.
        for line in arquivo_cli.readlines():
            #Each line is then separated by words spliting the string empty spaces. 
            #The substrings on this list are named here as part.
            for part in line.split():
                #Checks each substring to identify it's type.
                if '$$UNITS/' in part:
                    #This is a conversion value to transform CLI measurement unit to mm.
                    unit = float(part[8:])
                elif '$$LAYER/' in part:
                    if layer_found:
                        #Flag used to stop the for loop.
                        stop_search = True
                    
                    # Computes layer height and compares to desired layer number
                    string_layer_height = part[8:]
                    layer_height = float(string_layer_height)*unit # mm
                    layer_number = int(layer_height/0.045)
                    if (camada == layer_number):
                        layer_found = True
                        print("Camada encontrada!", layer_number)
                
                # This only executes if layer is found and stop search is not enabled.
                elif (('$$POLYLINE/' in part) and layer_found and not stop_search):
                    coordinates_only = part[11:]
                    #Separates each coordinate by a ','
                    tempPolyline = coordinates_only.split(',')
                    # Deletes three numbers that indicates ID, Z orientation and number of hatches
                    del tempPolyline[0:3]
                    # Converts every element from string to float
                    tempPolyline = [float(i) for i in tempPolyline]
                    #Polylines are divided into groups, in that way there are more than one polyline coordinate list.
                    polyline.append(tempPolyline)
            if stop_search:
                break

        #Vertices (X,Y)
        verts = []
        #Codes 
        # Path.MOVETO -> Moves without drawing
        # Path.CLOSEPOLY -> Closes (end) the polygon
        # Path.LINETO -> Draws a line between two subsequent vertices.
        codes = []

        for poly in polyline:
            # Transforms each coordinate from CLI measurement unit to mm
            poly = [(i*unit) for i in poly]

            #Coordinates are organized as (x1,y1,x2,y2,...,xn,yn)
            x = 0
            # Sweeps all polyline coordinates with a while loop.
            while x < (len(poly)/2):
                #Builds initial and final coordinates
                vert = (poly[2*x], poly[(2*x)+1])
                verts.append(vert)
                #First vert
                if x == 0:
                    codes.append(Path.MOVETO)
                #Final vert
                elif x == ((len(poly)/2)-1):
                    codes.append(Path.CLOSEPOLY)
                #All other verts
                else:
                    codes.append(Path.LINETO)
                #Increments index x
                x = x + 1

        rcParams['figure.figsize'] = [32,32]

        path = Path(verts, codes)

        print("Desenhando imagem da camada...")
        fig, ax = plt.subplots()
        #Facecolor is the image filling.
        #lw is the countour size. Normaly it is of 0.1mm for the beam spot size of the fiber laser.
        patch = patches.PathPatch(path, facecolor='black', lw=0.1)
        ax.add_patch(patch)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        #Ignores axis to have a cleaner image
        plt.axis('off')
        #===============================================
        # convert it to an OpenCV image/numpy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        # convert canvas to image
        layer_image = np.array(fig.canvas.get_renderer()._renderer)
        # it still is rgb, convert to opencv's default bgr
        layer_image = cv2.cvtColor(layer_image,cv2.COLOR_RGB2GRAY)
        #===============================================

        layer_image = cv2.bitwise_not(layer_image)
        # Redimensionando a área de 160 x 160 mm
        layer_image = layer_image[320:2880, 320:2880]

        cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/layer_image/layer_image_{i}.jpg', layer_image)

        print("Rotulando a imagem e obtendo propriedades das regiões conexas...")
        img = img_as_float(layer_image)
        # Podemos primeiro identificar as componentes conexas com label()
        imagem_rotulada = label(img)
        # Extraímos componentes individuais com regionprops()
        regions = regionprops(imagem_rotulada)

        # Tomamos a única região
        props = regions[0]
        # Desenhamos o casco convexo
        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        warped_layer_image = warp(sys.argv[1:], layer_image, bx, by)
        warped_layer_image = cv2.resize(warped_layer_image,(warped_rotated_image.shape[1],warped_rotated_image.shape[0]),None,None,cv2.INTER_LINEAR)
        ret, warped_layer_image = cv2.threshold(warped_layer_image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/pre_process/{i}_warped_layer_image.jpg', warped_layer_image)

        result = cv2.bitwise_xor(warped_layer_image, warped_rotated_image)
        cv2.imwrite(f'vision/TrabalhoFinal/22-11-03_11-12-42/intersection/intersection_printed_{i}.jpg', result)

        white_pixels_layer_image = np.sum(warped_layer_image == 255)
        white_pixels_layer_result = np.sum(result == 255)
        percentage = (white_pixels_layer_result/white_pixels_layer_image)*100
        print(f"Diferença entre arquivo e imagem da camada {i} é de ", round(percentage, 2),'%')

        csv_writer.writerow([f'{i}', str(round(percentage, 2))])
    except Exception as e:
        print(f"Ocorreu a exceção {e} na camada {i}")

print("Time elapsed:", time()-start)