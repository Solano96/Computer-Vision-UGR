
import cv2 as cv
import pickle
import numpy as np
import random
from auxFunc import *
from matplotlib import pyplot as plt
import os
import collections
from collections import Counter, defaultdict

black = [0,0,0]
white = [255,255,255]
font = cv.FONT_HERSHEY_PLAIN

np.random.seed(1024)
random.seed(1024)

def read_image(filename, flagColor):
    if flagColor == True:
        return cv.imread(filename, cv.IMREAD_COLOR)
    else:
        return cv.imread(filename, cv.IMREAD_GRAYSCALE)

def display_multiple_images(vim, title = None, col = 3, color = white):
    width_rows = []
    height_rows = []
    
    for i in range(0,len(vim)):
        vim[i] = cv.copyMakeBorder(vim[i],20,0,4,4,cv.BORDER_CONSTANT,value=color )     
        
        if title != None:
            cv.putText(vim[i],title[i],(10,15), font, 1,(0,0,0), 1, 0)

    for i in range(0,len(vim),col):
        width_rows.append(sum( [ im.shape[1] for im in vim[i:min(i+col,len(vim))] ] ))
        height_rows.append(max( [ im.shape[0] for im in vim[i:min(i+col,len(vim))] ] ))

    # ancho total del mapa de imagenes
    width = max(width_rows)
    # altura total del mapa de imagenes
    height = sum(height_rows)

    # creación de la matriz para el mapa de imagenes
    image_map = np.zeros( (height, width,3) , np.uint8)

    # rellenamos el mapa
    inicio_fil = 0
    for i in range(0,len(vim),col):
        inicio_col = 0
        # rellenamos una fila de imagenes
        for k in range(i,min(i+col,len(vim))):
            actual_image = vim[k]

            if len(actual_image.shape) < 3:
                actual_image = cv.cvtColor(actual_image, cv.COLOR_GRAY2RGB)

            for row in range(actual_image.shape[0]):
                for col in range(actual_image.shape[1]):
                    image_map[inicio_fil+row][inicio_col + col] = actual_image[row][col]
            inicio_col += actual_image.shape[1]
        inicio_fil += height_rows[i//col]

    # Visualization 
    display_image(image_map)

def display_image(im):
    cv.imshow('Imagen', im)
    k = cv.waitKey(0)   
    cv.destroyAllWindows()
    


############################ EJERCICIO 1 ############################


def get_keyPoints_and_descriptors(img, mask=None):
    sift = cv.xfeatures2d.SIFT_create() 

    return sift.detectAndCompute(img, mask)

def getMatchesKNN(desc_1, desc_2, k_=2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc_1, desc_2, k=k_)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            
    return good


# Función para calcular las correspondencias de las imágenes, pero la primera de ello con una máscara.
def draw_matches_mask(img1,img2,points):
    np.set_printoptions(threshold=np.nan)
    # Mascara
    mask = np.zeros(shape=(np.shape(img1)[0], np.shape(img1)[1]))    
    cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), color=1)

    # Descriptores
    kp1, desc1 = get_keyPoints_and_descriptors(img1, mask=np.array(mask, dtype=np.uint8))
    kp2,desc2 = get_keyPoints_and_descriptors(img2)

    # Correspondencias
    matches = getMatchesKNN(desc1, desc2)

    # Dibujamos correspondencias
    img_matches_2 = cv.drawMatchesKnn(img1, kp1, img2, kp2,matches1to2=matches,outImg=None,flags=2)
    display_image(img_matches_2)


print("EJERCICIO 1")

# prueba 1
print("Prueba 1")
img1 = cv2.imread("imagenes/64.png")
img2 = cv2.imread("imagenes/65.png")
points = [(470,125), (620, 125), (620, 330), (470, 330)]
draw_matches_mask(img1, img2,points)

# prueba 2
print("Prueba 2")
img3 = cv2.imread("imagenes/57.png")
img4 = cv2.imread("imagenes/58.png")
points_2 = [(10,90), (120,90), (120,220), (10,220)]
draw_matches_mask(img3, img4, points_2)

# prueba 3
print("Prueba 3")
img5 = cv2.imread("imagenes/54.png")
img6 = cv2.imread("imagenes/55.png")
points_3 = [(40,90), (250,90), (250,240), (40,240)]
draw_matches_mask(img5, img6, points_3)


############################ EJERCICIO 2 ############################

print("EJERCICIO 2")

# Se guardan los nombres de los diccionarios
dictionary_name = "imagenes/kmeanscenters2000.pkl"
# Se guarda el nombre del fichero que contiene los descriptores y parches
descriptors_and_patches_name = "imagenes/descriptorsAndpatches2000.pkl"

descriptors, patches = loadAux(descriptors_and_patches_name, True)
accuracy, labels, dictionary = loadDictionary(dictionary_name)


def get_histogram(img_name, dictionary, dictionary_norm):
    # Leer imagen
    img = cv2.imread("imagenes/"+img_name)
    # Obtener descriptores
    kp, desc = get_keyPoints_and_descriptors(img)
    desc_norm = np.apply_along_axis(np.linalg.norm, 1, desc)

    # Calcular similaridad y normalizar
    similarities = np.dot(dictionary, desc.T)
    similarities_norm = np.divide(similarities, desc_norm*dictionary_norm[:,None])

    histogram = Counter(np.argmax(similarities_norm, axis=0))

    return histogram


def get_inverted_file_index(dictionary_name):

    accuracy, labels, dictionary = loadDictionary(dictionary_name)
    dictionary_norm = np.apply_along_axis(np.linalg.norm, 1, dictionary)
    histograms = []

    inverted_file = collections.defaultdict(list)

    for img in range(441):
        img_name = str(img) + ".png"
        histogram = get_histogram(img_name, dictionary, dictionary_norm)
        histograms.append(histogram)

        for i in histogram:
            inverted_file[i].append(img)

    return dict(inverted_file), histograms


def retrieval_images(image_name, inverted_file, histograms):

    histogram = histograms[int(os.path.splitext(image_name)[0])]
    histogram_image = [histogram[k] for k in range(2000)]

    images = []

    for i in histogram:
        for img in inverted_file[i]:
            images.append(img)

    # Obtenemos las imágenes junto al número de apariciones
    images_names, _ = np.unique(images, return_counts=True)

    values = []

    for i in images_names:
        histogram_images_i = [histograms[int(i)][k] for k in range(2000)]

        similarity = np.dot(histogram_images_i, histogram_image)
        n1 = np.linalg.norm(histogram_images_i)
        n2 = np.linalg.norm(histogram_image)
        similarity = similarity/(n1*n2)

        values.append(similarity)

    values = np.array(values)
    index_sort = np.argsort(-values)

    images_names_sorted = images_names[index_sort]

    display_image(cv2.imread("imagenes/"+image_name))

    for i in images_names_sorted[0:5]:        
        display_image(cv2.imread("imagenes/"+str(i)+".png"))

print("Obteniendo indice invertido...")
inverted_file, histograms = get_inverted_file_index(dictionary_name)
print("Obtenido con éxito.")

print("Imagen-pregunta 1")
retrieval_images("353.png", inverted_file, histograms)

print("Imagen-pregunta 2")
retrieval_images("4.png", inverted_file, histograms)

print("Imagen-pregunta 3")
retrieval_images("115.png", inverted_file, histograms)