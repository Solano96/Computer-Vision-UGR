#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
import random
import pickle
from matplotlib import pyplot as plt
import os
import collections
from collections import Counter, defaultdict
from PIL import Image
import time
from sklearn import svm
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


black = [0,0,0]
white = [255,255,255]
font = cv.FONT_HERSHEY_PLAIN

np.random.seed(1024)
random.seed(1024)

#np.set_printoptions(threshold=np.nan)

''' ------------------------------ FUNCIONES UTILES ------------------------------ '''

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
    

''' --------------- HOG IMPLEMENTATION --------------- '''


def gamma_correction(image, gamma):
    image = np.float32(image)/255.0
    image = np.power(image, gamma, dtype = float)
    image = (image*255).astype('uint8')
    return image


# Función para obtener magnitud y ángulo del gradiente de la imagen. La imagen es RGB 
# luego hay 3 gradientes para cada pixel nos quedaremos con el de mayor magnitud
def get_gradients_largest_norm(img):

    # Obtenemos las derivadas parciales
    dx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
    dy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)

    # Obtenemos la magnitud y la orientación
    mag, angle = cv.cartToPolar(dx, dy, angleInDegrees=True)

    mag_final = np.zeros([mag.shape[0],mag.shape[1]])
    angle_final = np.zeros([angle.shape[0],angle.shape[1]])

    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            max_index = np.argmax(mag[i][j])
            mag_final[i][j] = mag[i][j][max_index]
            angle_final[i][j] = angle[i][j][max_index]

    return mag_final, angle_final

#Función para obtener el histograma de una celda
def get_histogram(mag, angle):
    h_size = 9
    histogram = np.zeros(h_size, dtype=np.float32)

    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            index_1 = (int)(angle[i,j]//20)%h_size
            index_2 = (index_1+1)%h_size

            alfa = (20-angle[i,j]%20)/20

            histogram[index_1] += alfa*mag[i,j]
            histogram[index_2] += (1-alfa)*mag[i,j]

    return histogram


# Función para dividir la imagen en celdad de cell_size x cell_size
def get_cells(mag, angle, cell_size):

    n_fil = mag.shape[0]//cell_size
    n_col = mag.shape[1]//cell_size

    cells = np.zeros((n_fil,n_col,9))

    for i in range(n_fil):
        for j in range(n_col):
            i_start, j_start = i*cell_size, j*cell_size
            i_end, j_end = (i+1)*cell_size, (j+1)*cell_size

            mag_cell = mag[i_start:i_end, j_start:j_end]       
            angle_cell = angle[i_start:i_end, j_start:j_end]  

            histogram_cell = get_histogram(mag_cell, angle_cell)

            cells[i,j] = histogram_cell

    return cells

def L2_epsilon(block, epsilon):

    norm = np.sqrt(np.sum(block * block) + epsilon * epsilon)

    if norm == 0:
        return block
    else:
        return block / norm

def normalize_vector(block, norm):

    norm = 1

    if norm == 0:
        return block
    else:
        return block / norm

    if norm == 'L2_epsilon':
        norm = np.sqrt(np.sum(block*block) + 0.1*0.1)
    else:
        norm = np.linalg.norm(block, ord = norm)

    if norm == 0:
        return block
    else:
        return block / norm


def block_normalization(cells, block_size = 2, norm = None):

    blocks = np.zeros((cells.shape[0]+1-block_size, cells.shape[1]+1-block_size, cells.shape[2]*block_size*block_size))

    for i in range(cells.shape[0]+1-block_size):
        for j in range(cells.shape[1]+1-block_size):
            block = []

            for i_b in range(block_size):
                for j_b in range(block_size):
                    block.extend(cells[i+i_b, j+j_b])

            blocks[i,j] = normalize_vector(np.array(block), norm)

    return blocks


num_iter = 0
num_images = 0

def HOG(image, gamma = 1, cell_size = 8, block_size = 2, norm = None):
    global num_iter
    global num_images

    if num_iter%100 == 0:
        print(round(100*num_iter/num_images,2),'%')

    num_iter+=1

    # Redimensionamos la imagen a 64x128
    image_resize = cv.resize(image, (64,128))

    # Correción gamma
    image_gamma_correction = gamma_correction(image_resize, gamma)

    # Obtenemos magnitud y ángulo de los gradientes
    mag, angle = get_gradients_largest_norm(image_gamma_correction)

    # Creamos las celdas de histogramas
    cells = get_cells(mag, angle, cell_size)

    # Normalizamos por bloques
    blocks = block_normalization(cells, block_size, norm)

    return blocks


def load_images_from_folder(folder):
    images = [cv.imread(os.path.join(folder,filename)) for filename in os.listdir(folder)]
    return images


def HOG_features(folder, gamma = 1, cell_size = 8, block_size = 2, norm = None):
    global num_iter
    global num_images

    print('Cargando imágenes para entrenamiento...')

    print('  Cargando casos positivos...')
    images_train_pos = load_images_from_folder('./INRIAPerson/train_64x128_H96/pos')
    
    print('  Cargando casos negativos...')
    images_train_neg_files = load_images_from_folder('./INRIAPerson/train_64x128_H96/neg')

    images_train_neg = []
    for img in images_train_neg_files:
        for i in range(4):
            row = random.randint(0, np.shape(img)[0] - 128)
            col = random.randint(0, np.shape(img)[1] - 64)
            sub_img = img[row:row + 128, col:col + 64]
            images_train_neg.append(sub_img)
    
    print('Cargando imágenes para test...')

    print('  Cargando casos positivos...')
    images_test_pos = load_images_from_folder('./INRIAPerson/test_64x128_H96/pos')

    print('  Cargando casos negativos...')
    images_test_neg_files = load_images_from_folder('./INRIAPerson/test_64x128_H96/neg')
    
    images_test_neg = []
    for img in images_test_neg_files:
        for i in range(4):
            row = random.randint(0, np.shape(img)[0] - 128)
            col = random.randint(0, np.shape(img)[1] - 64)
            sub_img = img[row:row + 128, col:col + 64]
            images_test_neg.append(sub_img)
    
    print('HOG POS TRAIN')
    num_iter = 0
    num_images = len(images_train_pos)
    hog_positives_train = np.array([HOG(img, gamma, cell_size, block_size, norm) for img in images_train_pos])
    pickle.dump(hog_positives_train, open("./" + folder + "/hog_positives_train.p", "wb"))
    
    print('HOG NEG TRAIN')
    num_iter = 0
    num_images = len(images_train_neg)
    hog_negatives_train = np.array([HOG(img, gamma, cell_size, block_size, norm) for img in images_train_neg])
    pickle.dump(hog_negatives_train, open("./" + folder + "/hog_negatives_train.p", "wb"))

    print('HOG POS TEST')
    num_iter = 0
    num_images = len(images_test_pos)
    hog_positives_test = np.array([HOG(img, gamma, cell_size, block_size, norm) for img in images_test_pos])
    pickle.dump(hog_positives_test, open("./" + folder + "/hog_positives_test.p", "wb"))

    print('HOG NEG TEST')
    num_iter = 0
    num_images = len(images_test_neg)
    hog_negatives_test = np.array([HOG(img, gamma, cell_size, block_size, norm) for img in images_test_neg])
    pickle.dump(hog_negatives_test, open("./" + folder + "/hog_negatives_test.p", "wb"))
    

def train(data_folder, model_folder, clf):

    ''' -------- TRAIN DATA SET -------- '''

    # lectura datos train
    print('Lectura train...')
    hog_train_pos = pickle.load(open("./" + data_folder + "/hog_positives_train.p", "rb"))
    hog_train_neg = pickle.load(open("./" + data_folder + "/hog_negatives_train.p", "rb"))

    # unir datos train
    X_train_temp = np.append(hog_train_pos, hog_train_neg, axis=0)

    X_train = []

    for i in range(len(X_train_temp)):
        x = []
        for j in range(len(X_train_temp[i])):
            for k in range(len(X_train_temp[i][j])):
                x.extend(X_train_temp[i][j][k])
        X_train.append(x)

    X_train = np.array(X_train)

    # creamos etiquetas
    y_train_pos = np.ones(len(hog_train_pos))
    y_train_neg = np.zeros(len(hog_train_neg))
    y_train = np.append(y_train_pos, y_train_neg)

    ''' -------- TEST DATA SET -------- '''

    # lectura datos test
    print('Lectura test...')
    hog_test_pos = pickle.load(open("./" + data_folder + "/hog_positives_test.p", "rb"))
    hog_test_neg = pickle.load(open("./" + data_folder + "/hog_negatives_test.p", "rb"))

    # unir datos test
    X_test_temp = np.append(hog_test_pos, hog_test_neg[0:5000], axis=0)

    X_test = []

    for i in range(len(X_test_temp)):
        x = []
        for j in range(len(X_test_temp[i])):
            for k in range(len(X_test_temp[i][j])):
                x.extend(X_test_temp[i][j][k])
        X_test.append(x)

    X_test = np.array(X_test)

    # creamos etiquetas
    y_test_pos = np.ones(len(hog_test_pos))
    y_test_neg = np.zeros(len(hog_test_neg[0:5000]))
    y_test = np.append(y_test_pos, y_test_neg)


    print('Entrenando...')
    #clf = svm.SVC(kernel='linear', C=0.01, verbose = True, max_iter = 10000000)
    clf.fit(X_train, y_train)

    filename = 'finalized_model.sav'
    pickle.dump(clf, open("./" + model_folder + ".p", 'wb'))
    print('Entrenamiento finalizado.')

    # Calculamos el score con dicho ajuste para test
    predictions_train = clf.predict(X_train)    
    score_train = clf.score(X_train, y_train)
        
    # Calculamos el score con dicho ajuste para test
    predictions_test = clf.predict(X_test)
    score_test = clf.score(X_test, y_test)

    print('Valor de acierto con el mejor c sobre el conjunto train: ', score_train)
    print('Valor de acierto con el mejor c sobre el conjunto test: ', score_test)


    #Matriz de confusión
    print ('\nMatriz de confusión:')
    cm = metrics.confusion_matrix(y_test, predictions_test)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True);
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score_test)
    plt.title(all_sample_title, size = 10);
    plt.show()

    # Curva roc
    print ('\nCurva ROC:')
    y_pred_rf = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_rf)  
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def main():

    HOG_features('pickle_1', gamma = 0.5, cell_size = 6, block_size = 3, norm = None)
    HOG_features('pickle_2', gamma = 0.5, cell_size = 6, block_size = 3, norm = 1)
    HOG_features('pickle_3', gamma = 0.5, cell_size = 6, block_size = 3, norm = np.inf)
    HOG_features('pickle_4', gamma = 0.5, cell_size = 6, block_size = 3, norm = 'L2_epsilon')

    '''
    HOG_features('pickle1', gamma = 2, cell_size = 8, block_size = 2, norm = None)
    HOG_features('pickle2', gamma = 1, cell_size = 8, block_size = 2, norm = None)
    HOG_features('pickle3', gamma = 0.6, cell_size = 8, block_size = 2, norm = None)
    '''
    '''
    HOG_features('pickle4', gamma = 0.6, cell_size = 6, block_size = 3, norm = None)
    HOG_features('pickle5', gamma = 0.5, cell_size = 6, block_size = 3, norm = None)
    HOG_features('pickle6', gamma = 0.4, cell_size = 6, block_size = 3, norm = None)
    HOG_features('pickle7', gamma = 0.3, cell_size = 6, block_size = 3, norm = None)

    HOG_features('pickle8', gamma = 0.5, cell_size = 6, block_size = 3, norm = 1)
    '''
    '''
    HOG_features('pickle9', gamma = 0.5, cell_size = 6, block_size = 3, norm = 'L2_epsilon')
    HOG_features('pickle10', gamma = 0.5, cell_size = 6, block_size = 3, norm = np.inf)
    '''
    '''
    train('pickle8', 'models/model3', RandomForestClassifier(n_estimators = 125, max_depth = 20))
    '''

if __name__ == "__main__":
    main()


    
