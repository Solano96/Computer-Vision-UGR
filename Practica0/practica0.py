import numpy as np
import cv2
from matplotlib import pyplot as plt

################################# EJERCICIO 1 #################################

def leeimagen(filename, flagColor):
	if flagColor == True:
		return cv2.imread(filename, cv2.IMREAD_COLOR)
	else:
		return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


################################# EJERCICIO 2 #################################

def pintaI(im):
	cv2.imshow('Imagen', im)
	k = cv2.waitKey(0)
	cv2.destroyAllWindows()


################################# EJERCICIO 3 #################################

def pintaMI(vim):
	ancho_filas = []
	altura_filas = []

	for i in range(0,len(vim),3):
		ancho_filas.append(sum( [ im.shape[1] for im in vim[i:min(i+3,len(vim))] ] ))
		altura_filas.append(max( [ im.shape[0] for im in vim[i:min(i+3,len(vim))] ] ))

	# ancho total del mapa de imagenes
	ancho = max(ancho_filas)
	# altura total del mapa de imagenes
	alto = sum(altura_filas)

	# creamos la matriz que será rellenada con las imagenes
	mapa_imagenes = np.zeros((alto,ancho,3), np.uint8)

	# rellenamos el mapa de imagenes
	inicio_fil = 0
	for i in range(0,len(vim),3):
		inicio_col = 0
		# rellenamos una fila de imagenes
		for k in range(i,min(i+3,len(vim))):
			imagen_actual = vim[k]

			if len(imagen_actual.shape) < 3:
				imagen_actual = cv2.cvtColor(imagen_actual,cv2.COLOR_GRAY2RGB)

			# añadimos la imagen a mapa_imagenes
			for fil in range(imagen_actual.shape[0]):
				for col in range(imagen_actual.shape[1]):
					mapa_imagenes[inicio_fil+fil][inicio_col + col] = imagen_actual[fil][col]
			inicio_col += imagen_actual.shape[1]
		inicio_fil += altura_filas[i//3]

	# Visualizamos
	pintaI(mapa_imagenes)


################################# EJERCICIO 4 #################################

def modifyPixelColor(im, pixels, color = [255, 0 , 0]):

	for x, y in pixels:
		im[y,x] = color

	return im


################################# EJERCICIO 5 #################################

def displayImages(images, titles):

	cols = 2
	n_images = len(images)

	fig = plt.figure()

	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image)
		a.set_title(title)
		a.axis('off')
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.show()


img = leeimagen('lena.jpg', True)





