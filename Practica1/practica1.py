import math
import numpy as np
import cv2
#from matplotlib import pyplot as plt

black = [0,0,0]
white = [255,255,255]
font = cv2.FONT_HERSHEY_PLAIN

def read_image(filename, flagColor):
	if flagColor == True:
		return cv2.imread(filename, cv2.IMREAD_COLOR)
	else:
		return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def display_multiple_images(vim, title = None, col = 3, color = white):
	width_rows = []
	height_rows = []
	
	for i in range(0,len(vim)):
		vim[i] = cv2.copyMakeBorder(vim[i],20,0,4,4,cv2.BORDER_CONSTANT,value=color )		
		
		if title != None:
			cv2.putText(vim[i],title[i],(10,15), font, 1,(0,0,0), 1, 0)

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
				actual_image = cv2.cvtColor(actual_image, cv2.COLOR_GRAY2RGB)

			for row in range(actual_image.shape[0]):
				for col in range(actual_image.shape[1]):
					image_map[inicio_fil+row][inicio_col + col] = actual_image[row][col]
			inicio_col += actual_image.shape[1]
		inicio_fil += height_rows[i//col]

	# Visualization 
	display_image(image_map)

def display_image(im):
	cv2.imshow('Imagen', im)	
	cv2.waitKey(0)		
	cv2.destroyAllWindows()
	

##############################################################################
############################## EJERCICIO 1 ###################################
##############################################################################


#################### EJERCICIO 1.A ###########################
	
image = read_image('imagenes/einstein.bmp', True)
image_gray = read_image('imagenes/einstein.bmp', False)

def gaussian_convolution(img, size, sigma):
	'''
	Esta función aplica a la imagen pasada un filtro gaussiano
	con el size y sigma pasados.
	'''
	img_gb = cv2.GaussianBlur(img,(size,size),sigma)
	return img_gb		

# Different sigma values
images_sigma = [image]
titles_sigma = ['Original']

for i in range(1,6):
	sigma = i*1.0
	images_sigma.append(gaussian_convolution(image, 0, sigma))
	titles_sigma.append('Sigma = ' + str(sigma))

display_multiple_images(images_sigma, titles_sigma, 3)

# Different size values
images_size = [image]
titles_size = ['Original']

for i in range(1,6):
	size = i*6+1
	images_size.append(gaussian_convolution(image, size, 0))
	titles_size.append('Size = ' + str(size))

display_multiple_images(images_size, titles_size, 3)


#################### EJERCICIO 1.B ###########################


image = read_image('imagenes/bicycle.bmp', True)
image_gray = read_image('imagenes/bicycle.bmp', False)

def get_mask(dx,dy, size=3):
	'''	Esta función devuelve las máscaras 1D de derivadas'''
	sobel3x = cv2.getDerivKernels(dx,dy, size, normalize=True)
	return sobel3x[0], sobel3x[1]
	
dx = get_mask(1,0)
dy = get_mask(0,1)
dx5 = get_mask(1,0,5)
	
print('\nDerivada respecto de x (size = 3): \n\nVector 1',dx[0].T, '\nVector 2', dx[1].T, '\nMáscara 2D\n', dx[1].dot(dx[0].T))
print('\nDerivada respecto de y (size = 3): \n\nVector 1',dy[0].T, '\nVector 2', dy[1].T, '\nMáscara 2D\n', dy[1].dot(dy[0].T))
print('\nDerivada respecto de x (size = 5): \n\nVector 1',dx5[0].T, '\nVector 2', dx5[1].T, '\nMáscara 2D\n', dx5[1].dot(dx5[0].T))

input("\n--- Pulsar enter para continuar ---\n")

#################### EJERCICIO 1.C ###########################


image = read_image('imagenes/bicycle.bmp', True)
image_gray = read_image('imagenes/bicycle.bmp', False)

def laplacian(img, sigma, borde):	
	# Aplicamos una convolucion gaussiana para eliminar ruido
	img_sigma = gaussian_convolution(img,0,sigma)
	# Devolvemos el laplaciano de a imagen suavizada
	return cv2.Laplacian(img_sigma, -1, ksize=3)
		
image_laplace1 = laplacian(image, 1, cv2.BORDER_CONSTANT)
image_laplace3 = laplacian(image, 3, cv2.BORDER_REPLICATE)

images_laplacian = [image_laplace1, image_laplace3]
titles_laplacian = ['Lapacian sigma = 1', 'Lapacian sigma = 3']

display_multiple_images(images_laplacian, titles_laplacian, 2)		


##############################################################################
############################## EJERCICIO 2 ###################################
##############################################################################
	
#################### EJERCICIO 2.A ###########################


def separable_convolution(img, kernelx, kernely, border = cv2.BORDER_DEFAULT):
	''' 
	Esta función realiza el cálculo de la convolución con máscaras separables
	'''
	img = cv2.sepFilter2D(img, -1, kernelx, kernely, border)
	return img

def separable_gaussian_convolution(img, size, sigma, border = cv2.BORDER_DEFAULT):
	'''	Filtro gaussiano utilizando convolución separable '''
	kernel = cv2.getGaussianKernel(size,sigma)
	return separable_convolution(img, kernel, kernel, border)

image_gc = gaussian_convolution(image_gray,7,2)	
image_sgc = separable_gaussian_convolution(image_gray, 7, 2, cv2.BORDER_REFLECT)
titles = ['Convolucion mascara 2D', 'Convolucion separable']
display_multiple_images([image_gc, image_sgc],titles)


#################### EJERCICIO 2.B ###########################

image_gray = read_image('imagenes/motorcycle.bmp', False)

def separable_first_derivate(img, dx, dy, size, border):
	'''
	Cálculo de la convolución separable con máscaras de primera	derivada.
	
	Los argumentos dx y dy son booleanos true indica que se realiza 
	la derivada correspondiente. 
	
	Nota: dx y dy no pueden ser falsos a la vez.
	'''
	if dx or dy:	
		kernel = cv2.getDerivKernels(dx*1, dy*1, size)	
		return separable_convolution(img, kernel[0], kernel[1], border)	

image_dx = separable_first_derivate(image_gray, True, False, 3, cv2.BORDER_CONSTANT)
image_dy = separable_first_derivate(image_gray, False, True, 3, cv2.BORDER_CONSTANT)
image_dxdy = separable_first_derivate(image_gray, True, True, 3, cv2.BORDER_CONSTANT)

images = [image_gray, image_dx, image_dy, image_dxdy]
titles = ['Original', 'dx', 'dy', 'dx dy']
display_multiple_images(images, titles, 2)

image3_dx = separable_first_derivate(image_gray, True, False, 3, cv2.BORDER_CONSTANT)
image5_dx = separable_first_derivate(image_gray, True, False, 5, cv2.BORDER_CONSTANT)
image7_dx = separable_first_derivate(image_gray, True, False, 7, cv2.BORDER_CONSTANT)

images = [image_gray, image3_dx, image5_dx, image7_dx]
titles = ['Original', 'dx size mask = 3', 'dx size mask = 5', 'dx size mask = 7']
display_multiple_images(images, titles, 2)
	

#################### EJERCICIO 2.C ###########################	

def separable_second_derivate(img, dx, dy, size, border):
	'''
	Cálculo de la convolución separable con máscaras de segunda	derivada.
	
	Los argumentos dx y dy son booleanos true indica que se realiza 
	la segunda derivada correspondiente. 
	
	Nota: dx y dy no pueden ser falsos a la vez.
	'''
	if dx or dy:	
		kernel = cv2.getDerivKernels(dx*2, dy*2, size)	
		return separable_convolution(img, kernel[0], kernel[1], border)	

image_dx = separable_second_derivate(image_gray, True, False, 3, cv2.BORDER_CONSTANT)
image_dy = separable_second_derivate(image_gray, False, True, 3, cv2.BORDER_CONSTANT)
image_dxdy = separable_second_derivate(image_gray, True, True, 3, cv2.BORDER_CONSTANT)

images = [image_gray, image_dx, image_dy, image_dxdy]
titles = ['Original', 'dx2', 'dy2', 'dx2 dy2']
display_multiple_images(images, titles, 2)

image3_dx = separable_second_derivate(image_gray, True, False, 3, cv2.BORDER_CONSTANT)
image5_dx = separable_second_derivate(image_gray, True, False, 5, cv2.BORDER_CONSTANT)
image7_dx = separable_second_derivate(image_gray, True, False, 7, cv2.BORDER_CONSTANT)

images = [image_gray, image3_dx, image5_dx, image7_dx]
titles = ['Original', 'dx size mask = 3', 'dx size mask = 5', 'dx size mask = 7']
display_multiple_images(images, titles, 2)

#################### EJERCICIO 2.D ###########################

image_gray = read_image('imagenes/cat.bmp', False)

def gaussian_pyramid(img, level=4, border = cv2.BORDER_DEFAULT):
	'''
	Esta función genera la representación de una piramide gaussiana del 
	nivel indicado (por defecto nivel 4)
	'''
	images = [img]
	
	# Almacenamos todas las imagenes reducidas
	for i in range(level):
		images.append(cv2.pyrDown(images[i], borderType = border))
	
	display_multiple_images(images, None, level+1, black)		
	
gaussian_pyramid(image_gray, 4, cv2.BORDER_REFLECT)

#################### EJERCICIO 2.E ###########################

image_gray = read_image('imagenes/motorcycle.bmp', False)

def laplacian_pyramid(img, level=4, border = cv2.BORDER_DEFAULT):
	'''
	Esta función genera la representación de una piramide laplaciana del 
	nivel indicado (por defecto nivel 4)
	'''
	images=[]
	image_actual = img
	image_down = None
	
	# Cálculamos las imagenes y las guardamos en un vector
	for i in range(level):
		w = image_actual.shape[1]
		h = image_actual.shape[0]
		image_down = cv2.pyrDown(image_actual, borderType = border)
		im = cv2.pyrUp(image_down, dstsize=(w, h), borderType = border)
		images.append(cv2.subtract(image_actual, im))
		image_actual = image_down
	
	images.append(image_down)
		
	display_multiple_images(images, None, level+1, black)	
	
laplacian_pyramid(image_gray, 4)


##############################################################################
############################## EJERCICIO 3 ###################################
##############################################################################


def hybrid_images(img1, img2, sigma1 = 1.5, sigma2 = 1.5):	
	# Obtenemos las imagenes de alta frecuencia y baja frecuencia
	high_frec = cv2.subtract(img1, cv2.GaussianBlur(img1,(0,0),sigma1))
	low_frec = cv2.GaussianBlur(img2,(0,0),sigma2)	
		
	# Hacemos una suma ponderada de las imagenes
	hybrid = cv2.addWeighted(high_frec, 0.6, low_frec, 0.4, 50)	
	
	titles = ['Alta frecuencia', 'Baja frecuencia', 'Hibrida']
	display_multiple_images([high_frec, low_frec, hybrid], titles, 3)
	
	
img1 = read_image('imagenes/einstein.bmp', False)
img2 = read_image('imagenes/marilyn.bmp', False)
hybrid_images(img1, img2, 5,3)

img1 = read_image('imagenes/cat.bmp', False)
img2 = read_image('imagenes/dog.bmp', False)
hybrid_images(img1, img2, 4,4)

img1 = read_image('imagenes/plane.bmp', False)
img2 = read_image('imagenes/bird.bmp', False)
hybrid_images(img1, img2, 4,2)

img1 = read_image('imagenes/motorcycle.bmp', False)
img2 = read_image('imagenes/bicycle.bmp', False)
hybrid_images(img1, img2, 4,3)

img1 = read_image('imagenes/fish.bmp', False)
img2 = read_image('imagenes/submarine.bmp', False)
hybrid_images(img1, img2, 4,2)


##############################################################################
################################# BONUS ######################################
##############################################################################

def f(x, sigma):
	return math.exp(-0.5*x**2/(sigma**2))

def gaussian_kernel(sigma):		
	# kernel de tamaño 6*sigma+1 
	kernel = [[f(x,sigma)] for x in range(int(-3*sigma), int(3*sigma)+1)]	
	
	kernel = np.asarray(kernel)	
	kernel = kernel/sum(kernel)
	
	return kernel

sigma = 2
# Kernel 1 con función de cv2 y kernel2 con mi función
kernel1 = cv2.getGaussianKernel(sigma*6+1, sigma)
kernel2 = gaussian_kernel(sigma)

print('Kernel1: ', kernel1.T)
print('Kernel2: ', kernel2.T)

input("\n--- Pulsar enter para continuar ---\n")

######################### BONUS 2 #############################

def convolution_1D(signal, mask):	
	signal_size = len(signal)
	mask_size = len(mask)
	border_size = (mask_size-1)//2
	
	# Creamos los bordes izquierdo y derecho 
	left_border = signal[border_size-1::-1]
	right_border = signal[signal_size:signal_size-border_size-1:-1] 
	# Creamos una extensión de signal añadiendo los bordes
	extension = np.concatenate((left_border,signal, right_border))
	
	convolution = signal.copy()

	for i in range(border_size, border_size+signal_size):
		convolution[i-border_size] = np.dot(extension[i-border_size:i-border_size+mask_size], mask)
	
	return convolution

def convolution_1D_color(signal, mask):
	b, g, r = cv2.split(signal)
	
	b = convolution_1D(b, mask)
	g = convolution_1D(g, mask)
	r = convolution_1D(r, mask)
	
	return cv2.merge([b,g,r])

v = [1,2,3,4,5,6,7,8,9]
m = [-1,0,1]

k = convolution_1D(v,m)

print('Vector original: ', v)
print('Tras aplicar la convolución: ', k)

input("\n--- Pulsar enter para continuar ---\n")
	
######################### BONUS 3 #############################

def truncate_matrix(matrix, down=0, up=255):
	
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if matrix[i,j] > up:
				matrix[i,j] = up
			elif matrix[i,j] < down:
				matrix[i,j] = down			
				
	return matrix

def separable_convolution2(img, kernelx, kernely):	
	img = img.astype(np.float32)
		
	if len(img.shape) != 3:		
		# Iteramos la imagen por filas y aplicamos la convolución con kernelx			
		for i in range(img.shape[0]):
			img[i,:] = convolution_1D(img[i,:], kernelx)			
		# Iteramos la imagen por columnas y aplicamos la convolución con kernely
		for j in range(img.shape[1]):
			img[:,j] = convolution_1D(img[:,j], kernely)
			
		img = truncate_matrix(img)		
	else:		
		b, g, r = cv2.split(img)
		
		for i in range(img.shape[0]):
			b[i,:] = convolution_1D(b[i,:], kernelx)
			g[i,:] = convolution_1D(g[i,:], kernelx)
			r[i,:] = convolution_1D(r[i,:], kernelx)
			
		for j in range(img.shape[1]):
			b[:,j] = convolution_1D(b[:,j], kernely)
			g[:,j] = convolution_1D(g[:,j], kernely)
			r[:,j] = convolution_1D(r[:,j], kernely)
			
		b = truncate_matrix(b)
		g = truncate_matrix(g)
		r = truncate_matrix(r)
		
		img = cv2.merge([b,g,r])
		
	return img

dx = cv2.getDerivKernels(1, 0, 3)

im1 = separable_convolution2(image, dx[0], dx[1])
im2 = separable_first_derivate(image, True, False, 3, cv2.BORDER_CONSTANT)
titles = ['Convolucion propia', 'Convolucion opencv']
display_multiple_images([im1, im2], titles, 2)
