from funciones import *
import math

'''
#Filter image
#imagenOriginal = readImageAsGrayscale('./images/FondoRojo.jpg')
imagenOriginal = readImageAsGrayscale('./images/punto2.jpg')
kernel = crearKernel([[0,0,1], [0,1,0], [0,0,1]])
imagenResultadoArray = filterImage(imagenOriginal, kernel)
guardarArrayComoImagen(imagenResultadoArray, "./images/ImagenModificada3.jpg")

hacerPlot(imagenOriginal, imagenResultadoArray, nBins=256)

#Kernel gaussiano 1D
kernelGauss1D = gaussKernel1D(0.125)
print(kernelGauss1D)

#Kernel gaussiano 2D
# imagenOriginal = readImageAsGrayscale('./images/FondoRojo.jpg')
imagenOriginal = readImageAsGrayscale('./images/punto2.jpg')
imagenModificada = gaussianFilter(imagenOriginal, 0.625)
guardarArrayComoImagen(imagenModificada, "./images/ImagenModificada4.jpg")

hacerPlot(imagenOriginal, imagenModificada, nBins=256)
'''

#Filtro de medianas
#imagenOriginal = readImageAsGrayscale('./images/Noise_salt_and_pepper.png')
imagenOriginal = readImageAsGrayscale('./images/grid.png')
imagenModificada = medianFilter(imagenOriginal, 3)
hacerPlot(imagenOriginal, imagenModificada, nBins=256)  

#Filtro de medianas
#imagenOriginal = readImageAsGrayscale('./images/Noise_salt_and_pepper.png')
imagenOriginal = readImageAsGrayscale('./images/grid.png')
imagenModificada = medianFilter(imagenOriginal, 5)
hacerPlot(imagenOriginal, imagenModificada, nBins=256)  

#Filtro de medianas
#imagenOriginal = readImageAsGrayscale('./images/Noise_salt_and_pepper.png')
imagenOriginal = readImageAsGrayscale('./images/grid.png')
imagenModificada = medianFilter(imagenOriginal, 7)
hacerPlot(imagenOriginal, imagenModificada, nBins=256)  
