from funciones import *
import math

'''
#Filter image
imagenOriginal = readImageAsGrayscale('./images/FondoRojo.jpg')
kernel = crearKernel([[0,0,1], [0,1,0], [0,0,1]])

imagenResultadoArray = filterImage(imagenOriginal, kernel)
hacerPlot(imagenOriginal, imagenResultadoArray, nBins=256)

#Kernel gaussiano 1D
kernelGauss1D = gaussKernel1D(0.125)
print(kernelGauss1D)

#Kernel gaussiano 2D
# imagenOriginal = readImageAsGrayscale('./images/FondoRojo.jpg')
imagenOriginal = readImageAsGrayscale('./images/punto2.jpg')

imagenModificada = gaussianFilter(imagenOriginal, 0.625)

hacerPlot(imagenOriginal, imagenModificada, nBins=256)
'''

imagenOriginal = readImageAsGrayscale('./images/punto2.jpg')
imagenModificada = medianFilter(imagenOriginal, 5)
hacerPlot(imagenOriginal, imagenModificada, nBins=256)