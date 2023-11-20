#DETECCION DE BORDES
from funciones import *



imagenOriginal = readImageAsGrayscale('./images/Lenna.png')

#gx, gy = gradientImage(imagenOriginal, "Sobel")
imagenModificada = LoG(imagenOriginal, 4)

hacerPlot(imagenOriginal, imagenModificada, nBins=256)
