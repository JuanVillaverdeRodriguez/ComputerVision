#DETECCION DE BORDES
from funciones import *

imagenOriginal = readImageAsGrayscale('./images/grid.png')
#imagenOriginal = readImageAsGrayscale('./images/Lenna.png')


gx, gy = gradientImage(imagenOriginal, "Sobel")
hacerPlot(gx, gy, nBins=256)

#guardarArrayComoImagen(imagenModificada, "./images/ImagenModificada6.jpg")


'''
imagenModificada = edgeCanny(imagenOriginal, 3, 0.01, 0.01)
guardarArrayComoImagen(imagenModificada, "./images/canny1.jpg")
hacerPlot(imagenOriginal, imagenModificada, nBins=256)

imagenModificada = edgeCanny(imagenOriginal, 3, 0.036, 0.036)
guardarArrayComoImagen(imagenModificada, "./images/canny2.jpg")
hacerPlot(imagenOriginal, imagenModificada, nBins=256)

imagenModificada = edgeCanny(imagenOriginal, 3, 0.08, 0.036)
guardarArrayComoImagen(imagenModificada, "./images/canny2.jpg")
hacerPlot(imagenOriginal, imagenModificada, nBins=256)
'''

#circles1.png
#Sigma: 0.2
#TLow: 0.08
#THigh: 0.036
