#DETECCION DE BORDES
from funciones import *

#imagenOriginal = readImageAsGrayscale('./images/imagenesDePrueba/circles1.png')
imagenOriginal = readImageAsGrayscale('./images/Lenna.png')


#gx, gy = gradientImage(imagenOriginal, "Sobel")
imagenModificada = LoG(imagenOriginal, 1.5)
#imagenModificada = edgeCanny(imagenOriginal, 0.2, 0.08, 0.036)

#circles1.png
#Sigma: 0.2
#TLow: 0.08
#THigh: 0.036

hacerPlot(imagenOriginal, imagenModificada, nBins=256)
