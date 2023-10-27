from funciones import *

#Lee la imagen
imagenOriginalArray = readImageAsGrayscale('./images/FondoRojo.jpg')
print(imagenOriginalArray.shape)

#Procesa la imagen
imagenFinalEQArray = equalizeIntensity(imagenOriginalArray, nBins=256)
imagenFinalArray = adjustIntensity(imagenOriginalArray,[0,1],[0.4,0.9])

#Guarda e imprime las imagenes transformadas
guardarArrayComoImagen(imagenFinalArray, "./images/ImagenModificada.jpg")
hacerPlot(imagenOriginalArray, imagenFinalEQArray, nBins=256)