#OPERADORES MORFOLOGICOS
from funciones import *
'''
imagenOriginal = imagen_erode_1()
elementoEstructurante = crearEE([[1,1,1],[1,1,1],[1,1,1]])
imagenModificada = erode(imagenOriginal, elementoEstructurante, center=[])
hacerPlot(imagenOriginal, imagenModificada, nBins=256)

imagenOriginal = imagen_erode_1()
elementoEstructurante = crearEE([[1,1,1],[1,1,1],[1,1,1]])
imagenModificada = dilate(imagenOriginal, elementoEstructurante, center=[])
hacerPlot(imagenOriginal, imagenModificada, nBins=256)

imagenOriginal = imagen_erode_1()
elementoEstructurante = crearEE([[1,1,1],[1,1,1],[1,1,1]])
imagenModificada = opening(imagenOriginal, elementoEstructurante, center=[])
hacerPlot(imagenOriginal, imagenModificada, nBins=256)

imagenOriginal = imagen_erode_1()
elementoEstructurante = crearEE([[1,1,1],[1,1,1],[1,1,1]])
imagenModificada = closing(imagenOriginal, elementoEstructurante, center=[])
hacerPlot(imagenOriginal, imagenModificada, nBins=256)

imagenOriginal = readImageAsGrayscale('./images/morfo.png')

#Detecta esquinas inferiores izquierdas
elementoEstructurante1 = crearEE([[0,1,0],[0,1,1],[0,0,0]])
elementoEstructuranteFondo1 = crearEE([[0,0,0],[1,0,0],[1,1,0]])

#Detecta esquinas inferiores derechas
elementoEstructurante2 = crearEE([[0,1,0],[1,1,0],[0,0,0]])
elementoEstructuranteFondo2 = crearEE([[0,0,0],[0,0,1],[0,1,1]])

#Detecta esquinas superiores derechas
elementoEstructurante3 = crearEE([[0,0,0],[1,1,0],[0,1,0]])
elementoEstructuranteFondo3 = crearEE([[0,1,1],[0,0,1],[0,0,0]])

#Detecta esquinas superiores izquierdas
elementoEstructurante4 = crearEE([[0,0,0],[0,1,1],[0,1,0]])
elementoEstructuranteFondo4 = crearEE([[1,1,0],[1,0,0],[0,0,0]])

imagenModificada1 = hit_or_miss(imagenOriginal, elementoEstructurante1, elementoEstructuranteFondo1, center=[])
imagenModificada2 = hit_or_miss(imagenOriginal, elementoEstructurante2, elementoEstructuranteFondo2, center=[])
imagenModificada3 = hit_or_miss(imagenOriginal, elementoEstructurante3, elementoEstructuranteFondo3, center=[])
imagenModificada4 = hit_or_miss(imagenOriginal, elementoEstructurante4, elementoEstructuranteFondo4, center=[])

hacerPlot(imagenOriginal, imagenModificada1, nBins=256)
hacerPlot(imagenOriginal, imagenModificada2, nBins=256)
hacerPlot(imagenOriginal, imagenModificada3, nBins=256)
hacerPlot(imagenOriginal, imagenModificada4, nBins=256)
'''

imagenOriginal = readImageAsGrayscale('./images/morph.png')
imagenOriginal = (imagenOriginal > 0.5) *1 
elementoEstructurante = crearEE([[1,0,0],[0,1,0],[0,0,0]])
elementoEstructuranteFondo = crearEE([[0,1,0],[0,0,1],[0,0,0]])
imagenModificada4 = hit_or_miss(imagenOriginal, elementoEstructurante, elementoEstructuranteFondo, center=[])
#imagenModificada4 = erode(imagenOriginal, elementoEstructurante, center=[])
hacerPlot(imagenOriginal, imagenModificada4, nBins=256)

