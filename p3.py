#OPERADORES MORFOLOGICOS
from funciones import *

imagenOriginal = readImageAsGrayscale('./images/morfo.png')
#imagenOriginal = imagen_erode_2()

# elementoEstructurante = crearEE([[0,1], [1,0], [1,0]])
# elementoEstructurante = crearEE([[1,1]])
# elementoEstructurante = crearEE([[1,0], [1,0]])
# elementoEstructurante = crearEE([[1,1,1],[1,1,1],[1,1,1]])

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

# print(elementoEstructurante)


# imagenModificada = erode(imagenOriginal, elementoEstructurante, center=[])
# imagenModificada = dilate(imagenOriginal, elementoEstructurante, center=[])
# imagenModificada = opening(imagenOriginal, elementoEstructurante, center=[])
# imagenModificada = closing(imagenOriginal, elementoEstructurante, center=[])
imagenModificada1 = hit_or_miss(imagenOriginal, elementoEstructurante1, elementoEstructuranteFondo1, center=[])
imagenModificada2 = hit_or_miss(imagenOriginal, elementoEstructurante2, elementoEstructuranteFondo2, center=[])
imagenModificada3 = hit_or_miss(imagenOriginal, elementoEstructurante3, elementoEstructuranteFondo3, center=[])
imagenModificada4 = hit_or_miss(imagenOriginal, elementoEstructurante4, elementoEstructuranteFondo4, center=[])

hacerPlot(imagenOriginal, imagenModificada1, nBins=256)
hacerPlot(imagenOriginal, imagenModificada2, nBins=256)
hacerPlot(imagenOriginal, imagenModificada3, nBins=256)
hacerPlot(imagenOriginal, imagenModificada4, nBins=256)
