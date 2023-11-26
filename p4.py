#DETECCION DE BORDES
from funciones import *

def magnitud(inImage, inImageX, inImageY):
    imageHeight, imageWidth = inImage.shape

    imagenMagnitud = np.arange(0, imageHeight*imageWidth, 1, np.float64)
    imagenMagnitud = np.reshape(imagenMagnitud, [imageHeight, imageWidth])
     
    for x in range(imageHeight):
        for y in range(imageWidth):
            imagenMagnitud[x][y] = math.sqrt(math.pow(inImageX[x][y], 2) + math.pow(inImageY[x][y], 2))
    
    return imagenMagnitud

def orientacion(inImage, inImageX, inImageY):
    imageHeight, imageWidth = inImage.shape

    imagenOrientacion = np.arange(0, imageHeight*imageWidth, 1, np.float64)
    imagenOrientacion = np.reshape(imagenOrientacion, [imageHeight, imageWidth])
     
    for x in range(imageHeight):
        for y in range(imageWidth):
            if (inImageX[x][y] == 0):
                    imagenOrientacion[x][y] = 1
            imagenOrientacion[x][y] = math.degrees(math.atan(inImageY[x][y]/inImageX[x][y]))
    
    return imagenOrientacion

def supresionNoMaxima(inImageMagnitud, inImageOrientacion):
    inImageMagnitudHeigth, inImageMagnitudWidth = inImageMagnitud.shape

    imagenResultado = np.arange(0, inImageMagnitudHeigth*inImageMagnitudWidth, 1, np.float64)
    imagenResultado = np.reshape(imagenResultado, [inImageMagnitudHeigth, inImageMagnitudWidth])

    paddedImage = np.pad(inImageMagnitud, 1, mode='constant', constant_values=0)

    for y in range(inImageMagnitudHeigth):
        for x in range(inImageMagnitudWidth):
            posY = y+1
            posX = x+1

            angulo = inImageOrientacion[y][x]

            # Determinar los píxeles vecinos para cada direccion del gradiente
            if (0 <= angulo < 22.5) or (157.5 <= angulo <= 180):
                vecinos = [paddedImage[posY][posX-1], paddedImage[posY][posX+1]]
            elif (22.5 <= angulo < 67.5):
                vecinos = [paddedImage[posY-1][posX-1], paddedImage[posY+1][posX+1]]
            elif (67.5 <= angulo < 112.5):
                vecinos = [paddedImage[posY-1][posX], paddedImage[posY+1][posX]]
            else:  # 112.5 <= angle < 157.5
                vecinos = [paddedImage[posY-1][posX+1], paddedImage[posY+1][posX-1]]

            # Comparar la intensidad del píxel actual con los vecinos a lo largo de la dirección del gradiente
            if (paddedImage[posY][posX] < vecinos[0] or paddedImage[posY][posX] < vecinos[1]):
                imagenResultado[y][x] = 0
            else:
                imagenResultado[y][x] = paddedImage[posY][posX]

    return imagenResultado

def umbralizacionHisteresis(image, low_threshold, high_threshold):
    # Etapa 1: Aplicar umbral alto para identificar píxeles de borde fuertes
    strong_edges = (image >= high_threshold)

    # Etapa 2: Aplicar umbral bajo para identificar píxeles de borde débiles
    weak_edges = (image >= low_threshold) & (image < high_threshold)

    # Etapa 3: Conectar píxeles de borde fuertes y débiles para formar bordes continuos
    result = np.zeros_like(image, dtype=np.float64)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if strong_edges[i, j]:
                result[i, j] = 1
            elif weak_edges[i, j] and np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                result[i, j] = 1

    return result

def edgeCanny(inImage, sigma, tLow, tHigh):
    # Suavizar imagen y eliminar ruido (Mejora de la imagen)
    imagenSuavizada = gaussianFilter(inImage, sigma) # Suavizado gausiano
    
    # Localizar los bordes en la imagen mejorada (Mejora de la imagen)
    imagenSuavizadaX, imagenSuavizadaY = gradientImage(imagenSuavizada, "Sobel") # Calcular las componentes del gradiente (Jx, Jy)
    imagenMagnitud = magnitud(imagenSuavizada, imagenSuavizadaX, imagenSuavizadaY) # Calcular la magnitud de los bordes
    imagenMagnitudNormalizada = adjustIntensity(imagenMagnitud, [], [])
    imagenOrientacion = orientacion(imagenSuavizada, imagenSuavizadaX, imagenSuavizadaY) # Calcular la orientacion de los bordes (Grados)
    
    # Producir bordes de 1 pixel de grosor (Supresion no maxima)
    imagenSuprimida = supresionNoMaxima(imagenMagnitudNormalizada, imagenOrientacion)

    # Reducir la probabilidad de falsos contornos (Umbralizacion con histeresis)
    imagenUmbralizada = umbralizacionHisteresis(imagenSuprimida, tLow, tHigh)
    #imagenUmbralizada = filters.apply_hysteresis_threshold(imagenSuprimida, tLow, tHigh)
    return imagenUmbralizada

imagenOriginal = readImageAsGrayscale('./images/imagenesDePrueba/circles1.png')

#gx, gy = gradientImage(imagenOriginal, "Sobel")
#imagenModificada = LoG(imagenOriginal, 1.5)
imagenModificada = edgeCanny(imagenOriginal, 0.2, 0.08, 0.036)

#circles1.png
#Sigma: 0.2
#TLow: 0.08
#THigh: 0.036

hacerPlot(imagenOriginal, imagenModificada, nBins=256)
