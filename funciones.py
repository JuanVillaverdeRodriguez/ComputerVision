import skimage as ski
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from skimage import data, filters
import matplotlib.image

################################################################################
####################### HISTOGRAMAS: MEJORAS DE CONTRASTE #############################
################################################################################
#Lee y convierte a escala de grises la imagen pasada por parametro
def readImageAsGrayscale(inImageRuta):
    with im.open(inImageRuta) as img:
        img = img.convert('L')  # Convierte la imagen a escala de grises
        image = np.array(img)  # Convierte la imagen en un array NumPy

    return ski.img_as_float64(image)   

#Crea el histograma de una imagen
def crearHistograma(inImage, nBins):
    h, w = inImage.shape

    print(h)
    print(w)

    aux = []
    for x in range(0,nBins+1):
        aux.append([x, 0])

    print(len(aux))
    for height in range(h):
        for widht in range(w):
            print(inImage[height][widht])
            aux[int(inImage[height][widht])][1] += 1

    histograma = []
    for x in aux:
        histograma.append(x[1])

    return histograma

def modHist(xy,Gmin, Gmax, GminNorm,GmaxNorm):
    return (GminNorm + ((GmaxNorm-GminNorm)*(xy-Gmin)/(Gmax-Gmin)))

def getMinMax(inImage):
    h, w = inImage.shape
    
    min = inImage[0][0]
    max = inImage[0][0]
    for height in range(h):
        for widht in range(w):
            if (inImage[height][widht] <= min): min = inImage[height][widht]
            if (inImage[height][widht] >= max): max = inImage[height][widht]
    
    return [min, max]
            
def adjustIntensity(inImage, inRange, outRange):    
    h, w = inImage.shape

    Gnorm = np.arange(0, h*w, 1, np.float64)
    Gnorm = np.reshape(Gnorm, inImage.shape)
    
    if (inRange == []):
        inRange = getMinMax(inImage)
    
    if (outRange == []):
        outRange = [0,1]
    
    for height in range(h):
        for widht in range(w):
            Gnorm[height][widht] = modHist(inImage[height][widht],inRange[0], inRange[1],outRange[0], outRange[1])

    return Gnorm

def hacerPlot(inImageOriginal, inImageModificada, nBins):
    plt.style.use('_mpl-gallery')

    #Divide la pantalla de plot en una cuadricula 2x2
    fig, ax = plt.subplots(2,2, figsize=(10, 6)) 

    #1º) Plot de la imagen original
    ax.ravel()[0].imshow(inImageOriginal, cmap='gray')
    ax.ravel()[0].set_title("Imagen original")
    ax.ravel()[0].set_axis_off()
    
    #2º) Plot de la imagen modificada
    ax.ravel()[1].imshow(inImageModificada, cmap='gray')
    ax.ravel()[1].set_title("Imagen modificada")
    ax.ravel()[1].set_axis_off()

    #3º) Plot del histograma de la imagen original
    ax.ravel()[2].hist(inImageOriginal.flatten(), nBins)
    ax.ravel()[2].set_title("Histograma imagen original")
    
    #4º) Plot del histograma de la imagen modificada
    ax.ravel()[3].hist(inImageModificada.flatten(), nBins)
    ax.ravel()[3].set_title("Histograma imagen modificada")

    plt.tight_layout() #Ajuste del padding
    plt.show() #Mostrar imagen

def modHistAcumulado(histAcumulado, pixel, nBins, h, w):
    return ((histAcumulado[pixel]/(h*w))*nBins)

def guardarArrayComoImagen(arrayEntrada, rutaImagenSalida):
    #data = im.fromarray(arrayEntrada, mode='L')
    #data.save(rutaImagenSalida)
    # h, w = arrayEntrada.shape
    # imagen = im.new("L", (w, h))
    # imagen.putdata(arrayEntrada.flatten())
    # imagen.save(rutaImagenSalida)
    matplotlib.image.imsave(rutaImagenSalida, arrayEntrada, cmap='gray')

def equalizeIntensity(inImage, nBins=256):
    h, w = inImage.shape
    
    Gnorm = np.arange(0, h*w, 1, np.float64)
    Gnorm = np.reshape(Gnorm, inImage.shape)

    imagenAjustada = adjustIntensity(inImage, [0, 1], [0, nBins])
    
    #Vector de tamaño h*w que contiene la intensidad de cada pixel
    arrayPixelesImagen = imagenAjustada.flatten()
    histograma = crearHistograma(imagenAjustada, nBins)

    histAcumulado = []
    sumatorio = 0

    for val in histograma:
        sumatorio+=val
        histAcumulado.append(sumatorio)

    pixel = 0
    for height in range(h):
         for widht in range(w):
             intensidadPixel = arrayPixelesImagen[pixel]
             Gnorm[height][widht] = modHistAcumulado(histAcumulado, int(intensidadPixel), nBins, h, w)
             pixel += 1
    
    GnormNormalizado = adjustIntensity(Gnorm, [0, nBins], [0, 1])
    return GnormNormalizado

################################################################################
####################### FUCIONES FILTRADO ESPACIAL #############################
################################################################################
def es_lista_dos_dimensiones(lista):
    if isinstance(lista, list):
        # Verificar si al menos un elemento interno no es una lista
        return all(isinstance(sublista, list) for sublista in lista)
    return False

def crearKernel(inLista):
    if (es_lista_dos_dimensiones(inLista)):
        h = len(inLista)
        w = len(inLista[0])

        kernel = np.arange(0, h*w, 1, np.float64)
        kernel = np.reshape(kernel, [h, w])

        for height in range(h):
            for widht in range(w):
                kernel[height][widht] = inLista[height][widht]

        return kernel
    else: 
        longitud = len(inLista)

        kernel = np.arange(0, longitud, 1, np.float64)
        kernel = np.reshape(kernel, -1)

        for val in range(longitud):
            kernel[val] = inLista[val]

        return kernel
    
def gaussKernel1D(sigma):
    N = (2*round((3*sigma)+0.5)) + 1
    centro = (math.floor(N/2) + 1)

    listaKernel = []
    x = -(N - centro)
    for posX in range(N):
        calculo = (math.e**((-(x**2))/(2*(sigma**2))))/(math.sqrt(2*math.pi)*sigma)
        x +=1
        listaKernel.append(calculo)

    return crearKernel(listaKernel)

def filterImage(inImage, kernel):
    outImage =  np.copy(inImage)

    kernel = np.rot90(kernel,2)
    imageHeight, imageWidht = inImage.shape
    kernelHeight, kernelWidht = kernel.shape

    resultingImageArray = np.copy(inImage)
    
    centrokernelHeight = math.floor(kernelHeight/2)+1
    centrokernelWidht = math.floor(kernelWidht/2)+1

    if (centrokernelWidht >= kernelHeight):
        mayor = centrokernelWidht
    else:
        mayor = centrokernelHeight

    paddedImage = np.pad(inImage, mayor, mode='constant', constant_values=0)

    #Este bloque sirve para hacer que el trozo de la imagen sobre el que se va a 
    #realizar el producto escalar tenga las mismas dimensiones que el kernel
    limiteHeight = centrokernelHeight
    limiteWidht = centrokernelWidht
    if (kernelHeight % 2) == 0:
        limiteHeight = centrokernelHeight-1
    if (kernelWidht % 2) == 0:
        limiteWidht = centrokernelWidht-1

    for h in range(imageHeight):
        for w in range(imageWidht):
            posY = h+mayor
            posX = w+mayor
            
            mult = (paddedImage[posY-(centrokernelHeight-1):posY + limiteHeight, posX-(centrokernelWidht-1):posX + limiteWidht])*kernel

            resultingImageArray[h][w] = mult.sum()

    return resultingImageArray

def gaussianFilter(inImage, sigma):
    kernelGauss1D = gaussKernel1D(sigma)
    a = kernelGauss1D.shape

    kernelGauss1D = np.reshape(kernelGauss1D, (1,a[0]))
    kernelGauss1DTraspuesto = np.rot90(kernelGauss1D,1)

    imagenFiltradaUnaVez = filterImage(inImage, kernelGauss1D)
    imagenFiltradaDosVeces = filterImage(imagenFiltradaUnaVez, kernelGauss1DTraspuesto)

    return imagenFiltradaDosVeces

def medianFilter(inImage, filterSize):
    outImage =  np.copy(inImage)
    imageHeight, imageWidht = inImage.shape
    
    resultingImageArray = np.copy(inImage)
    
    centroKernel = math.floor(filterSize/2)+1

    paddedImage = np.pad(inImage, filterSize, mode='constant', constant_values=0)

    #Este bloque sirve para hacer que el trozo de la imagen sobre el que se va a 
    #realizar el producto escalar tenga las mismas dimensiones que el kernel
    limite = centroKernel
    if (filterSize % 2) == 0:
        limite = centroKernel-1

    for h in range(imageHeight):
        for w in range(imageWidht):
            posY = h+filterSize
            posX = w+filterSize
            
            vecindario = (paddedImage[posY-(centroKernel-1):posY + limite, posX-(centroKernel-1):posX + limite])

            resultingImageArray[h][w] = np.median(vecindario)

    return resultingImageArray

################################################################################
####################### FUCIONES OPERADORES MORFOLOGICOS #######################
################################################################################
def crearEE(EE):
    return crearKernel(EE)

def erode(inImage, SE, center=[]):
    imageHeight, imageWidht = inImage.shape
    SEHeight, SEWidht = SE.shape

    kernelHeight, kernelWidht = SE.shape

    resultingImageArray = np.copy(inImage)
    if (len(center) == 0):
        centrokernelHeight = math.floor(kernelHeight/2)+1
        centrokernelWidht = math.floor(kernelWidht/2)+1
    else:
        centrokernelHeight = center[0] +1
        centrokernelWidht = center[1] +1


    if (centrokernelWidht >= kernelHeight):
        mayor = centrokernelWidht
    else:
        mayor = centrokernelHeight

    paddedImage = np.pad(inImage, mayor, mode='constant', constant_values=0)

    #Este bloque sirve para hacer que el trozo de la imagen sobre el que se va a 
    #realizar el producto escalar tenga las mismas dimensiones que el kernel
    limiteHeight = centrokernelHeight
    print(centrokernelHeight)
    limiteWidht = centrokernelWidht
    print(centrokernelWidht)

    if (kernelHeight % 2) == 0:
        limiteHeight = centrokernelHeight-1
    if (kernelWidht % 2) == 0:
        limiteWidht = centrokernelWidht-1

    print(paddedImage)

    imagenYKernelCoincidenEnAlgo = False
    for h in range(imageHeight):
        for w in range(imageWidht):
            posY = h+mayor
            posX = w+mayor
            print("HOLA")
            #-centrokernelHeight, limiteHeight
            for i in range(kernelHeight):
                if imagenYKernelCoincidenEnAlgo == True:
                    break
                for j in range(kernelWidht):
                    if round(SE[i][j]) != round(paddedImage[posY-(centrokernelHeight-1)+i][posX-(centrokernelWidht-1)+j]):
                        if (round(SE[i][j]) == 1):
                            resultingImageArray[h][w] = 0
                            imagenYKernelCoincidenEnAlgo = True

            if imagenYKernelCoincidenEnAlgo != True:
                resultingImageArray[h][w] = 1
            imagenYKernelCoincidenEnAlgo = False    
                    
    
    return resultingImageArray

def dilate(inImage, SE, center=[]):
    imageHeight, imageWidht = inImage.shape
    SEHeight, SEWidht = SE.shape

    kernelHeight, kernelWidht = SE.shape

    resultingImageArray = np.copy(inImage)
    if (len(center) == 0):
        centrokernelHeight = math.floor(kernelHeight/2)+1
        centrokernelWidht = math.floor(kernelWidht/2)+1
    else:
        centrokernelHeight = center[0] +1
        centrokernelWidht = center[1] +1


    if (centrokernelWidht >= kernelHeight):
        mayor = centrokernelWidht
    else:
        mayor = centrokernelHeight

    paddedImage = np.pad(inImage, mayor, mode='constant', constant_values=0)

    #Este bloque sirve para hacer que el trozo de la imagen sobre el que se va a 
    #realizar el producto escalar tenga las mismas dimensiones que el kernel
    limiteHeight = centrokernelHeight
    print(centrokernelHeight)
    limiteWidht = centrokernelWidht
    print(centrokernelWidht)

    if (kernelHeight % 2) == 0:
        limiteHeight = centrokernelHeight-1
    if (kernelWidht % 2) == 0:
        limiteWidht = centrokernelWidht-1

    print(paddedImage)

    imagenYKernelCoincidenEnAlgo = False
    for h in range(imageHeight):
        for w in range(imageWidht):
            posY = h+mayor
            posX = w+mayor
            print("HOLA")
            #-centrokernelHeight, limiteHeight
            for i in range(kernelHeight):
                if imagenYKernelCoincidenEnAlgo == True:
                    break
                for j in range(kernelWidht):
                    print("SE i j: ")
                    print(i)
                    print(j)
                    print(round(SE[i][j]))
                    print("PADDEDIMAGE posY posX: ")
                    print(posY-(centrokernelHeight-1)+i)
                    print(posX-(centrokernelWidht-1)+j)
                    print(round(paddedImage[posY-(centrokernelHeight-1)+i][posX-(centrokernelWidht-1)+j]))
                    print("---------------------------")

                    if round(SE[i][j]) == round(paddedImage[posY-(centrokernelHeight-1)+i][posX-(centrokernelWidht-1)+j]):
                        print("NO SON IGUALES")
                        resultingImageArray[h][w] = 1
                        imagenYKernelCoincidenEnAlgo = True

            if imagenYKernelCoincidenEnAlgo != True:
                resultingImageArray[h][w] = 0
            imagenYKernelCoincidenEnAlgo = False    
                    
    
    return resultingImageArray

def opening(inImage, SE, center=[]):
    imagenErosionada = erode(inImage, SE, center)
    imagenDilatada = dilate(imagenErosionada, SE, center)
    
    return imagenDilatada

def closing(inImage, SE, center=[]):
    imagenDilatada = dilate(inImage, SE, center)
    imagenErosionada = erode(imagenDilatada, SE, center)
    
    return imagenErosionada
    
def invertirImagen(inImage):
    inImageCopy = np.copy(inImage)
    h, w = inImageCopy.shape
    for y in range(h):
        for x in range(w):
            if inImageCopy[y][x] == 0:
                inImageCopy[y][x] = 1
            else: inImageCopy[y][x] = 0

    return inImageCopy

def intersect(inImage1, inImage2):
    inImageCopy = np.copy(inImage1)
    h, w = inImageCopy.shape
    for y in range(h):
        for x in range(w):
            if inImageCopy[y][x] != inImage2[y][x]:
                inImageCopy[y][x] = 0

    return inImageCopy

def comprobarPosicionesSE(obSEj, bgSE):
    h, w = obSEj.shape
    h2, w2 = bgSE.shape

    if (h != h2 or w != w2):
        return False
    
    for y in range(h):
        for x in range(w):
            if obSEj[y][x] == bgSE[y][x]:
                if obSEj[y][x] == 1:
                    return False

    return True
    
def hit_or_miss(inImage, objSEj, bgSE, center=[]):
    #Comprobar si hay unos en las mismas posiciones o no
    if comprobarPosicionesSE(objSEj, bgSE) == False:
        print("ERROR: Elemento estructurantes incoherentes")
        return 
    
    #Obtener lo que no pertenece a la figura.
    imagenInvertida = invertirImagen(inImage)

    #Calcular A (inImage) erosinado con objSEj
    primeraErosion = erode(inImage, objSEj, center)

    #Calcular Ac (imagenInvertida) erosionado con bgSE
    segundaErosion = erode(imagenInvertida, bgSE, center)

    #Calcular interseccion 
    interseccion = intersect(primeraErosion, segundaErosion)

    return interseccion

#############################################################################
####################### FUNCIONES DETECCIÓN DE BORDES #######################
#############################################################################

def gradientImage(inImage, operator):
    match operator:
        case "Roberts":
            kernel_x = crearKernel([[-1, 0], [0, 1]])
            kernel_y = crearKernel([[0, -1], [1, 0]])
        case "CentralDiff":
            kernel_x = crearKernel([[-1, 0, 1]])
            kernel_y = crearKernel([[-1], [0], [1]])
        case "Prewitt":
            kernel_x = crearKernel([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
            kernel_y = crearKernel([[-1,-1,-1], [0,0,0], [1,1,1]])
        case "Sobel":
            kernel_x = crearKernel([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
            kernel_y = crearKernel([[-1,-2,-1], [0,0,0], [1,2,1]])
        case _:
            print("Operador [" + operator + "] no reconocido")

    gx = filterImage(inImage, kernel_x)
    gy = filterImage(inImage, kernel_y)

    return gx, gy

def buscarCrucePorCero(inImage, x, y, umbral):
    for x2 in range(-1,1):
        for y2 in range(-1,1):
            if (x2 == y2):
                continue
            if (inImage[x+x2][y+y2] > umbral):
                return 1
            
    return inImage[x][y]

def LoG (inImage, sigma):
    # Convolucionar inImage con el kernel gausiano
    imgConvolGaussianKernel = gaussianFilter(inImage, sigma)

    # Convolucionar imgConvolGaussianKernel con diferencias finitas en x dos veces (2º derivada en x)
    kernel_x = crearKernel([[-1, 0, 1]])
    gx1 = filterImage(imgConvolGaussianKernel, kernel_x)
    gx2 = filterImage(gx1, kernel_x)

    # Convolucionar imgConvolGaussianKernel con diferencias finitas en x dos veces (2º derivada en x)
    kernel_y = crearKernel([[-1], [0], [1]])
    gy1 = filterImage(imgConvolGaussianKernel, kernel_y)
    gy2 = filterImage(gy1, kernel_y)


    # Sumar las dos derivadas segundas (gx2, gy2) en una unica imagen
    imageHeight, imageWidth = gx2.shape

    imagenUnica = np.arange(0, imageHeight*imageWidth, 1, np.float64)
    imagenUnica = np.reshape(imagenUnica, [imageHeight, imageWidth])


    for x in range(imageHeight):
        for y in range(imageWidth):
            imagenUnica[x][y] = gx2[x][y] + gy2[x][y]


    # Buscar cruces por cero

    paddedImage = np.pad(imagenUnica, 1, mode='constant', constant_values=0)

    imagenResultado = np.arange(0, imageHeight*imageWidth, 1, np.float64)
    imagenResultado = np.reshape(imagenResultado, [imageHeight, imageWidth])

    umbral = 0.03
    for x in range(imageHeight):
        for y in range(imageWidth):
            posY = y+1
            posX = x+1
            if (paddedImage[posX][posY] < umbral):
                imagenResultado[x][y] = buscarCrucePorCero(paddedImage, posX, posY, umbral)

    return imagenResultado

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

#################################################################################################
#################### FUNCIONES PARA CREAR IMAGENES DE PRUEBA BINARIAS ###########################
#################################################################################################
def imagen_erode_1():
    tamY = 16
    tamX = 16
    imagenOriginal = np.arange(0, tamY*tamX, 1, np.float64)
    imagenOriginal = np.reshape(imagenOriginal, [tamY, tamX])
    for height in range(tamY):
        for widht in range(tamX):
            imagenOriginal[height][widht] = 0
            
    imagenOriginal[2][3] = 1
    imagenOriginal[2][4] = 1
    imagenOriginal[2][5] = 1

    imagenOriginal[3][2] = 1
    imagenOriginal[3][3] = 1
    imagenOriginal[3][4] = 1
    imagenOriginal[3][5] = 1
    imagenOriginal[3][6] = 1

    imagenOriginal[4][2] = 1
    imagenOriginal[4][3] = 1
    imagenOriginal[4][4] = 1
    imagenOriginal[4][5] = 1
    imagenOriginal[4][6] = 1
    imagenOriginal[4][11] = 1
    imagenOriginal[4][12] = 1
    imagenOriginal[4][13] = 1

    imagenOriginal[5][2] = 1
    imagenOriginal[5][3] = 1
    imagenOriginal[5][4] = 1
    imagenOriginal[5][5] = 1
    imagenOriginal[5][10] = 1
    imagenOriginal[5][11] = 1
    imagenOriginal[5][12] = 1
    imagenOriginal[5][13] = 1

    imagenOriginal[6][3] = 1
    imagenOriginal[6][4] = 1
    imagenOriginal[6][9] = 1
    imagenOriginal[6][10] = 1
    imagenOriginal[6][11] = 1
    imagenOriginal[6][12] = 1
    imagenOriginal[6][13] = 1

    imagenOriginal[7][8] = 1
    imagenOriginal[7][9] = 1
    imagenOriginal[7][10] = 1
    imagenOriginal[7][11] = 1
    imagenOriginal[7][12] = 1

    imagenOriginal[8][7] = 1
    imagenOriginal[8][8] = 1
    imagenOriginal[8][9] = 1
    imagenOriginal[8][10] = 1
    imagenOriginal[8][11] = 1

    imagenOriginal[9][6] = 1
    imagenOriginal[9][7] = 1
    imagenOriginal[9][8] = 1
    imagenOriginal[9][9] = 1
    imagenOriginal[9][10] = 1

    imagenOriginal[10][5] = 1
    imagenOriginal[10][6] = 1
    imagenOriginal[10][7] = 1
    imagenOriginal[10][8] = 1
    imagenOriginal[10][9] = 1

    imagenOriginal[11][4] = 1
    imagenOriginal[11][5] = 1
    imagenOriginal[11][6] = 1
    imagenOriginal[11][7] = 1
    imagenOriginal[11][8] = 1

    imagenOriginal[12][4] = 1
    imagenOriginal[12][5] = 1
    imagenOriginal[12][6] = 1
    imagenOriginal[12][7] = 1
    imagenOriginal[12][8] = 1
    imagenOriginal[12][9] = 1
    imagenOriginal[12][10] = 1
    imagenOriginal[12][11] = 1

    imagenOriginal[13][4] = 1
    imagenOriginal[13][5] = 1
    imagenOriginal[13][6] = 1
    imagenOriginal[13][7] = 1
    imagenOriginal[13][8] = 1
    imagenOriginal[13][9] = 1
    imagenOriginal[13][10] = 1
    imagenOriginal[13][11] = 1

    imagenOriginal[14][5] = 1
    imagenOriginal[14][6] = 1
    imagenOriginal[14][7] = 1
    imagenOriginal[14][8] = 1
    imagenOriginal[14][9] = 1
    imagenOriginal[14][10] = 1

    return imagenOriginal

def imagen_erode_2():
    tamY = 7
    tamX = 7
    imagenOriginal = np.arange(0, tamY*tamX, 1, np.float64)
    imagenOriginal = np.reshape(imagenOriginal, [tamY, tamX])
    for height in range(tamY):
        for widht in range(tamX):
            imagenOriginal[height][widht] = 0
            
    imagenOriginal[1][3] = 1
    imagenOriginal[1][4] = 1

    imagenOriginal[2][2] = 1
    imagenOriginal[2][3] = 1
    imagenOriginal[2][4] = 1
    imagenOriginal[2][5] = 1

    imagenOriginal[3][2] = 1
    imagenOriginal[3][3] = 1
    imagenOriginal[3][4] = 1
    imagenOriginal[3][5] = 1

    imagenOriginal[4][3] = 1
    imagenOriginal[4][4] = 1

    imagenOriginal[5][3] = 1

    return imagenOriginal

def imagen_erode_3():
    tamY = 16
    tamX = 16
    imagenOriginal = np.arange(0, tamY*tamX, 1, np.float64)
    imagenOriginal = np.reshape(imagenOriginal, [tamY, tamX])
    for height in range(tamY):
        for widht in range(tamX):
            imagenOriginal[height][widht] = 0
            
    imagenOriginal[2][8] = 1
    imagenOriginal[2][9] = 1
    imagenOriginal[2][10] = 1
    imagenOriginal[2][11] = 1
    imagenOriginal[2][12] = 1

    imagenOriginal[3][2] = 1
    imagenOriginal[3][3] = 1
    imagenOriginal[3][4] = 1
    imagenOriginal[3][5] = 1
    imagenOriginal[3][8] = 1
    imagenOriginal[3][9] = 1
    imagenOriginal[3][10] = 1
    imagenOriginal[3][11] = 1
    imagenOriginal[3][12] = 1
    imagenOriginal[3][13] = 1

    imagenOriginal[4][2] = 1
    imagenOriginal[4][3] = 1
    imagenOriginal[4][4] = 1
    imagenOriginal[4][5] = 1
    imagenOriginal[4][8] = 1
    imagenOriginal[4][9] = 1
    imagenOriginal[4][10] = 1
    imagenOriginal[4][11] = 1
    imagenOriginal[4][12] = 1

    imagenOriginal[5][1] = 1
    imagenOriginal[5][2] = 1
    imagenOriginal[5][3] = 1
    imagenOriginal[5][4] = 1
    imagenOriginal[5][5] = 1
    imagenOriginal[5][6] = 1
    imagenOriginal[5][7] = 1
    imagenOriginal[5][8] = 1
    imagenOriginal[5][9] = 1
    imagenOriginal[5][10] = 1
    imagenOriginal[5][11] = 1
    imagenOriginal[5][12] = 1

    imagenOriginal[6][2] = 1
    imagenOriginal[6][3] = 1
    imagenOriginal[6][4] = 1
    imagenOriginal[6][5] = 1
    imagenOriginal[6][6] = 1
    imagenOriginal[6][7] = 1
    imagenOriginal[6][8] = 1
    imagenOriginal[6][9] = 1
    imagenOriginal[6][10] = 1
    imagenOriginal[6][11] = 1

    imagenOriginal[7][2] = 1
    imagenOriginal[7][3] = 1
    imagenOriginal[7][9] = 1
    imagenOriginal[7][10] = 1
    imagenOriginal[7][11] = 1
    imagenOriginal[7][12] = 1

    imagenOriginal[8][2] = 1
    imagenOriginal[8][3] = 1
    imagenOriginal[8][9] = 1
    imagenOriginal[8][10] = 1
    imagenOriginal[8][11] = 1
    imagenOriginal[8][12] = 1

    imagenOriginal[9][3] = 1
    imagenOriginal[9][9] = 1
    imagenOriginal[9][10] = 1
    imagenOriginal[9][11] = 1
    imagenOriginal[9][12] = 1

    imagenOriginal[10][2] = 1
    imagenOriginal[10][3] = 1
    imagenOriginal[10][4] = 1
    imagenOriginal[10][5] = 1
    imagenOriginal[10][6] = 1
    imagenOriginal[10][7] = 1
    imagenOriginal[10][8] = 1
    imagenOriginal[10][9] = 1
    imagenOriginal[10][10] = 1
    imagenOriginal[10][11] = 1

    imagenOriginal[11][2] = 1
    imagenOriginal[11][3] = 1
    imagenOriginal[11][4] = 1
    imagenOriginal[11][5] = 1
    imagenOriginal[11][6] = 1
    imagenOriginal[11][7] = 1
    imagenOriginal[11][8] = 1
    imagenOriginal[11][9] = 1
    imagenOriginal[11][10] = 1

    imagenOriginal[12][2] = 1
    imagenOriginal[12][3] = 1
    imagenOriginal[12][4] = 1
    imagenOriginal[12][5] = 1
    imagenOriginal[12][6] = 1
    imagenOriginal[12][7] = 1
    imagenOriginal[12][8] = 1
    imagenOriginal[12][9] = 1

    imagenOriginal[13][2] = 1
    imagenOriginal[13][3] = 1
    imagenOriginal[13][4] = 1
    imagenOriginal[13][5] = 1
    imagenOriginal[13][6] = 1
    imagenOriginal[13][7] = 1
    imagenOriginal[13][8] = 1

    return imagenOriginal