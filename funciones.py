import skimage as ski
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import math
import os

#Lee y convierte a escala de grises la imagen pasada por parametro
# def readImageAsGrayscale(inImageRuta):
    
#     image = ski.io.imread(inImageRuta)
#     if len(image.shape)>=3:
#         image = ski.color.rgb2gray(image)
#     return ski.util.img_as_ubyte(image)
#     #return ski.util.img_as_float64(image)


# def readImageAsGrayscale(inImageRuta):
#     try:
#         with im.open(inImageRuta) as img:
#             img = img.convert('L')  # Convierte la imagen a escala de grises
#             image = np.array(img)  # Convierte la imagen en un array NumPy

#         return ski.img_as_ubyte(image)
#     except Exception as e:
#         print(f"Error al cargar la imagen: {str(e)}")

def readImageAsGrayscale(inImageRuta):
    with im.open(inImageRuta) as img:
        img = img.convert('L')  # Convierte la imagen a escala de grises
        image = np.array(img)  # Convierte la imagen en un array NumPy

    return ski.img_as_float64(image)   


#Crea el histograma de una imagen
def crearHistograma(inImage):
    h, w = inImage.shape

    print(h)
    print(w)

    aux = []
    for x in range(0,256):
        aux.append([x, 0])

    print(len(aux))
    for height in range(h):
        for widht in range(w):
            aux[inImage[height][widht]][1] += 1

    histograma = []
    for x in aux:
        histograma.append(x[1])

    return histograma

def modHist(xy,GminNorm, GmaxNorm, Gmin,Gmax):
    return (GminNorm + ((GmaxNorm-GminNorm)*(xy-Gmin)/(Gmax-Gmin)))

def adjustIntensity(inImage, inRange, outRange):    
    h, w = inImage.shape

    Gnorm = np.arange(0, h*w, 1, np.float64)
    Gnorm = np.reshape(Gnorm, inImage.shape)
    
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
    # data = im.fromarray(arrayEntrada, mode='L')
    # data.save(rutaImagenSalida)
    imagen = im.new("L", (10, 10))
    imagen.putdata(arrayEntrada.flatten())
    imagen.save(rutaImagenSalida)

def equalizeIntensity(inImage, nBins=10):
    h, w = inImage.shape
    
    Gnorm = np.arange(0, h*w, 1, np.uint8)
    Gnorm = np.reshape(Gnorm, inImage.shape)
    
    #Vector de tamaño h*w que contiene la intensidad de cada pixel
    arrayPixelesImagen = inImage.flatten()
    histograma = crearHistograma(inImage)

    histAcumulado = []
    sumatorio = 0

    for val in histograma:
        sumatorio+=val
        histAcumulado.append(sumatorio)

    pixel = 0
    for height in range(h):
         for widht in range(w):
             intensidadPixel = arrayPixelesImagen[pixel]
             Gnorm[height][widht] = modHistAcumulado(histAcumulado, intensidadPixel, nBins, h, w)
             pixel += 1
    
    return Gnorm

#FUNCIONES p2------------------------------
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
            
            # print("AHORA ESTOY EN: ")
            # print(posY)
            # print(posX)
            # print(vecindario)
            # print(np.median(vecindario))
            resultingImageArray[h][w] = np.median(vecindario)

    return resultingImageArray
