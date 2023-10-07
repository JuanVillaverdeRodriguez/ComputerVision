import skimage as ski
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt

#Lee y convierte a escala de grises la imagen pasada por parametro
def readImageAsGrayscale(inImageRuta):
    image = ski.io.imread(inImageRuta)
    image = ski.color.rgb2gray(image)
    return ski.util.img_as_ubyte(image)

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
    ski.io.imshow(inImageOriginal)

    plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots()
    ax.set_ylim([0, 5000])
    #plt.figure(figsize=(10,6))
    plt.hist(inImageModificada.flatten(), nBins)
    fig, ax = plt.subplots()
    ax.set_ylim([0, 5000])

    plt.hist(inImageOriginal.flatten(), nBins)

    plt.show()

def modHistAcumulado(histAcumulado, pixel, nBins, h, w):
    return ((histAcumulado[pixel]/(h*w))*nBins)

def guardarArrayComoImagen(arrayEntrada, rutaImagenSalida):
    data = im.fromarray(arrayEntrada).convert('L')
    data.save(rutaImagenSalida)

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
