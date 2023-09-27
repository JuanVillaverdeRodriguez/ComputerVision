# Librearias utiles => matplotlib
import skimage as ski
from PIL import Image as im
import numpy as np

image = ski.io.imread('./images/PeepoOriginal.jpg')

image = ski.color.rgb2gray(image)
ski.io.imshow(image)
#ski.io.show()
image = ski.util.img_as_float(image)
h, w = image.shape
print(image.shape)
print(image.dtype)
lista = []
for x in range(0,101):
    lista.append([x, 0])

for height in range(h):
    for widht in range(w):
        lista[(int(round(float(str(image[height][widht])), 2)*100))][1] += 1

print(lista)

def invertirImagen(inImage):
    array = np.arange(0, h*w, 1, np.float64)
    array = np.reshape(array, inImage.shape)

    for height in range(h):
        for widht in range(w):
            array[height][widht] = 1 - inImage[height][widht]

    for height in range(h):
        for widht in range(w):
            array[height][widht] = array[height][widht]*255
            print(array[height][widht])
    
    data = im.fromarray(array)
    if data.mode != 'RGB':
        new_p = data.convert('RGB')
        new_p.save('./images/gfg_dummy_pic2.png')

#invertirImagen(image)
#ski.io.imshow(imagenInvertida)
#ski.io.show()
def modHistogram(xy,GminNorm, GmaxNorm, Gmin,Gmax):
    return (GminNorm + ((GmaxNorm-GminNorm)*(xy-Gmin)/(Gmax-Gmin)))

def adjustIntensity(inImage, inRange, outRange):
    Gnorm = np.arange(0, h*w, 1, np.float64)
    Gnorm = np.reshape(Gnorm, inImage.shape)
    
    for height in range(h):
        for widht in range(w):
            Gnorm[height][widht] = modHistogram(inImage[height][widht],inRange[0], inRange[1],outRange[0], outRange[1])

    for height in range(h):
        for widht in range(w):
            Gnorm[height][widht] = Gnorm[height][widht]*255
            print(Gnorm[height][widht])

    data = im.fromarray(Gnorm)
    if data.mode != 'RGB':
        new_p = data.convert('RGB')
        new_p.save('./images/PeepoModificado.jpg')

adjustIntensity(image,[0,1],[0.4,0.9])