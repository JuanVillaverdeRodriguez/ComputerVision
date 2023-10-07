from funciones import *

def convolucionar(inImage, kernel):
    kernelHeight, kernelWidht = kernel.shape

    a = (kernelHeight - 1) / 2
    b = (kernelWidht - 1) / 2
    s = -a
    t = -b
    
    #for x in range(a):
        #for y in range(b):
            

def filterImage(inImage, kernel):
    print("AYUDA")

def crearKernel(inLista):
    h, w = inLista.shape
    kernel = np.arange(0, h*w, 1, np.uint8)
    kernel = np.reshape(kernel, inLista.shape)
    return kernel

#Lee la imagen
imagenOriginalArray = readImageAsGrayscale('./images/FondoRojo.jpg')
print(imagenOriginalArray.shape)

#Define el kernel como un NDArray
kernel = crearKernel([[1,1,1],[1,1,1],[1,1,1]])

