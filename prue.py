def es_lista_dos_dimensiones(lista):
    if isinstance(lista, list):
        # Verificar si al menos un elemento interno no es una lista
        return all(isinstance(sublista, list) for sublista in lista)
    return False

# Ejemplos
lista_una_dimension = [1, 2, 3, 4]
lista_dos_dimensiones = [[1, 2], [3, 4], [5, 6]]

if es_lista_dos_dimensiones(lista_una_dimension):
    print("La lista es de dos dimensiones")
else:
    print("La lista no es de dos dimensiones")

if es_lista_dos_dimensiones(lista_dos_dimensiones):
    print("La lista es de dos dimensiones")
else:
    print("La lista no es de dos dimensiones")


#Amplia el tamaño de la imagen colocando 0 por los bordes, 
#permitiendo que se puedan convolucionar los pixeles exteriores
def resizeImage(inImage, kernel):
    kernelHeight, kernelWidht = kernel.shape
    imageHeight, imageWidht = inImage.shape

    #P(i:j+N, k:k+n)
    
    imagenResized = np.arange(0, ((imageHeight + kernelHeight + 5)*(imageWidht + kernelWidht + 5)), 1, np.float64)
    imagenResized = np.reshape(imagenResized, [(imageHeight + kernelHeight + 5), (imageWidht + kernelWidht + 5)])

    print(imagenResized.shape)

    return imagenResized

def convolucinarImagen(inImage, kernel, h, w):
    kernelHeight, kernelWidht = kernel.shape

    #*******COMPROBAR SI PARA KERNELS 2x2 EL ROUND LO HACE BIEN*********
    #Centro del kernel
    a = math.floor(kernelHeight/2)+1
    b = math.floor(kernelWidht/2)+1

    #print("A y B")
    #print(a)
    #print(b)
    # valorResultante = 0
    # for s in range(a+2):
    #     for t in range(b+2):
    #         #valorResultante += ((inImage[h + (s-1)][w + (t-1)])*(kernel[(s)][t]))
    #         valorResultante += ((inImage[h + (s-1)][w + (t-1)])*(kernel[(s-1)][t-1]))

    # valorResultante = 0
    # for s in range(a+1):
    #     count = 0
    #     #print("\n")
    #     for t in range(b+1):
    #         count += 1
    #         #print(count)
    #         valorResultante += (inImage[h + (s-1)][w + (t-1)])*(kernel[s][t])

    #     valorResultante = 0

    valorResultante = 0
    s = -a-1
    t = -b-1
    for h_conv in range(kernelHeight-1):
        s += 1
        for w_conv in range(kernelWidht-1):
            t += 1

            valorResultante += (inImage[h + (s)][w + (t)])*(kernel[h_conv][w_conv])

    if (valorResultante < 0):
        return 0
    elif (valorResultante > 1):
        return 1
    else:
        return valorResultante


def filterImage(inImage, kernel):
    resizedImage = resizeImage(img1, kernel)

    imageHeight, imageWidht = inImage.shape
    kernelHeight, kernelWidht = kernel.shape

    
    resultingImageArray = inImage
    
    centrokernelHeight = math.floor(kernelHeight/2)+1
    centrokernelWidht = math.floor(kernelWidht/2)+1


    print(centrokernelHeight)
    print(centrokernelWidht)

    for h in range(imageHeight):
        for w in range(imageWidht):
            posY = h+(centrokernelHeight-1 + 2)
            posX = w+(centrokernelWidht-1 + 2)
            print("POS X Y POS Y")
            print(posX)
            print(posY)
            resultingImageArray[h][w] = convolucinarImagen(resizedImage, kernel,posY, posX)
    
    # resultingImageArray = resizedImage
    # for h in range(imageHeight):
    #     for w in range(imageWidht):
    #         valorResultante = 0
    #         for s in range(a+2):
    #             for t in range(b+2):
    #                 valorResultante += ((inImage[h + (s-1)][w + (t-1)])*(kernel[(s)][t]))
    #         if (valorResultante < 0):
    #             resultingImageArray[h][w] = 0
    #         elif (valorResultante > 255):
    #             resultingImageArray[h][w] = 255
    #         else:
    #             resultingImageArray[h][w] = valorResultante
                    
    #print(resultingImageArray)
    return resultingImageArray



def vecindario(posY, posX, paddedImage, filterSize):
    imageHeight, imageWidht = paddedImage.shape
    
    lista = []
    for i in range(-1,2):
        for j in range(-1,2):
            if (i != 0 or j != 0):
                lista.append(paddedImage[posY+i][posX+j])
    
    return np.array(lista)