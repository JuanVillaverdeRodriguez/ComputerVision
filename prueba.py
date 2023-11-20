# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0, 2 * np.pi, 200)
# y = np.sin(x)

# fig, ax = plt.subplots()
# ax.plot(x, y)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# make data:
x = 0.5 + np.arange(8)
y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]

# plot
fig, ax = plt.subplots()

ax.bar(x, y, width=1, edgecolor="green", linewidth=0.7)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()

# def invertirImagen(inImage):
#     array = np.arange(0, h*w, 1, np.float64)
#     array = np.reshape(array, inImage.shape)

#     for height in range(h):
#         for widht in range(w):
#             array[height][widht] = 1 - inImage[height][widht]

#     for height in range(h):
#         for widht in range(w):
#             array[height][widht] = array[height][widht]*255
#             print(array[height][widht])
    
#     data = im.fromarray(array)
#     if data.mode != 'RGB':
#         new_p = data.convert('RGB')
#         new_p.save('./images/gfg_dummy_pic2.png')

# #invertirImagen(image)
# #ski.io.imshow(imagenInvertida)
# #ski.io.show()

#
#lista = []
#for x in range(0,101):
#    lista.append([x, 0])
#
#for height in range(h):
#    for widht in range(w):
#        lista[(int(round(float(str(imageArray[height][widht])), 2)*100))][1] += 1
#
#print(lista)

# def adjustIntensity(inImage, inRange, outRange):
#     Gnorm = np.arange(0, h*w, 1, np.float64)
#     Gnorm = np.reshape(Gnorm, inImage.shape)
    
#     for height in range(h):
#         for widht in range(w):
#             Gnorm[height][widht] = modHistogram(inImage[height][widht],inRange[0], inRange[1],outRange[0], outRange[1])

#     #for height in range(h):
#         #for widht in range(w):
#             #Gnorm[height][widht] = Gnorm[height][widht]*255
#             #print(Gnorm[height][widht])
#     return Gnorm
#     data = im.fromarray(Gnorm)
    
#     if data.mode != 'RGB':
#         new_p = data.convert('RGB')
#         new_p.save('./images/PeepoModificado.jpg')


tam = 50
img1 = np.arange(0, tam*tam, 1, np.float64)
img1 = np.reshape(img1, [tam, tam])

for height in range(tam):
    for widht in range(tam):
        img1[height][widht] = 0

img1[25][25] = 1

tam = 11
kernel2 = np.arange(0, tam*tam, 1, np.float64)
kernel2 = np.reshape(kernel2, [tam, tam])

for height in range(tam):
    for widht in range(tam):
        kernel2[height][widht] = 0

kernel2[5][5] = 1

#Kernel 4x3
#kernel = crearKernel([[0,1,0], [1,1,1], [0,1,0], [1,1,1]])

#Kernel 3x3
#kernel = crearKernel([[0,1,0], [1,1,1], [0,1,0]])

#Kernel 4x4
#kernel = crearKernel([[0,1,0,1], [1,1,1,1], [0,1,0,1], [1,1,1,1]])

#Kernel 4x5
#kernel = crearKernel([[0,1,0,1,1], [1,1,1,1,1], [0,1,0,1,1], [1,1,1,1,1]])

#Kernel 5x4
#kernel = crearKernel([[0,1,0,1], [1,1,1,1], [0,1,0,1], [1,1,1,1], [1,1,1,1]])

#kernel 7x7
#kernel = crearKernel([[0,1,0,1,1,1,1], [1,1,1,1,1,1,1], [0,1,0,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]])

#kernel 7x6
#kernel = crearKernel([[0,1,0,1,1,1], [1,1,1,1,1,1], [0,1,0,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1], [1,1,1,1,1,1]])

#kernel 6x7
#kernel = crearKernel([[0,1,0,1,1,1,1], [1,1,1,1,1,1,1], [0,1,0,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]])


#Kernel 6x5
#kernel = crearKernel([[0,1,0,1,1], [1,1,1,1,1], [0,1,0,1,1], [1,1,1,1,1], [1,1,1,1,1], [1,1,1,1,1]])

#Kernel 2x1
#kernel = crearKernel([[1], [1]])

#print(kernel)

tam = 10
imagenOriginal = np.arange(0, tam*tam, 1, np.uint8)
imagenOriginal = np.reshape(imagenOriginal, [tam, tam])

for height in range(tam):
    for widht in range(tam):
        imagenOriginal[height][widht] = 0

imagenOriginal[5][5] = 255

tamY = 5
tamX = 4
imagenOriginal = np.arange(0, tamY*tamX, 1, np.float64)
imagenOriginal = np.reshape(imagenOriginal, [tamY, tamX])
for height in range(tamY):
    for widht in range(tamX):
        imagenOriginal[height][widht] = 0
imagenOriginal[0][0] = 1
imagenOriginal[1][0] = 1
imagenOriginal[2][1] = 1
imagenOriginal[2][2] = 1
imagenOriginal[3][1] = 1
imagenOriginal[4][1] = 1

# elementoEstructurante1 = crearEE([[-1,1,-1],[0,1,1],[0,0,-1]])
# elementoEstructuranteFondo1 = crearEE([[-1,0,-1],[1,0,0],[1,1,-1]])

# elementoEstructurante2 = crearEE([[-1,1,-1],[1,1,0],[-1,0,0]])
# elementoEstructuranteFondo2 = crearEE([[-1,0,-1],[0,0,1],[-1,1,1]])

# elementoEstructurante3 = crearEE([[-1,0,0],[1,1,0],[-1,1,-1]])
# elementoEstructuranteFondo3 = crearEE([[-1,1,1],[0,0,1],[-1,0,-1]])

# elementoEstructurante4 = crearEE([[0,0,-1],[0,1,1],[-1,1,-1]])
# elementoEstructuranteFondo4 = crearEE([[1,1,-1],[1,0,0],[-1,0,-1]])