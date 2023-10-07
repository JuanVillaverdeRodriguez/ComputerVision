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