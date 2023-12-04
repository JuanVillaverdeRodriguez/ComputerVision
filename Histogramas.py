from funciones import *

#Lee la imagen
imagenOriginalArray = readImageAsGrayscale('./images/grays.png')
print(imagenOriginalArray.shape)

#Procesa la imagen
#imagenFinalEQArray = equalizeIntensity(imagenOriginalArray, nBins=256)
imagenFinalArray = adjustIntensity(imagenOriginalArray,[],[])

#Guarda e imprime las imagenes transformadas
#guardarArrayComoImagen(imagenFinalEQArray, "./images/ImagenModificada.jpg")
#guardarArrayComoImagen(imagenFinalArray, "./images/ImagenModificada2.jpg")
matplotlib.image.imsave("./images/ImagenModificada2.jpg", imagenFinalArray, cmap='gray', vmin=0, vmax=1)

#hacerPlot(imagenOriginalArray, imagenFinalEQArray, nBins=256)

plt.style.use('_mpl-gallery')

#Divide la pantalla de plot en una cuadricula 2x2
fig, ax = plt.subplots(2,2, figsize=(10, 6)) 
#1º) Plot de la imagen original
ax.ravel()[0].imshow(imagenOriginalArray, cmap='gray')
ax.ravel()[0].set_title("Imagen original")
ax.ravel()[0].set_axis_off()

#2º) Plot de la imagen modificada
ax.ravel()[1].imshow(imagenFinalArray, cmap='gray', vmin = 0, vmax = 1)
ax.ravel()[1].set_title("Imagen modificada")
ax.ravel()[1].set_axis_off()
#3º) Plot del histograma de la imagen original
ax.ravel()[2].hist(imagenOriginalArray.flatten(), 256)
ax.ravel()[2].set_title("Histograma imagen original")

#4º) Plot del histograma de la imagen modificada
ax.ravel()[3].hist(imagenFinalArray.flatten(), 256)
ax.ravel()[3].set_title("Histograma imagen modificada")
plt.tight_layout() #Ajuste del padding
plt.show() #Mostrar imagen