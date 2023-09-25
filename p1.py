# Librearias utiles => matplotlib
import skimage as ski
from PIL import Image


image = ski.data.coins()
# ... or any other NumPy array!
im = Image.fromarray(image)
im.save('./images/image.jpg')

ski.io.imshow(image)
ski.io.show()

def adjustIntensity(inImage, inRange, outRange):
    print("sda")

adjustIntensity(1,1,1)