# -*- coding: utf-8 -*-
"""
Revised 8/25/2022
Original template:
https://cs.brown.edu/courses/csci1430/2021_Spring/resources/python_tutorial/
Edited by
@author: murali subbarao, sbu, ece
"""

# Read in original RGB image.

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

rgbImage = io.imread('olympics1.jpeg')
(m,n,o) = rgbImage.shape
# Extract color channels.
redChannel = rgbImage[:,:,0] # Red channel
greenChannel = rgbImage[:,:,1] # Green channel
blueChannel = rgbImage[:,:,2] # Blue channel
# Create an all black channel.
allBlack = np.zeros((m, n), dtype=np.uint8)
# Create color versions of the individual color channels.
justRed = np.stack((redChannel, allBlack, allBlack), axis=2)
justGreen = np.stack((allBlack, greenChannel, allBlack),axis=2)
justBlue = np.stack((allBlack, allBlack, blueChannel),axis=2)
# Recombine the individual color channels to create the original RGB image again.
recombinedRGBImage = np.stack(( redChannel, greenChannel, blueChannel),axis=2)
plt.imshow(recombinedRGBImage)
plt.show()
print(justRed.shape)
print(justRed.dtype)
print(justRed.size)
print(justRed[0:5, 0:5, 0])
io.imsave('justRed1.jpg' , justRed)

# image values not normalized to be in 0.0 to 1.0) range
imageFloat1  = rgbImage.astype(np.float32)
plt.imshow(imageFloat1)
plt.show()
print(imageFloat1[0:5, 0:5 , 0])

#normalize the image and display
from skimage import img_as_float
floatImage2 = img_as_float(rgbImage)
plt.imshow(floatImage2)
plt.show()
print(floatImage2[0:5, 0:5, 0])

for i in range(5,10) :
    for j in range(20,25) :
       print(floatImage2[i,j,0])
    print('\n')

#compute the histogram
hist=np.zeros(256, dtype = np.intc)
for i in range(m) :
    for j in range(n) :
        hist[rgbImage[i,j,0]] += 1

#print(hist)

import matplotlib.pyplot as plt
#plot example
x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

#plot the histogram
plt.plot(hist)
plt.show()

plt.imshow(greenChannel, cmap=plt.cm.gray)
plt.show()

import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray

original = data.astronaut()
grayscale = rgb2gray(original)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.show()
