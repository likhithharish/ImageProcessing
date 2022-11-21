import numpy as np
from skimage import io, color
import math
import matplotlib.pyplot as plt
def displayNsaveImage(image,imageName):
    plt.title(imageName)
    plt.imshow(image)
    plt.show()
    io.imsave(imageName+'Solution.jpg', image)

A = io.imread('shed1-small.jpeg')
(a,b,c) = A.shape
RC = A[:,:,0]
GC = A[:,:,1]
BC = A[:,:,2]
def rgb2gray(image):
    grayImage = np.rint((RC*0.333+GC*0.333+BC*0.333))
    return grayImage

TE =  50 #int(input('Enter Edge threshold value: '))
AE = rgb2gray(A)
AG = rgb2gray(A)
GX = [[0 for i  in range(a)] for j in range(b)]
GY = [[0 for x in range(a)] for y in range(b)]
GM = [[0 for x in range(a)] for y in range(b)]
for i in range(a):
    for j in range(b-1):
        GX[i][j] = AG[i,j+1] - AG[i,j]

for i in range(a-1):
    for j in range(b):
            GY[i][j] = AG[i+1,j] - AG[i,j]

for i in range(a):
    for j in range(b):
        GM[i][j] = math.sqrt(GX[i][j]**2+GY[i][j]**2)

for i in range(a):
    for j in range(b):
        if GM[i][j]>TE:
            AE[i,j]=255
        else:
            AE[i,j]=0

displayNsaveImage(AE,'Simple Edge Detection AE')