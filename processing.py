import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import math

def displayNsaveImage(image,imageName):
    plt.title(imageName)
    plt.imshow(image)
    plt.show()
    io.imsave(imageName+'Solution.jpg', image)

######## Task Step 1 ########
A = io.imread('shed1-small.jpeg')
(a,b,c) = A.shape
print("Original Image size :",A.shape)
displayNsaveImage(A,'Original Image A')

######## Task Step 2 - Extract channels and save individual images ########
RC = A[:,:,0]
GC = A[:,:,1]
BC = A[:,:,2]

allBlack = np.zeros((a, b), dtype=np.uint8)
justRed = np.stack((RC, allBlack, allBlack), axis=2)
justGreen = np.stack((allBlack, GC, allBlack),axis=2)
justBlue = np.stack((allBlack, allBlack, BC),axis=2)

displayNsaveImage(justRed,'Red Channel Image RC')
displayNsaveImage(justGreen,'Green Channel Image - GC')
displayNsaveImage(justBlue,'Blue Channel image - BC')


######## Task 3 - GrayScale Image ########
def rgb2gray(image):
    grayImage = np.rint((RC*0.333+GC*0.333+BC*0.333))
    return grayImage

AG = rgb2gray(A)
plt.title('GreyLevel Image AG')
plt.imshow(AG,cmap=plt.get_cmap('gray'))
plt.show()


######## Task 4 - Compute Histograms of RC,GC,BC and AG ########
def plotHist(image,a,b,color,title):
    pixelCount = {}
    for i in range(256):
        pixelCount[i] = 0

    for i in range(a):
        for j in range(b):
            pixelCount[image[i,j]] += 1

    print(pixelCount)
    plt.title(title)
    plt.bar(pixelCount.keys(),pixelCount.values(),color=color)
    plt.show()

plotHist(RC,a,b,'red','RED CHANNEL RC HISTOGRAM')
plotHist(GC,a,b,'green','GREEN CHANNEL GC HISTOGRAM')
plotHist(BC,a,b,'blue','BLUE CHANNEL BC HISTOGRAM')
plotHist(AG,a,b,'grey','GREY LEVEL AG HISTOGRAM')

######## Task 5 - Binarizing the image ########
#TB = input('Enter threshold Brightness: ') =--------- UNCOMMENT THIS LINE WHILE SUBMIT -----------
TB = 100
AB = rgb2gray(A)
for i in range(a):
    for j in range(b):
        if AB[i,j]<TB:
            AB[i,j]=0
        else:
            AB[i,j]=255

io.imsave('BinaryImage<100.jpg',AB)
plt.title('Binarizing the Image')
plt.imshow(AB, cmap=plt.get_cmap('gray'))
plt.show()

##### TASK 6 - SIMPLE EDGE DETECTION ######
TE = 75 #int(input('Enter Edge threshold value: '))
AE = rgb2gray(A)
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

######## Task 7 - Downsampling ########
def downSample(image):
    resultImage = rgb2gray(A)
    for i in range(a):
        for j in range(b):
            resultImage[i,j]=np.rint(image[i,j]+image[i,j+1]+image[i+1,j]+image[i+1,j+1])*0.25

    plt.imshow(resultImage)
    plt.show()

AG2 = downSample(AG)
