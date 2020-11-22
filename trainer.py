import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import os
import statistics

directory = '/Users/davidgiansanti/Desktop/images/'
database = []
Features=[]
for photo in os.listdir(directory):
    print(photo)
    count = 0 
    pname = photo.replace('.bmp', "")
    # photo = "/Users/davidgiansanti/Desktop/images/w.bmp"
    img = io.imread(directory + photo);
    #print(img.shape)
    
    io.imshow(img)
    plt.title('Original Image')
    io.show()
    
    hist = exposure.histogram(img)
    plt.bar(hist[1], hist[0])
    plt.title('Histogram')
    plt.show()
    
    th = 200
    img_binary = (img < th).astype(np.double)
    io.imshow(img_binary)
    plt.title('Binary Image')
    io.show();
    
    img_label = label(img_binary, background = 0)
    
    io.imshow(img_label)
    plt.title('Labeled image')
    io.show()
    
    #print(np.amax(img_label))
    
    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    
    count = 0
    Char = []
    temp = []
    for props in regions:
        cnr = 0
        minr,minc, maxr,maxc = props.bbox
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr,cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        Features.append(hu)
        Char.append(hu)
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr,
        fill = False, edgecolor = 'red', linewidth = 1))
        #Char.append(pname)
    ax.set_title('Bounding Boxes')
    io.show()
   # Char.append(Features)
    for ch in Char:
        database.append(pname)
    #database.append(Features)
             
    

print(database[3])
    
cnr =1
temp = []
for f in Features:
    temp.append(cnr)
    cnr  =cnr+1 
print(len(temp))
   

for x in Features:
    #for l in x:
    mean = statistics.mean(x)
    sdev = statistics.stdev(x)

for c in Features:
    for r in c:
       r = r - mean
       r = r / sdev
    

D = cdist(Features, Features)
io.imshow(D)
plt.title('Distance Matrix')
io.show()

dist = []
for c in D:
    for r in c:
        dist = D

for clm in dist:
    D_index = np.argsort(dist, axis = 1)
    
    
    
Ytrue = []
Ypred = []
print(len(database))
print(len(Features))
def column(matrix, i):
    return [row[i] for row in matrix]

Ypred = column(D_index, 1)
Ytrue = temp



confM = confusion_matrix(Ytrue, Ypred)
io.imshow(confM)
plt.title('Confusion Matrix')
io.show()

    
