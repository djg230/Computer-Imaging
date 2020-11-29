import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central,inertia_tensor, moments_normalized, moments_hu, find_contours, profile_line
from skimage import io, exposure
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing
from matplotlib.patches import Rectangle
import pickle
import os
import statistics
import sys
from skimage.filters import threshold_mean, threshold_minimum, threshold_otsu

pkl_file = open('test_gt_py3.pkl', 'rb')
mydict = pickle.load(pkl_file)
pkl_file.close()
classes = mydict[b'classes']
locations = mydict[b'locations']

directory = '/Users/davidgiansanti/Desktop/images/'
file = sys.argv[1]
database = []
Features=[]
for photo in os.listdir(directory):
    if(photo == 'test.bmp'):
        continue
    if(photo == '.DS_Store'):
        continue
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
    
    th = threshold_otsu(img)
    img_binary = (img < th).astype(np.double)
    img_binary = binary_closing(img_binary)
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
        
    ax.set_title('Bounding Boxes')
    io.show()
   
    for ch in Char:
        database.append(pname)
  
for fe in Features:
    temp.append(cnr)
    cnr  =cnr+1 
for x in Features:
    #for l in x:
    mean = statistics.mean(x)
    sdev = statistics.stdev(x)
             
for c in Features:
    for r in c:
       r = r - mean
       r = r / sdev
photo = directory + file
count = 0 
pname = photo.replace('.bmp', "")
# photo = "/Users/davidgiansanti/Desktop/images/w.bmp"
img = io.imread(photo);
#print(img.shape)

io.imshow(img)
plt.title('Original Image')
io.show()

hist = exposure.histogram(img)
plt.bar(hist[1], hist[0])
plt.title('Histogram')
plt.show()
 
th = threshold_otsu(img)
img_binary = (img < th).astype(np.double)
img_binary = binary_closing(img_binary)
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
ccount = []
testF = []
Cha = []
for props in regions:
     minr,minc, maxr,maxc = props.bbox
     roi = img_binary[minr:maxr, minc:maxc]
     m = moments(roi)
     cc = m[0, 1] / m[0, 0]
     cr = m[1, 0] / m[0, 0]
     mu = moments_central(roi, center=(cr,cc))
     nu = moments_normalized(mu)
     hu = moments_hu(nu)
     testF.append(hu)
     ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr,
     fill = False, edgecolor = 'red', linewidth = 1))
 #Char.append(pname)
ax.set_title('Bounding Boxes')
io.show()
z =0
for q in testF:
    for f in q:
       f = f - mean
       f = f / sdev
    z=z+1
print(z)      
D = cdist(testF, Features)
io.imshow(D)
plt.title('Distance Matrix')
io.show()

dist = []
for c in D:
    for r in c:
        dist = D

for clm in dist:
    D_index = np.argsort(dist, axis = 1)
    
    
def column(matrix, i):
    return [row[i] for row in matrix]
nbr = 0
neighbor = []
neighbors = []
countr = 0
D_index = np.argsort(D, axis = 1)
for ind in D_index:
    for i in ind:
        if nbr < 10:
            neighbor.append(i)
        nbr = nbr + 1
    neighbor = []
    nbr = 0
    neighbors.append(neighbor)
    
similar = 0
oldsim = 0
voted = []
for n in neighbors:
    for nb in n:
        nb = database[nb]
        for x in n:
            x = database[x]
            if nb == x:
                similar = similar + 1
        if similar > oldsim:
            vote = nb
        oldsim = similar
        similar = 0
    voted.append(vote)
            

Ypred = voted
for v in Ypred:
    print(v)
cnr = 1


c= 0

location = column(locations,1)

l1 = column(locations, 0)
l2 = column(locations, 1)

co = 0
for l in classes:
    if Ypred[c] == l:
        co = co+1
    c = c+1


c  = 1
c3 = 0



print("Letters Recognized: " + str(co))
print("Recognition Rate: " + str((co/len(Ypred)) *100) + "%")


            