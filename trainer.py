import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central,
moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle


directory =
for photo in os.listdir(directory)
    count = 0
    pname = photo.replace('.bmg', "")
    img = io.imread(photo);

    print (img.shape)

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

    img_label = label(img-binary, background = 0)

    io.imshow(img_label)
    plt.title('Labeled image')
    io.show()

    print(np.amax(img_label))

    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()

    Features=[]
    Char =[][]
    Char[1][count] = pname
    for props in regions:
        minr,minc, maxr,maxc = props.bbox
        roi = img_binary[minr:maxr, minc:maxc]
        m= moments(roi)
        cc = m[0, 1] / m[0, 0]
        cr = m[1, 0] / m[0, 0]
        mu = moments_central(roi, center=(cr,cc))
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        features.append(hu)
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr,
        fill = False, edgecolor = 'red', linewidth = 1))
    ax.set_title('Bounding Boxes')
    io.show()


D = cdist(Features, Features)
io.imshow(D)
plt.title('Distance Matrix')
io.show()

D_index = np.argsort(dist, axis = 1)

cntr = 0
cls = []
def column(matrix, i):
    return [row[i] for row in matrix]

for clmn in D_index
    cls = class[cntr]
    Ypred = column(D_index, 1)
    Ytrue = cls

confusion_matrix(Ytrue, Ypred)

io.imshow(confM)
plt.title('Confusion Matrix')
io.show()
