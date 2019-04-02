from skimage.filters.rank import minimum
from skimage import io
from scipy import ndimage
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter
from sklearn.neighbors import KDTree
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import blob_dog
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy.spatial import distance
import re
import pandas as pd

def roll_ball(img, ch=3, size =20):
    result = np.empty(img.shape)
    # Selection channel
    im = img[:,:,:,ch]
    # Mean normalization
    to_ana = (im - np.mean(im))/(im.max() - im.min())
    # Background substraction
    background = ndimage.minimum_filter(to_ana, size = 20)
    result = to_ana-background
    return(result)

def background_correct(image, ch=3, size =20):

    img = image[:,:,:,ch]

    gauss = gaussian(img, sigma=5)
    size = 5
    background = uniform_filter(gauss, size)
    img_cor = img - background

    return(img_cor, background)


def binarization(image):
    data = image.ravel().reshape(1,-1)
    kmeans = MiniBatchKMeans(init='k-means++',n_clusters=2, batch_size=10000, 
                             n_init=10, max_no_improvement=10, verbose=0).fit(data.T)
    #kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit(data.T)
    binary = kmeans.labels_.reshape(image.shape)
    #cluster can be "reverse" background = 1 and foreground = 0
    if np.count_nonzero(binary) > np.count_nonzero(1 - binary):
        binary = 1-binary
    # Adding some erosion to be more conservative
    #binary = morphology.opening(binary, morphology.ball(2))
    return(binary)

def find_blob(img, smaller = 1, largest = 5, thresh = 60, view=True):
    threshold = int((threshold_otsu(img)*thresh)/100)
    blobs = blob_dog(img,  min_sigma=smaller,
                     max_sigma=largest, threshold=threshold)
    if view:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(np.amax(img, axis=0), vmax=img.max()/2)
        ax.axis('off')
        ax.set_title("detected blobs seen on max int projection")
        for blob in blobs:
            z,x,y,s = blob
            ax.scatter(y, x, s=s*50, facecolors='none', edgecolors='r')
    return blobs

def find_foci(blobs, image, binary):
    blob_im = np.zeros(image.shape, dtype=np.int)
    blob_im[(blobs[:,0]).astype(np.int),
            (blobs[:,1]).astype(np.int),
            (blobs[:,2]).astype(np.int)] = np.arange(len(blobs[:,1])) + 1

    #before = morphology.dilation(blob_im, morphology.ball(3))
    masked = np.copy(blob_im)
    masked[~binary.astype(bool)] = 0
    #masked[~mask_nucleus.astype(bool)] = 0
    #after = morphology.dilation(masked, morphology.ball(3))
    return(masked)

def distance_to_tip(point, skeleton, meta):
    coords = np.copy(point[:,0:2])
    coords[:,0] = (coords[:,0]+35)*meta['PhysicalSizeX'] + meta['PositionX']-19
    coords[:,1] = (480 - (coords[:,1]+35))*meta['PhysicalSizeY'] + meta['PositionY']-19
    tree = KDTree(skeleton)
    #point = np.array([[10445, 9855]])
    distance_tip = np.empty(len(coords))
    for i, coord in enumerate(coords):
        closest_dist, closest_id = tree.query(coord[np.newaxis, :] , k=1)
        dist = distance.cdist(skeleton, skeleton, 'euclidean')[0][closest_id] + closest_dist
        distance_tip[i] = dist
    return distance_tip

def final_table(meta, bbox_ML,bb_mask, dist_tip, cts, num, directory, save = False):
    ID = re.findall(r"\d\d(?=_D3D)", meta["Name"])
    ID_array = np.repeat(ID, len(bbox_ML[bb_mask]))
    chro_pos = np.squeeze(np.dstack((bbox_ML[bb_mask][:,0]+35,
                          bbox_ML[bb_mask][:,1]+35, bbox_ML[bb_mask][:,4])))
    df = pd.DataFrame(ID_array, columns = ['Image ID'])
    df["Chromosome position y,x,z"] = list(map(tuple, chro_pos.astype("int")))
    df["distance from tip in um"] = dist_tip.astype("int")
    df["Numbers of FOCI"] = cts
    df["cell number on image"] = num

    if save == True:
        try:
            df.to_csv(directory+'/'+'{}.csv'.format(meta["Name"]))
        except FileNotFoundError:
            df.to_csv('{}.csv'.format(meta["Name"]))
    return df
