from skimage.filters.rank import minimum
from skimage import io
from scipy import ndimage
import numpy as np
from sklearn.cluster import KMeans
from skimage import morphology

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

def binarization(image):
    data = image.ravel().reshape(1,-1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=4).fit(data.T)
    binary = kmeans.labels_.reshape(image.shape)
    #cluster can be "reverse" background = 1 and foreground = 0
    if np.count_nonzero(binary) > np.count_nonzero(1 - binary):
        binary = 1-binary
    # Adding some erosion to be more conservative
    #binary = morphology.opening(binary, morphology.ball(2))
    return(binary)

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
