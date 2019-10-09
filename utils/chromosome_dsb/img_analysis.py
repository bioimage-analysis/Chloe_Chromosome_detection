from skimage.filters.rank import minimum
from skimage import io
from scipy import ndimage
import numpy as np
from sklearn.cluster import KMeans

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

def find_blob(img, meta, directory, smaller = 1, largest = 5, thresh = 60, plot=True, save=False):
    #threshold = int((threshold_otsu(img)*thresh)/100)

    blobs = blob_dog(img,  min_sigma=smaller,
                     max_sigma=largest, threshold=thresh)
    if plot == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        ax1.imshow(np.amax(img,axis=0), vmax=img.max()/1.8, cmap="viridis")
        ax1.axis('off')
        ax2.imshow(np.amax(img,axis=0), vmax=img.max()/1.8, cmap="viridis")
        ax2.axis('off')
        for blob in blobs:
            z,x,y,s = blob
            loci = ax2.scatter(y, x, s=40, facecolors='none', edgecolors='y')
        if save:
            try:
                filename = meta['Name']+"FOCI"+'.pdf'
                plt.savefig(directory+'/'+filename, transparent=True)
            except FileNotFoundError:
                plt.savefig(filename, transparent=True)
    elif plot ==False:
        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        ax1.imshow(np.amax(img,axis=0), vmax=img.max()/1.8, cmap="viridis")
        ax1.axis('off')
        ax2.imshow(np.amax(img,axis=0), vmax=img.max()/1.8, cmap="viridis")
        ax2.axis('off')
        for blob in blobs:
            z,x,y,s = blob
            loci = ax2.scatter(y, x, s=40, facecolors='none', edgecolors='y')
        if save:
            try:
                filename = meta['Name']+"_FOCI"+'.pdf'
                plt.savefig(directory+'/'+filename, transparent=True)
            except FileNotFoundError:
                plt.savefig(filename, transparent=True)
        plt.close(fig)

    return blobs

def distance_to_tip(point, skeleton, meta):
    coords = np.copy(point[:,0:2])
    coords[:,0] = (coords[:,0]+35)*meta['PhysicalSizeX'] + meta['PositionX']-19
    coords[:,1] = (480 - (coords[:,1]+35))*meta['PhysicalSizeY'] + meta['PositionY']-19
    tree = KDTree(skeleton)
    #accumulated distance from tip
    dist = []
    for i in range(len(skeleton)-1):
        dist.append(distance.cdist(skeleton[i].reshape(1,2), skeleton[i+1].reshape(1,2), 'euclidean'))
    accumulated_dist = np.copy(dist)
    for i in range(len(accumulated_dist)-1):
        accumulated_dist[i+1] = accumulated_dist[i] + accumulated_dist[i+1]

    accumulated_dist = np.pad(accumulated_dist, ((0,1), (0,0), (0,0)), 'edge')
    distance_tip = np.empty(len(coords))
    for i, coord in enumerate(coords):
        closest_dist, closest_id = tree.query(coord[np.newaxis, :] , k=1)
        #dist = distance.cdist(skeleton, skeleton, 'euclidean')[0][closest_id] + closest_dist
        distance_tip[i] = accumulated_dist[closest_id] + closest_dist
    return distance_tip

def final_table(meta, bbox_ML, dist_tip, cts, num, directory, save = False):
    ID = re.findall(r"\d+(?=_D3D)", meta["Name"])
    ID_array = np.repeat(ID, len(bbox_ML))

    chro_pos_y = bbox_ML[:,0]+35
    chro_pos_x = bbox_ML[:,1]+35
    chro_pos_z = bbox_ML[:,4]

    coords = np.copy(bbox_ML[:,0:2])
    coords[:,0] = (bbox_ML[:,0]+35)*meta['PhysicalSizeX'] + meta['PositionX']-19
    coords[:,1] = (480 - (bbox_ML[:,1]+35))*meta['PhysicalSizeY'] + meta['PositionY']-19

    df = pd.DataFrame(ID_array, columns = ['Image ID'])
    df["Chromosome position x"] = chro_pos_x
    df["Chromosome position y"] = chro_pos_y
    df["Chromosome position z"] = chro_pos_z

    df["Chromosome position x in stage coordinate"] = coords[:,0]
    df["Chromosome position y in stage coordinate"] = coords[:,1]

    df["distance from tip in um"] = dist_tip.astype("int")
    df["Numbers of FOCI"] = cts
    df["cell number on image"] = num

    if save == True:
        try:
            df.to_csv(directory+'/'+'{}.csv'.format(meta["Name"]))
        except FileNotFoundError:
            df.to_csv('{}.csv'.format(meta["Name"]))
    return df

'''
def final_table(meta, bbox_ML, dist_tip, cts, num, directory, save = False):
    ID = re.findall(r"\d+(?=_D3D)", meta["Name"])
    ID_array = np.repeat(ID, len(bbox_ML))
    chro_pos = np.squeeze(np.dstack((bbox_ML[:,0]+35,
                          bbox_ML[:,1]+35, bbox_ML[:,4])))

    coords = np.copy(bbox_ML[:,0:2])
    coords[:,0] = (bbox_ML[:,0]+35)*meta['PhysicalSizeX'] + meta['PositionX']-19
    coords[:,1] = (480 - (bbox_ML[:,1]+35))*meta['PhysicalSizeY'] + meta['PositionY']-19

    df = pd.DataFrame(ID_array, columns = ['Image ID'])
    if chro_pos.ndim ==1:
        df["Chromosome position y,x,z"] = list(map(tuple, chro_pos.astype("int").reshape(1,3)))
    else:
        df["Chromosome position y,x,z"] = list(map(tuple, chro_pos.astype("int")))
    df["Chromosome position in stage coordinate"] = list(map(tuple, coords.astype("int")))
    df["distance from tip in um"] = dist_tip.astype("int")
    df["Numbers of FOCI"] = cts
    df["cell number on image"] = num

    if save == True:
        try:
            df.to_csv(directory+'/'+'{}.csv'.format(meta["Name"]))
        except FileNotFoundError:
            df.to_csv('{}.csv'.format(meta["Name"]))
    return df
'''

def distance_to_tip_no_nucleus(skeleton, meta):
    coords = np.asarray((meta['PositionX'], meta['PositionY'])).reshape(1,2)
    tree = KDTree(skeleton)

    dist = []
    for i in range(len(skeleton)-1):
        dist.append(distance.cdist(skeleton[i].reshape(1,2), skeleton[i+1].reshape(1,2), 'euclidean'))
    accumulated_dist = np.copy(dist)
    for i in range(len(accumulated_dist)-1):
        accumulated_dist[i+1] = accumulated_dist[i] + accumulated_dist[i+1]

    accumulated_dist = np.pad(accumulated_dist, ((0,1), (0,0), (0,0)), 'edge')
    distance_tip = np.empty(len(coords))
    for i, coord in enumerate(coords):
        closest_dist, closest_id = tree.query(coord[np.newaxis, :] , k=1)
        #dist = distance.cdist(skeleton, skeleton, 'euclidean')[0][closest_id] + closest_dist
        distance_tip[i] = accumulated_dist[closest_id] + closest_dist
    return distance_tip

def final_table_no_nucleus(meta, dist_tip, directory, save = False):

    ID = re.findall(r"\d+(?=_D3D)", meta["Name"])

    chro_pos_y = np.nan
    chro_pos_x = np.nan
    chro_pos_z = np.nan

    df = pd.DataFrame(ID, columns = ['Image ID'])
    df["Chromosome position x"] = chro_pos_x
    df["Chromosome position y"] = chro_pos_y
    df["Chromosome position z"] = chro_pos_z

    df["Chromosome position x in stage coordinate"] = int(meta['PositionX'])
    df["Chromosome position y in stage coordinate"] = int(meta['PositionY'])

    df["distance from tip in um"] = dist_tip.astype("int")
    df["Numbers of FOCI"] = 0
    df["cell number on image"] = 0

    if save == True:
        try:
            df.to_csv(directory+'/'+'{}.csv'.format(meta["Name"]))
        except FileNotFoundError:
            df.to_csv('{}.csv'.format(meta["Name"]))
    return df

'''
def final_table_no_nucleus(meta, dist_tip, directory, save = False):
    ID = re.findall(r"\d+(?=_D3D)", meta["Name"])
    coords = np.asarray((meta['PositionX'], meta['PositionY'])).reshape(1,2)
    df = pd.DataFrame(ID, columns = ['Image ID'])
    df["Chromosome position y,x,z"] = np.nan
    df["Chromosome position in stage coordinate"] = list(map(tuple, coords.astype("int")))
    df["distance from tip in um"] = dist_tip.astype("int")
    df["Numbers of FOCI"] = 0
    df["cell number on image"] = 0

    if save == True:
        try:
            df.to_csv(directory+'/'+'{}.csv'.format(meta["Name"]))
        except FileNotFoundError:
            df.to_csv('{}.csv'.format(meta["Name"]))
    return df
'''
