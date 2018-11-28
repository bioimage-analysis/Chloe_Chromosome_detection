import numpy as np
from skimage.feature import hog
from skimage.draw import ellipsoid

def _sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def sliding_window(image, clf, scaler, stepSize=8, Zstep =4, windowSize = (70,70)):

    result = []
    # I use a sliding window every 4 Z steps
    for i in range(0,len(image),Zstep):
        for (x, y, window) in _sliding_window(image[i], stepSize=stepSize, windowSize=windowSize):
            try:
                im = hog(window, orientations=8, pixels_per_cell=(6, 6), block_norm = 'L2-Hys',
                                cells_per_block=(3, 3), visualize=False, multichannel=False)
                X_img = scaler.transform(im.reshape(1, -1))

                #X_img = (im-np.mean(im))/np.std(im)
                res = clf.predict_proba(X_img)
                result.append((x,y,i, res[:,1]))
            except ValueError:
                pass
    return np.asarray(result)

def non_max_suppression(result, probaThresh=0.1, overlapThresh=0.3):

    high_proba = result[np.where(result[:,3] > probaThresh)]

    x1 = high_proba[:,0]
    y1 = high_proba[:,1]
    x2 = high_proba[:,0]+70
    y2 = high_proba[:,1]+70
    boxes = np.stack((x1,y1,x2,y2), axis=1)

    # Vector with steps (Z steps)
    step = high_proba[:,2]

    # Vector with probability
    probs = high_proba[:,3]

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
     # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs
    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return np.concatenate((boxes[pick].astype("int"),step[pick].reshape(-1,1)),axis=1)

def binary(box, image):
    ellip_base = ellipsoid(35, 35, 35, spacing=(1.05, 1.05, 2.1), levelset=False).astype("int")

    pts = np.transpose(np.nonzero(ellip_base))
    z,x,y = image.shape
    # Need to add some extra padding around the z axis
    binary = np.zeros((x, y, z+30))
    for i in range(len(box)):
        binary[pts[:,0]+box[i,1].astype(int),
               pts[:,1]+box[i,0].astype(int),
               pts[:,2]+box[i,4].astype(int)] = 1

    #Remove extra padding
    return(binary[:,:,15:z+15])

def binary_select_foci(box, image, masked):
    ellip_base = ellipsoid(35, 35, 35, spacing=(1.05, 1.05, 2.1), levelset=False).astype("int")

    pts = np.transpose(np.nonzero(ellip_base))
    z,x,y = image.shape
    # Need to add some extra padding around the z axis
    liste = []
    for i in range(len(box)):
        binary = np.zeros((x, y, z+30))
        mask = np.copy(masked)
        binary[pts[:,0]+box[i,1].astype(int),
               pts[:,1]+box[i,0].astype(int),
               pts[:,2]+box[i,4].astype(int)] = 1
        binary = binary[:,:,15:z+15].transpose(2,0,1)
        mask[~binary.astype(bool)] = 0
        liste.append(np.squeeze(np.dstack((np.where(mask>0)[0],
                                np.where(mask>0)[1],
                                np.where(mask>0)[2],
                                np.full(len(np.where(mask>0)[0]), i)))))

    #Remove extra padding
    return(np.vstack(liste))

def find_duplicate(selected_blobs):
    unq, unq_idx, unq_cnt = np.unique(selected_blobs[:,0:3],
                                      axis=0,
                                      return_inverse=True,
                                      return_counts=True)
    cnt_mask = unq_cnt > 1
    #duplicate
    dup_ids = unq[cnt_mask]
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_idx, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx = np.argsort(unq_idx[idx_mask])
    # Duplicate indexes
    dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
    # Find number of blobs per nucleus
    num, cts = np.unique(selected_blobs[:,3], return_counts=True)
    # mask of duplicate
    index = np.arange(0,max(selected_blobs[:,3])+1,1)
    mask = np.in1d(index, np.unique(selected_blobs[np.asarray(dup_idx)][:,:,3]))
    return(num, cts, dup_idx, mask)

def remove_duplicate(cts, dup_idx, selected_blobs):
    new_cts = np.copy(cts)
    for dupl in selected_blobs[np.asarray(dup_idx)][:,:,3]:
        if new_cts[dupl[0]] > new_cts[dupl[1]]:
            new_cts[dupl[0]] = new_cts[dupl[0]] - 1
        elif new_cts[dupl[0]] <= new_cts[dupl[1]]:
            new_cts[dupl[1]] = new_cts[dupl[1]] - 1
    return(new_cts)
