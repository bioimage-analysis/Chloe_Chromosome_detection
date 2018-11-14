import re
from skimage import io
from glob import glob
import numpy as np
import os

def dataset(path, path_n):
    positive = []
    titles_pos = []
    for files in glob(path + '*.tif'):
        titles_pos.append(re.findall('\d+.tif', files ))
        positive.append(io.imread(files))

    negative = []
    titles_neg = []
    for files in glob(path_n + '*.tif'):
        titles_neg.append(re.findall('\d+_n.tif', files ))
        negative.append(io.imread(files))

    X1 = np.squeeze(np.stack(positive))
    X2 = np.squeeze(np.stack(negative))
    X = np.vstack((X1,X2))
    return(X, X1, X2, titles_pos, titles_neg)

def save_file(path, filename, data):
    if os.path.isfile(path+filename+".npy"):
        expand = 0
        while True:
            expand += 1
            new_filename = filename + "_" + str(expand)
            if os.path.isfile(path+new_filename):
                continue
            else:
                filename = new_filename
            break
    np.save(path+filename, data)
