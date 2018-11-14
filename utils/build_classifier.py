import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split

def hog_convert_split(X, Y):

    dat = []
    for data in X:
        dat.append(hog(data, orientations=8, pixels_per_cell=(6, 6), block_norm = 'L1',
                        cells_per_block=(3, 3), visualize=False, multichannel=False))
    dat = np.asarray(dat)

    X_train, X_test, y_train, y_test = train_test_split(dat, Y)
    return(X_train, X_test, y_train, y_test, dat)
