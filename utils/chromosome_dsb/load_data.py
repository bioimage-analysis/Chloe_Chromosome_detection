import re
from skimage import io
from glob import glob
import numpy as np
import os
import bioformats
from sklearn.externals import joblib
from skimage.filters import roberts
import os
import time
#os.chdir(os.getcwd() + '\utils')
#print(os.getcwd())
#from utils import *

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

def save_file(path, filename, data, model=True):
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
    if model:
        joblib.dump(data, path+filename)
    else:
        np.save(path+filename, data)

def metadata(path):
    xml = bioformats.get_omexml_metadata(path)
    md = bioformats.omexml.OMEXML(xml)

    meta={'AcquisitionDate': md.image().AcquisitionDate}
    meta['Name']=md.image().Name
    meta['SizeC']=md.image().Pixels.SizeC
    meta['SizeT']=md.image().Pixels.SizeT
    meta['SizeX']=md.image().Pixels.SizeX
    meta['SizeY']=md.image().Pixels.SizeY
    meta['SizeZ']=md.image().Pixels.SizeZ
    meta['PhysicalSizeX'] = md.image().Pixels.PhysicalSizeX
    meta['PhysicalSizeY'] = md.image().Pixels.PhysicalSizeY
    meta['PhysicalSizeZ'] = md.image().Pixels.PhysicalSizeZ
    meta['PositionX'] = md.image().Pixels.Plane().PositionX
    meta['PositionY'] = md.image().Pixels.Plane().PositionY
    meta['Timepoint'] = md.image().Pixels.Plane().DeltaT
    return(meta)

def stage_position(path):
    position = []
    time_point = []
    variance = []
    boxes = []
    for files in glob(path + '*ALX.dv'):
        meta = metadata(files)
        position.append((meta['PositionX'], meta['PositionY']))
        time_point.append(int(meta['Timepoint']))
        ####WILL NEED THAT SOMEWHERE ELSE!!!
        #img = load_bioformats(files)
        #var = roberts(np.amax(img[:,:,:,3], axis=0)).var()
        #variance.append(var)
        ####
        #if var > 2e+07:
        #    result = search.sliding_window(img[:,:,:,3], clf, scaler, stepSize=16, Zstep =8)
        #    boxes.append(search.non_max_suppression(result, probaThresh=0.01,
        #                                                overlapThresh=0.3))
        #elif var <= 2e+07:
        #    boxes.append([])
    position = np.asarray(position)
    time_point = np.asarray(time_point)
    #variance = np.asarray(variance)
    #boxes = np.asarray(boxes)

    #return position, time_point, variance
    return position, time_point

def load_bioformats(path, channel = None, no_meta_direct = False):
    meta = metadata(path)

    if channel:
        image = np.empty((meta['SizeT'], meta['SizeZ'], meta['SizeX'], meta['SizeY'], 1))
        with bioformats.ImageReader(path) as rdr:
            for t in range(0, meta['SizeT']):
                for z in range(0, meta['SizeZ']):
                    image[t,z,:,:,0]=rdr.read(c=channel, z=z, t=t, series=None,
                                                     index=None, rescale=False, wants_max_intensity=False,
                                                     channel_names=None)
    else:
        image = np.empty((meta['SizeT'], meta['SizeZ'], meta['SizeX'], meta['SizeY'], meta['SizeC']))
        with bioformats.ImageReader(path) as rdr:
            for t in range(0, meta['SizeT']):
                for z in range(0, meta['SizeZ']):
                    for c in  range(0, meta['SizeC']):
                        image[t,z,:,:,c]=rdr.read(c=c, z=z, t=t, series=None,
                                                     index=None, rescale=False, wants_max_intensity=False,
                                                     channel_names=None)

    if no_meta_direct == True:
        return(np.squeeze(image))
    else:
        return(np.squeeze(image), meta, _new_directory(path, meta))

def _new_directory(path, meta):

    directory = os.path.dirname(path)+"/"+"result"+'_'+meta["Name"]+'_'+ time.strftime('%m'+'_'+'%d'+'_'+'%Y')
    if os.path.exists(directory):
        expand = 0
        while True:
            expand += 1
            new_directory = directory+"_"+str(expand)
            if os.path.exists(new_directory):
                continue
            else:
                directory = new_directory
                os.makedirs(directory)
                break
    else:
        os.makedirs(directory)
    return(directory)

def _line_coord(position1, position2):
    i = [position1[0], position2[0]]
    j = [position1[1], position2[1]]
    a, b = np.polyfit(i, j, 1)
    #y = ax + b
    if position1[0] == position2[0]:
        # No change in x, slope is infinite, adjust with change in y
        step = abs(position1[1] - position2[1])/20
        x = np.repeat(position1[0], 20)
        if position1[1] < position2[1]:
            #if it goes up..
            y = np.arange(position1[1], position2[1], step)
        if position1[1] > position2[1]:
            #if it goes down...
            y = np.arange(position2[1], position1[1], step)
    if position1[0] > position2[0]:
        # Goes negative direction
        step = abs(position2[0] - position1[0])/20
        x = np.arange(position2[0], position1[0], step)
        y = a*x+b
        x = x[::-1]
        y = y[::-1]
    if position1[0] < position2[0]:
        step = abs(position1[0] - position2[0])/20
        x = np.arange(position1[0], position2[0], step)
        y = a*x+b

    return x, y

def skeleton_coord(position,time_point):
    data = np.concatenate((position,time_point[:, np.newaxis]), axis=1)
    sort_data = data[np.argsort(data[:,2])]
    list_line = []
    for position1, position2 in zip(sort_data, sort_data[1:]):
        x,y = _line_coord(position1, position2)
        list_line.append((x,y))
    return np.transpose(np.hstack(list_line), (1,0))
