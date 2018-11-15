import re
from skimage import io
from glob import glob
import numpy as np
import os
import bioformats
from sklearn.externals import joblib

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

def load_bioformats(path):
    meta = metadata(path)
    image = np.empty((meta['SizeT'], meta['SizeZ'], meta['SizeX'], meta['SizeY'], meta['SizeC']))

    with bioformats.ImageReader(path) as rdr:
        for t in range(0, meta['SizeT']):
            for z in range(0, meta['SizeZ']):
                for c in  range(0, meta['SizeC']):
                    image[t,z,:,:,c]=rdr.read(c=c, z=z, t=t, series=None,
                                                 index=None, rescale=False, wants_max_intensity=False,
                                                 channel_names=None)
    return(np.squeeze(image))
