import os
import time
import javabridge
import bioformats
from skimage.filters import roberts
import numpy as np
import pandas as pd

def _remove_duplicated_nucleus(coords):
    index = []
    tt_i = [0]
    for i in range(len(coords)):
        # Find coordinate that are close (+-3) and hide (delete) the first (tt_i) after every loop then create a mask
        tt = np.all(np.isclose(coords[i],  np.delete(coords,tt_i, axis=0), atol = 2), axis=1)
        tt_i.append(i+1)
        index.append((tt.nonzero()[0]+(i+1)).tolist())
        # remove dupliacted result_index
        result_index = np.asarray(list(set([item for sublist in index for item in sublist])))
        # Create a boolean array with only ones
        mask = np.ones(len(coords), dtype=bool)
        # Create the mask
        try:
            mask[result_index] = False
        except IndexError:
            pass
    return mask

def batch_analysis(path, clf, scaler, folder_batch, skelete, parameters = {}):
    from chromosome_dsb import load_data, search, img_analysis, visualization

    """Go through evry image files in the directory (path).
    Parameters
    ----------
    path : str
    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - imageformat
    """

    imageformat= ('D3D_ALX.dv')
    back_sub_ch1= parameters.get('back_sub_ch1')
    back_sub_ch2= parameters.get('back_sub_ch2')
    back_sub_ch3= parameters.get('back_sub_ch3')
    smaller= parameters.get('smaller')
    largest= parameters.get('largest')
    thresh= parameters.get('thresh')

    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]

    #Create liste for accumulation of all analysis
    final_data = []

    for file in imfilelist:
        print('\n')
        print("###############################")
        print("working on {}".format(file))
        tp1 = time.time()
        javabridge.start_vm(class_path=bioformats.JARS)
        print("opening data")
        image, meta, directory = load_data.load_bioformats(file, folder_batch)
        img = image[:,:,:,3]
        # Check image quality
        #print("check image quality")
        #var = roberts(np.amax(img, axis=0)).var()
        #if var <= 7e+06:
        #    print("quality not good")
        #    continue
        # Find the chromosome
        print("searching nucleus")
        result = search.rolling_window(img, clf, scaler)
        bbox_ML = search.non_max_suppression(result, probaThresh=0.5, overlapThresh=0.3)
        if len(bbox_ML)>0:
            #Substract background
            print("substract background")
            ch1, _ = img_analysis.background_correct(image, ch=1, size=back_sub_ch1)
            ch2, _ = img_analysis.background_correct(image, ch=2, size=back_sub_ch2)
            ch3, _ = img_analysis.background_correct(image, ch=3, size=back_sub_ch3)
            # Find the FOCI
            print("finding FOCI")
            blobs = img_analysis.find_blob(ch1, meta, directory, smaller = smaller,
                                   largest = largest, thresh = thresh,
                                   plot=False, save=True)

            # Binarization of the chromosome
            print('image binarization')
            binary = img_analysis.binarization(ch3)
            # Mask FOCI that are not on the chromosome
            masked = search.find_foci(blobs, ch1, ch3, binary, bbox_ML)
            # Mask FOCI that are not on a chromosome found by the Machine Learning
            res, bb_mask = search.binary_select_foci(bbox_ML, ch3, masked)
            # Find and remove FOCI that were counted twice
            num, cts, dup_idx, mask = search.find_duplicate(res, bb_mask)
            visualization.plot_result(img, res, bbox_ML, \
                                  cts, num, meta, directory, save = True, plot = False)
            dist_tip = img_analysis.distance_to_tip(bbox_ML, skelete, meta)
            df = img_analysis.final_table(meta, bbox_ML,\
                                 dist_tip, cts, num, \
                                 directory, save = True)
        else:
            dist_tip = img_analysis.distance_to_tip_no_nucleus(skelete, meta)
            df = img_analysis.final_table_no_nucleus(meta, dist_tip, directory, save = True)

        final_data.append(df)

        tp2 = time.time()
        print("It took {}sec to analyse it".format((tp2-tp1)))

    batch_result = pd.concat(final_data)
    print("lens of data before removing duplicate = {}".format(len(batch_result )))

    coord_stage = batch_result[['Chromosome position z',
                                'Chromosome position x in stage coordinate',
                                'Chromosome position y in stage coordinate']].values

    mask = _remove_duplicated_nucleus(coord_stage)
    #batch_result.drop_duplicates(subset ="Chromosome position in stage coordinate")
    print("lens of data after removing duplicate = {}".format(len(batch_result[mask])))
    try:
        batch_result[mask].to_csv(folder_batch+'/'+'full.csv')
    except FileNotFoundError:
        batch_result[mask].to_csv('full.csv')
