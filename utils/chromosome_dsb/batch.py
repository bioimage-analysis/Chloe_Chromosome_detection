import os
import time
import javabridge
import bioformats
from skimage.filters import roberts
import numpy as np

def batch_analysis(path, clf, scaler, skelete, parameters = {}):
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
    for file in imfilelist:
        print('\n')
        print("###############################")
        print("working on {}".format(file))
        tp1 = time.time()
        javabridge.start_vm(class_path=bioformats.JARS)
        print("opening data")
        image, meta, directory = load_data.load_bioformats(file)
        img = image[:,:,:,3]
        # Check image quality
        print("check image quality")
        var = roberts(np.amax(img, axis=0)).var()
        if var <= 2e+07:
            print("quality not good")
            continue
        # Find the chromosome
        print("searching nucleus")
        result = search.rolling_window(img, clf, scaler)
        bbox_ML = search.non_max_suppression(result, probaThresh=0.01, overlapThresh=0.3)
        #Substract background
        print("substract background")
        ch1, _ = img_analysis.background_correct(image, ch=1, size=back_sub_ch1)
        ch2, _ = img_analysis.background_correct(image, ch=2, size=back_sub_ch2)
        ch3, _ = img_analysis.background_correct(image, ch=3, size=back_sub_ch3)
        # Find the FOCI
        print("finding FOCI")
        blobs = img_analysis.find_blob(ch1, smaller = smaller,
                                   largest = largest, thresh = thresh,
                                   view=True)

        # Binarization of the chromosome
        print('image binarization')
        binary = img_analysis.binarization(ch3)
        # Mask FOCI that are not on the chromosome
        masked = img_analysis.find_foci(blobs, ch3, binary)
        # Mask FOCI that are not on a chromosome found by the Machine Learning
        res, bb_mask = search.binary_select_foci(bbox_ML, ch3, masked)
        # Find and remove FOCI that were counted twice
        num, cts, dup_idx, mask = search.find_duplicate(res)
        visualization.plot_result(img, res, bbox_ML, bb_mask,\
                              cts, num, meta, directory, save = True, plot = False)
        dist_tip = img_analysis.distance_to_tip(bbox_ML[bb_mask], skelete, meta)
        img_analysis.final_table(meta, bbox_ML, bb_mask, \
                             dist_tip, cts, num, \
                             directory, save = True)
        tp2 = time.time()
        print("It took {}sec to analyse it".format((tp2-tp1)))
