import os
import time

def batch_analysis(path, **kwargs):

    """Go through evry image files in the directory (path).
    Parameters
    ----------
    path : str
    kwargs : dict
        Additional keyword-argument to be pass to the function:
         - imageformat
    """


    imageformat= kwargs.get('imageformat', 'D3D_ALX.dv')
    imfilelist=[os.path.join(path,f) for f in os.listdir(path) if f.endswith(imageformat)]
    i = 0
    for file in imfilelist:
        tp1 = time.time()
        i+=1
        javabridge.start_vm(class_path=bioformats.JARS)
        image, meta, directory = load_data.load_bioformats(file)
        img = image[:,:,:,3]
        # Check image quality
        var = roberts(np.amax(img, axis=0)).var()
        if var <= 2e+07:
            return
        # Find the chromosome
        result = search.rolling_window(img, clf, scaler)
        bbox_ML = search.non_max_suppression(result, probaThresh=0.01, overlapThresh=0.3)
        #Substract background
        ch1, _ = img_analysis.background_correct(image, ch=1, size=back_sub_ch1)
        ch2, _ = img_analysis.background_correct(image, ch=2, size=back_sub_ch2)
        ch3, _ = img_analysis.background_correct(image, ch=3, size=back_sub_ch3)
        # Find the FOCI
        blobs = img_analysis.find_blob(ch1, smaller = smaller,
                                   largest = largest, thresh = thresh,
                                   view=True)

        # Binarization of the chromosome
        binary = img_analysis.binarization(ch3)
        # Mask LOCI that are not on the chromosome
        masked = img_analysis.find_foci(blobs, ch3, binary)
        # Mask LOCI that are not on a chromosome found by the Machine Learning
        res, bb_mask = search.binary_select_foci(bbox_ML, ch3, masked)
        # Find and remove LOCI that were counted twice
        num, cts, dup_idx, mask = search.find_duplicate(res)
        visualization.plot_result(img, res, bbox_ML, bb_mask,\
                              cts, num, meta, directory, save = True, plot = False)
        dist_tip = img_analysis.distance_to_tip(bbox_ML[bb_mask], skelete, meta)
        img_analysis.final_table(meta, bbox_ML, bb_mask, \
                             dist_tip, cts, num, \
                             directory, save = True)
        tp2 = time.time()
        print("image_{} finished after {}sec".format(i, (tp2-tp1)))
