import os

def log_file(directory, meta, **kwargs):
    back_sub_ch1 = kwargs['back_sub_ch1']
    back_sub_ch2 = kwargs['back_sub_ch2']
    back_sub_ch3 = kwargs['back_sub_ch3']
    small_object = kwargs['small_object']
    large_object = kwargs['large_object']
    threshold = kwargs['threshold']

    with open(directory+'/'+meta["Name"]+".txt",'w') as file:
        file.write('Size kernel ch1 : {}\n'\
                   'Size kernel ch2 : {}\n'\
                   'Size kernel ch3 : {}\n'\
                   'Size of the smallest Sigma : {}\n'\
                   'Size of the largest Sigma : {}\n'\
                   'Threshold: {}\n'\
                    .format(str(back_sub_ch1),str(back_sub_ch2),
                            str(back_sub_ch3), str(small_object),
                            str(large_object), str(threshold)))
