import os

def log_file(directory, meta, **kwargs):
    back_sub_FOC = kwargs['back_sub_ch1']
    back_sub_Nucleus = kwargs['back_sub_ch2']
    small_object = kwargs['small_object']
    large_object = kwargs['large_object']
    threshold = kwargs['threshold']

    with open(directory+'/'+meta["Name"]+".txt",'w') as file:
        file.write('Size kernel ch1 : {}\n'\
                   'Size kernel ch2 : {}\n'\
                   'Size of the smallest Sigma : {}\n'\
                   'Size of the largest Sigma : {}\n'\
                   'Threshold: {}\n'\
                    .format(str(back_sub_FOC),str(back_sub_Nucleus),
                            str(small_object),
                            str(large_object), str(threshold)))
