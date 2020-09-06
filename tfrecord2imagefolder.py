#!/usr/bin/env python

from PIL import Image
import io, os
import numpy as np
import tensorflow as tf

infn = '00-512x512-798.tfrec'
record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=infn)

# create output folder
outpn = 'train'
if not os.path.exists(outpn):
    os.makedirs(outpn)

# loop through all binary data
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # peek at the dictionary keys
    # print(dict(example.features.feature).keys())
    # print(dict(example.features.feature)['id'], dict(example.features.feature)['class'])
    
    class_name = str(dict(example.features.feature)['class'].int64_list.value[0])
    # PyTorch's ImageFolder takes the class as the folder name, so create it.
    outpn2 = os.path.join(outpn, class_name)
    if not os.path.exists(outpn2):
        os.makedirs(outpn2)
    
    file_name = str(dict(example.features.feature)['id'].bytes_list.value[0].decode()) + '.jpg'
    outfpn = os.path.join(outpn2, file_name)
    
    # retrieve image
    example_bytes = dict(example.features.feature)['image'].bytes_list.value[0]
    PIL_array = np.array(Image.open(io.BytesIO(example_bytes)))
    PIL_img = Image.fromarray(PIL_array, 'RGB')

    # save to file
    PIL_img.save(outfpn)
    