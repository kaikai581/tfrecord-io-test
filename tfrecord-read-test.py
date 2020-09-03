#!/usr/bin/env python
# This script works.
# ref: https://stackoverflow.com/questions/54716696/tensorflow-load-unknown-tfrecord-dataset

from PIL import Image
import io, os
import numpy as np
import tensorflow as tf

infn = '00-512x512-798.tfrec'
record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=infn)

# make output directory
dir_name = os.path.splitext(infn)[0]
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# loop through all binary data
i = 0
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    # print(example)

    # print(dict(example.features.feature).keys())
    # OPTION 1: convert bytes to arrays using PIL and IO
    example_bytes = dict(example.features.feature)['image'].bytes_list.value[0]
    PIL_array = np.array(Image.open(io.BytesIO(example_bytes)))

    PIL_img = Image.fromarray(PIL_array, 'RGB')
    PIL_img.save(os.path.join(dir_name, 'img{}.jpg'.format(i)))
    i += 1

    # Exit after 1 iteration as this is purely demonstrative.
    # break
