#!/usr/bin/env python
# Usage: if one stores multiple TFRecord files in ./tfrecords-jpeg-224x224/train,
# execute the following command to convert files all at once.
# $ time find tfrecords-jpeg-224x224/train -type f -name '*.tfrec' -exec ./tfrecord2imagefolder.py -i {} +

from PIL import Image
import argparse
import io, os
import numpy as np
import tensorflow as tf

def process_one_file(infn, outpn):
    record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=infn)

    # loop through all binary data
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        # peek at the dictionary keys
        # print(dict(example.features.feature).keys())
        # print(dict(example.features.feature)['id'], dict(example.features.feature)['class'])
        
        # For validation data, class field does not exists.
        if 'class' in dict(example.features.feature).keys():
            class_name = str(dict(example.features.feature)['class'].int64_list.value[0])
            # PyTorch's ImageFolder takes the class as the folder name, so create it.
            outpn2 = os.path.join(outpn, class_name)
        else:
            outpn2 = outpn
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', nargs='+')
    infns = parser.parse_args()._get_kwargs()[0][1]

    # create output folder
    outpn = 'train'
    outpn_from_user = input('Where do you want to output files? [./train] ')
    if not outpn_from_user == '':
        outpn = outpn_from_user
    if not os.path.exists(outpn):
        os.makedirs(outpn)

    for infn in infns:
        print('processing', infn)
        process_one_file(infn, outpn)
