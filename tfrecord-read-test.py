#!/usr/bin/env python

import tensorflow as tf

record_iterator = tf.compat.v1.python_io.tf_record_iterator(path='00-512x512-798.tfrec')

for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    print(example)
    # Exit after 1 iteration as this is purely demonstrative.
    break
