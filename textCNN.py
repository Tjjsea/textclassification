#!/usr/bin/env python
# -*- coding : UTF-8 -*-

import tensorflow as tf

class tcModel():
    def __init__(self,flags):
        self.input_x=tf.placeholder(tf.int32,[None,None],name="input_x")