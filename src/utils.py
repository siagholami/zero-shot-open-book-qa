# coding=utf-8
# Copyright 2021 - Sia Gholami

# py
import os, sys
import shutil
import pickle
import json, gzip, itertools

# pip
import boto3
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from sklearn.metrics import classification_report
import tqdm, h5py

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

# me
import gconfig

tf.get_logger().setLevel('ERROR')

def load_dict_from_h5(src_file, limit=None):
    data_dict = {}
    with h5py.File(src_file, 'r') as h5Obj:
        datasetnames = h5Obj.keys()
        for datasetname in datasetnames:
            data_dict[datasetname] = np.array(h5Obj[datasetname][:limit], dtype=h5Obj[datasetname].dtype)

    return data_dict