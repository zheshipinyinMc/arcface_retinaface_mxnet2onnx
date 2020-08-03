from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import sqrt

import os
import os.path
import time
import json
import sys
import numpy as np
import argparse
import struct
import cv2
import mxnet as mx
from mxnet import ndarray as nd

def get_feature(image_path, model):
    img = cv2.imread(image_path)
    #img = img[...,::-1] #BGR->RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose( img, (2,0,1) ) #HWC->CHW
    
    #img=(img-127.5)
    #img=img*0.0078125
    
    embedding = None
    input_blob = np.expand_dims(img, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    
    _norm=np.linalg.norm(embedding)
    print(_norm)
    #embedding /= _norm  #归一化
    
    return embedding


def similarity(v1, v2):
    
    a=sqrt(np.dot(v1, v1))
    b=sqrt(np.dot(v2, v2))
    if a==0 or b==0:
        return -1
    cos_dis=np.dot(v1, v2) / (b * a)
    #print('cos:',cos_dis)
    
    return cos_dis


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=str, help='', default='3,112,112')
    parser.add_argument('--gpu', type=int, help='', default=-1)
    parser.add_argument('--model', type=str, help='', default='./mobilefacenet-res2-6-10-2-dim512/model,0')
    
    
    return parser.parse_args(argv)

def main(args):
    
    print(args)
    
    gpuid = args.gpu
    if gpuid>=0:
        ctx = mx.gpu(gpuid)
    else:
        ctx = mx.cpu()
    vec = args.model.split(',')
    assert len(vec)>1
    prefix = vec[0]
    epoch = int(vec[1])
    image_shape = [int(x) for x in args.image_size.split(',')]
    
    print('loading',prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    
    all_layers = sym.get_internals()
    
    sym = all_layers['fc1_output']
    
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(data_shapes=[('data', (1, 3, image_shape[1], image_shape[2]))])
    model.set_params(arg_params, aux_params)
    
    image_path2='0.jpg'
    fea2 = get_feature(image_path2, model)
    
    print(fea2)
    print(fea2.shape)
    
    #=====计算caffe模型和mxnet模型输出特征的余弦相似性=====
    #fea_caffe=caffe_demo()
    #score=similarity(fea_caffe, fea2[0])
    #print(score)
    


if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))

