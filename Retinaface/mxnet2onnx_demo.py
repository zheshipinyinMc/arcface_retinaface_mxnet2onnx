import cv2
import numpy as np
import os
import time

import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnx import helper
from onnx import TensorProto

#mxnet2onnx
def mxnet2onnx_test():
    sym = './mnet.25/mnet.25-symbol.json'
    params = './mnet.25/mnet.25-0000.params'
    
    #NCHW
    input_shape = [(1,3,640,640)]
    
    onnx_file = './mnet.25/onnx/mnet.25_onnx.onnx'
    
    #返回转换后的onnx模型的路径
    converted_model_path = onnx_mxnet.export_model(sym, params, input_shape, np.float32, onnx_file) #np.float32导致Sub、Mul报错！！！
    
    #Check the model
    onnx.checker.check_model(onnx_file)
    print('The model is checked!')


#onnx model inferrence
def onnx_inferred_demo():
    onnx_file = './mnet.25/onnx/mnet.25_onnx.onnx'
    
    image_path='./mnet.25/01.jpg'
    img = cv2.imread(image_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = np.transpose( img, (2,0,1) ) #HWC->CHW
    
    ort_session = ort.InferenceSession(onnx_file)
    input_name = ort_session.get_inputs()[0].name #'data'
    outputs = ort_session.get_outputs()[0].name 
    
    #=32
    #0:face_rpn_cls_prob_reshape_stride32
    #1:face_rpn_bbox_pred_stride32
    #2:face_rpn_landmark_pred_stride32
    
    #=16
    #3:face_rpn_cls_prob_reshape_stride16
    #4:face_rpn_bbox_pred_stride16
    #5:face_rpn_landmark_pred_stride16
    
    #=8
    #6:face_rpn_cls_prob_reshape_stride8
    #7:face_rpn_bbox_pred_stride8
    #8:face_rpn_landmark_pred_stride8
    
    input_blob = np.expand_dims(img, axis=0).astype(np.float32) #NCHW
    
    out = ort_session.run([outputs], input_feed={input_name: input_blob})
    
    print(out[0])
    print(out[0].shape)


if __name__ == '__main__':
    
    
    #mxnet2onnx_test() #==mxnet2onnx
    
    onnx_inferred_demo() #===onnx前向推导===
    
    
    
