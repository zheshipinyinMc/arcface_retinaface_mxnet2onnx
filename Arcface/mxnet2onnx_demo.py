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
    sym = './mobilefacenet-res2-6-10-2-dim512/model-symbol.json'
    params = './mobilefacenet-res2-6-10-2-dim512/model-0000.params'
    
    #NCHW
    input_shape = [(1,3,112,112)]
    
    onnx_file = './mobilefacenet-res2-6-10-2-dim512/onnx/modelnew_onnx.onnx'
    
    #返回转换后的onnx模型的路径
    converted_model_path = onnx_mxnet.export_model(sym, params, input_shape, np.float, onnx_file)
    
    #Check the model
    onnx.checker.check_model(onnx_file)
    print('The model is checked!')


#onnx model inferrence
def onnx_inferred_demo():
    onnx_file = './mobilefacenet-res2-6-10-2-dim512/onnx/modelnew2_onnx.onnx' 
    
    image_path='0.jpg'
    img = cv2.imread(image_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #img=(img-127.5)
    #img=img*0.0078125
    
    img = np.transpose( img, (2,0,1) ) #HWC->CHW
    
    ort_session = ort.InferenceSession(onnx_file)
    input_name = ort_session.get_inputs()[0].name #'data'
    outputs = ort_session.get_outputs()[0].name #'fc1'
    
    input_blob = np.expand_dims(img, axis=0).astype(np.float32) #NCHW
    
    out = ort_session.run([outputs], input_feed={input_name: input_blob})
    
    print(out[0])


def createGraphMemberMap(graph_member_list):
    member_map=dict();
    for n in graph_member_list:
        member_map[n.name]=n;
    return member_map


#修改onnx的结构
def onnx_modify_demo():
    
    onnx_file = './mobilefacenet-res2-6-10-2-dim512/onnx/modelnew_onnx.onnx'
    
    model = onnx.load(onnx_file)
    
    graph = model.graph
    initializer_map = createGraphMemberMap(graph.initializer)
    input_map = createGraphMemberMap(graph.input) #'data' 'scalar_op1' 'scalar_op2'
    node_map = createGraphMemberMap(graph.node) #'_minusscalar0' '_mulscalar0'
    output_map = createGraphMemberMap(graph.output) #'fc1'
    #print(input_map)
    #print(input_map.keys())
    
    #====data, double to float====
    name='data'
    #print(input_map[name])
    graph.input.remove(input_map[name])
    new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, [1,3,112,112])
    graph.input.extend([new_nv])
    #input_map = createGraphMemberMap(graph.input)
    #print(input_map[name])
    
    #====Sub, double to float====
    #修改input的dtype
    input_map = createGraphMemberMap(graph.input)
    name='scalar_op1'
    graph.input.remove(input_map[name])
    new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
    graph.input.extend([new_nv])
    #同时要修改initializer中初始值的type
    initializer_map = createGraphMemberMap(graph.initializer)
    graph.initializer.remove(initializer_map[name])
    new_nv = helper.make_tensor(name, TensorProto.FLOAT, [1],[127.5])
    graph.initializer.extend([new_nv])
    
    #====Mul, double to float====
    input_map = createGraphMemberMap(graph.input)
    name='scalar_op2'
    graph.input.remove(input_map[name])
    new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])
    graph.input.extend([new_nv])
    
    initializer_map = createGraphMemberMap(graph.initializer)
    graph.initializer.remove(initializer_map[name])
    new_nv = helper.make_tensor(name, TensorProto.FLOAT, [1],[0.0078125])
    graph.initializer.extend([new_nv])
    
    #====FC1, double to float====
    output_map = createGraphMemberMap(graph.output)
    name='fc1'
    graph.output.remove(output_map[name])
    new_nv = helper.make_tensor_value_info(name, TensorProto.FLOAT, [1,512])
    graph.output.extend([new_nv])
    
    
    #===PReLU slop===
    #C--->C*1*1
    for input_name in input_map.keys():
        if input_name.endswith('relu_gamma'):
            #print(input_name)
            input_shape = input_map[input_name].type.tensor_type.shape.dim
            input_dim_val=input_shape[0].dim_value
            #print(input_dim_val)
            
            graph.input.remove(input_map[input_name])
            new_nv = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [input_dim_val,1,1])
            graph.input.extend([new_nv])
            
            initializer_map = createGraphMemberMap(graph.initializer)
            graph.initializer.remove(initializer_map[input_name])
            weight_array = numpy_helper.to_array(initializer_map[input_name])
            #print(weight_array.shape)
            weight_array=weight_array.astype(np.float) #np.float32与initializer中不匹配
            #===可以查看权重输出！！！
            
            b=[]
            for w in weight_array:
                b.append(w)
            new_nv = helper.make_tensor(input_name, TensorProto.FLOAT, [input_dim_val,1,1], b)
            graph.initializer.extend([new_nv])
    
    onnx.save(model, './mobilefacenet-res2-6-10-2-dim512/onnx/modelnew2_onnx.onnx')


if __name__ == '__main__':
    
    
    #mxnet2onnx_test() #==mxnet2onnx
    
    #onnx_modify_demo() #===onnx修改===
    
    onnx_inferred_demo() #===onnx前向推导===
    
    
    
