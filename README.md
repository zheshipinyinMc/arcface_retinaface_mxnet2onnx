# arcface_retinaface_mxnet2onnx
arcface and retinaface model convert mxnet to onnx
# environment
MxNet 1.5.0  
onnx 1.7.0 (protobuf 3.0.0)  
onnxruntime 1.3.0  
Python 3.6.9  
cv2 3.3.1  
# tested models
arcface:from [ZQCNN](https://github.com/zuoqing1988/ZQCNN) [mobilefacenet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw#list/path=%2F)  
retinaface:[mnet.25](https://link.zhihu.com/?target=https%3A//github.com/deepinsight/insightface/issues/669)  

# arcface  
[Insightface中ArcFace MxNet2ONNX踩坑](https://zhuanlan.zhihu.com/p/165294876)  

# retinaface  
[Insightface中Retinaface MxNet2ONNX踩坑](https://zhuanlan.zhihu.com/p/166267806)  

# Results
![arcface onnx_mxnet_output](https://github.com/zheshipinyinMc/arcface_retinaface_mxnet2onnx/tree/master/Arcface/onnx_mxnet_output.jpg)  
![retinaface onnx_mxnet_output](https://github.com/zheshipinyinMc/arcface_retinaface_mxnet2onnx/tree/master/Retinaface/mxnet_onnx_result.jpg)  

# reference
[insightface](https://github.com/deepinsight/insightface)
