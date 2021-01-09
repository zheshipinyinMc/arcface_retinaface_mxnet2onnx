# Update2 models
[retinaface_mnet025_v1](http://insightface.ai/files/models/retinaface_mnet025_v1.zip)  
[retinaface_mnet025_v2](http://insightface.ai/files/models/retinaface_mnet025_v2.zip)  

# Update1 fix_gamma  
[retinaface_mnet025_v1](http://insightface.ai/files/models/retinaface_mnet025_v1.zip)  
[retinaface_mnet025_v2](http://insightface.ai/files/models/retinaface_mnet025_v2.zip)  
In mxnet symbol, BN has fix_gamma, if fix_gamma is true, then set gamma to 1 and its gradient to 0, you can find this in [mxnet API](https://mxnet.apache.org/versions/1.7.0/api/python/docs/api/symbol/symbol.html#mxnet.symbol.BatchNorm).  
In retinaface_mnet025_v1, fix_gamma in 'conv_3_dw_batchnorm' is true，but its value is 0.000007107922556315316(you can see weight by Netron).However, forward mxnet model, the gamma of 'conv_3_dw_batchnorm' is 1.This bug may cause mxnet output is different from onnx model.  
  
fix bn gamma model have upload(/Retinaface/retinaface_mnet025_v1, /Retinaface/retinaface_mnet025_v2).  

# Update  
Retinaface fixed softmax bug.  
Upsample is implemented using Resize.  
Upsample is implemented using ConvTranspose.  

# arcface_retinaface_mxnet2onnx
arcface and retinaface model convert mxnet to onnx  

# environment
Ubuntu 18.04  
MxNet 1.5.0  
onnx 1.7.0 (protobuf 3.0.0)  
onnxruntime 1.3.0  
Python 3.6.9  
cv2 3.3.1  

# tested models
arcface:from [ZQCNN](https://github.com/zuoqing1988/ZQCNN) [mobilefacenet-res2-6-10-2-dim512](https://pan.baidu.com/s/1_0O3kJ5dMmD-HdRwNR0Hpw#list/path=%2F)  
retinaface:[mnet.25](https://link.zhihu.com/?target=https%3A//github.com/deepinsight/insightface/issues/669)  
arcface/model-r34-amf-slim and retinaface-R50 also can convert successfully, but model file is too big to upload.  

# arcface  
[Insightface中ArcFace MxNet2ONNX踩坑](https://zhuanlan.zhihu.com/p/165294876)  

# retinaface  
[Insightface中Retinaface MxNet2ONNX踩坑](https://zhuanlan.zhihu.com/p/166267806)  

# Results
![arcface onnx_mxnet_output](https://github.com/zheshipinyinMc/arcface_retinaface_mxnet2onnx/tree/master/Arcface/onnx_mxnet_output.jpg)  
![retinaface onnx_mxnet_output](https://github.com/zheshipinyinMc/arcface_retinaface_mxnet2onnx/tree/master/Retinaface/mxnet_onnx_result.jpg)  

# reference
[insightface](https://github.com/deepinsight/insightface)
