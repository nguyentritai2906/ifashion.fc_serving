import grpc
from protos.tensorflow_serving.apis import predict_pb2  
from protos.tensorflow_serving.apis import prediction_service_pb2, prediction_service_pb2_grpc
import numpy as np
import cv2
from protos.tensorflow.core.framework import tensor_pb2  
from protos.tensorflow.core.framework import tensor_shape_pb2, tensor_shape_pb2_grpc 
from protos.tensorflow.core.framework import types_pb2, types_pb2_grpc



def grpc_infer(imgs):
    """recommendclothes - serving with gRPC"""
    # Connect to host
    channel = grpc.insecure_channel("localhost:8500")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Predict request
    request = predict_pb2.PredictRequest()
    # Model name
    request.model_spec.name = "fashion-compatibility"
    # Signature name, default is `serving_default`
    request.model_spec.signature_name = "serving_default"
    # Add the image input 
    tensor_shape1 = imgs.shape
    dims1 = [tensor_shape_pb2.TensorShapeProto.Dim(size=dim) for dim in tensor_shape1]  
    tensor_shape1 = tensor_shape_pb2.TensorShapeProto(dim=dims1)
    request.inputs["input.1"].CopyFrom(
        tensor_pb2.TensorProto(
            dtype=types_pb2.DT_FLOAT,
            tensor_shape=tensor_shape1,
            float_val=imgs.reshape(-1))
        )    
    # Predict result
    # try:
    result = stub.Predict(request, 10.0)
    outputs_tensor_proto = result.outputs["173"]
    outputs = []
    for out in outputs_tensor_proto.float_val:
        outputs.append(out)
    print(outputs)
    return outputs


def center_crop(img, dim):
    """Returns center cropped image
    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped
    """
    width, height = img.shape[1], img.shape[0]

    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def preprocess_image(image_path):
    i = cv2.imread(image_path)
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
    i = cv2.resize(i, (112, 112))
    i = center_crop(i, (112, 112))
    i = cv2.flip(i, 1)
    i = i/255
    i = np.transpose(i,(2,0,1))
    i0 = (i[0] - 0.485) / 0.229
    i1 = (i[1] - 0.456) / 0.224
    i2 = (i[2] - 0.406) / 0.225
    ithree = np.concatenate([[i0], [i1], [i2]], 0)
    output = np.expand_dims(ithree, 0)
    return output
