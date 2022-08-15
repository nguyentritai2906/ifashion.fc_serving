import grpc
import base64
import cv2
from protos.tensorflow_serving.apis import predict_pb2  
from protos.tensorflow_serving.apis import prediction_service_pb2, prediction_service_pb2_grpc
import numpy as np
from PIL import Image 
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
    request.model_spec.name = "mnist-serving"
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
    outputs_tensor_proto = result.outputs["OutputLayer"]
    outputs = []
    for out in outputs_tensor_proto.float_val:
        outputs.append(out)
    return outputs
    # except Exception as e:
    #     print(e)
    #     return None


def preprocess_image(image_path):
    pil_image = Image.open(image_path)
    old_size = pil_image.size
    desired_size = 256 
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    pil_image = pil_image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(pil_image, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    new_im = np.array(new_im)
    new_im = np.expand_dims(new_im, axis=0)
    new_im = new_im.astype(np.float32)
    new_im /= 255.0    
    return new_im
