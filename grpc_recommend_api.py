import grpc
from protos.tensorflow_serving.apis import predict_pb2  
from protos.tensorflow_serving.apis import prediction_service_pb2, prediction_service_pb2_grpc
import numpy as np
from PIL import Image 
from torchvision import transforms as T
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
    return outputs


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
                            T.Resize(112),
                            T.CenterCrop(112),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
    img = transform(img)
    img = np.expand_dims(img, 0)
    return img
