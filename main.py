import argparse
import numpy as np
from processing_serving.grpc_recommend_api import preprocess_image, grpc_infer


def make_parser():
    parser = argparse.ArgumentParser("Find Similar Images Serving")
    parser.add_argument('-image_path', type=str, help='a path of the input image')
    parser.add_argument('-n', type=int, help='top n similar images')
    return parser

if __name__=="__main__":
    args = make_parser().parse_args()   
    # Image input
    # input_image = preprocess_image(args.image_path)
    # Output embedding
    output_embedding = grpc_infer(np.zeros((256, 3, 112, 112)))
    print(output_embedding)
