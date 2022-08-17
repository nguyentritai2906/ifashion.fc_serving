import argparse
from PIL import Image 
from torchvision import transforms as T
from processing_serving.grpc_recommend_api import preprocess_image, grpc_infer
from preprocessing import main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', type=int, action='append')
    parser.add_argument('--types', type=str, action='append')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--datadir', type=str, default='./data/polyvore_outfits/')
    parser.add_argument('--outputdir', type=str, default='./output/')
    parser.add_argument('--is_save', type=bool, default=False)

    parser.add_argument('--new_product', type=int, default=None)
    parser.add_argument('--new_type', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()       
    # Output embedding
    if args.new_product:
        assert args.new_type is not None, 'Please add category for new product!'
        # Image input
        input_image = preprocess_image(args.image_path)        
        output_embedding = grpc_infer(input_image)
        print(output_embedding)
        # INSERT NEW PRODUCT TO DATABASE #
        # .... #
    # Get answers
    question_indexes, candidate_indexes = main(args.k, args.datadir, args.outputdir, args.questions, args.types, args.is_save)
    print(candidate_indexes)


    
