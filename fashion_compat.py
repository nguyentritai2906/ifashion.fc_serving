import argparse
from preprocessing import preprocessing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--questions', type=int, action='append')
    parser.add_argument('--ans_type', type=str, action='append')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()       
    # Get answers
    candidate_indexes = preprocessing(args.qid, args.ans_type, args.k)
    print(candidate_indexes)
