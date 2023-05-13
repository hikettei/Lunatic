
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import argparse
#from transformers import AutoTokenizer, pipeline

from src import maddness
## Tasks
## MLP
## MHA
import numpy as np

def run_lunatic(args):
    generator = pipeline('text-generation', model=args.model)
    model = torch.compile(generator.unet)
    print(model)

def give_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
                        help="The model name.")
    
if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #give_parser_arguments(parser)
    #args = parser.parse_args()
    #run_lunatic(args)

    A = np.random.randn(128, 512)
    B = np.random.randn(128, 64)
    
    buckets, protos = maddness.train_encoder(A)
    print(protos)
    #lut, alpha, beta = maddness.construct_lut(B, protos, 16, 4)
    #dims, vals, scals, offsets = maddness.flatten_buckets(buckets, 4)

    #B_enc = maddness.maddness_encode(B, dims, vals, scals, offsets, 16)

    #result = maddness.lut_scan(B_enc, lut, 16, 4)
    #print(alpha)
    #print(beta)
    #print(result)
    #print(np.matmul(A, B.T))

#  a b c ...
# a. . .
# b. . .
# c. . .
# ...
# vocab_size**2 LUT + PE
