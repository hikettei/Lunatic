
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import argparse
from transformers import AutoTokenizer, pipeline


def run_lunatic(args):
    generator = pipeline('text-generation', model=args.model)
    model = torch.compile(generator.unet)
    print(model)

def give_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
                        help="The model name.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    give_parser_arguments(parser)
    args = parser.parse_args()
    run_lunatic(args)
