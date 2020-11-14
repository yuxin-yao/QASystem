#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import argparse
from ask_pipeline.ask_main_pipeline import ask_main_pipeline

def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('ask')
    parser.add_argument('fpath', help='Path to your article.', type=str)
    parser.add_argument('nquestions', help='Number of questions to be generated.', type=int)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # ./ask article.txt nquestions
    args = parse_args()
    fpath = args.fpath
    nquestions = args.nquestions
    with open(fpath,"r") as f:
        doc=f.read()
    questions = ask_main_pipeline(doc,nquestions)
    # for i in questions:
    #     print(i[0])

    with open('listfile.txt', 'w') as f:
        for i in questions:
            print(i)
            f.write('%s\n' % i)
