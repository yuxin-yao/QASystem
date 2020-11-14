#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import argparse
from answer_pipeline.answer_main_pipeline import answer_main_pipeline

def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('answer')
    parser.add_argument('article_fpath', help='Path to your article.', type=str)
    parser.add_argument('questions_fpath', help='Path to your questions..', type=str)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    # ./answer article.txt questions.txt
    args = parse_args()
    article_fpath = args.article_fpath
    questions_fpath = args.questions_fpath
    with open(article_fpath,"r") as f:
        doc=f.read()
    with open(questions_fpath,"r") as f:
        questions=f.readlines()
    answers = answer_main_pipeline(doc,questions)
    for i in answers:
        print(i[0][0])
    pass