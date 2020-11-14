#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import json
import prettytable as pt
import logging
from answer_pipeline.answer_main_pipeline import answer_main_pipeline
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

squad = "./data/squad/train-v2.0.json"

with open(squad,"r") as f:
    squad=f.read()
j = json.loads(squad)["data"]

# 更改这里的下标就可以跳到任意编号的文章再开始测试
j=j[100:]

for article in j:
    logging.info("\nTesting article:%s"% article['title'])
    article_str = ''
    questions = []
    correct_answers = []
    for paragraph in article['paragraphs']:
        article_str += paragraph["context"] + "\n\n"
        for qa in paragraph["qas"]:
            if not qa['is_impossible']:
                questions.append(qa['question'])
                correct_answers.append(qa['answers'])
        # remove this break for full article
        break

    # test pipeline here
    logging.info("Started answering.")
    answers = answer_main_pipeline(article_str,questions)
    logging.info("Done answering.")
    tb = pt.PrettyTable()
    tb.field_names = ["Question", "Your answer", "Correct answer"]
    assert len(answers)==len(questions)==len(correct_answers)
    for i in range(len(questions)):
        tb.add_row([questions[i],answers[i],correct_answers[i]])

    logging.debug("Article: %s" % article_str)
    # logging.debug("Preprocessed article: %s" % article_str)
    print(tb)

    input("输入任意字符串按回车键开始下一篇文章")