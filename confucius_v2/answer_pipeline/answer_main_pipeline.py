#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

from sklearn.neural_network import MLPClassifier
import pickle
from answer_pipeline.Question_classification import TransformerSKLearnClassifier
from answer_pipeline.Question_classification import HardClassifier
from answer_pipeline.SentenceFinder import TransformerFinder
from ask_pipeline.preprocessing import preprocess

from answer_pipeline.QuestionAnswer.YXQuestionAnswerer import YesNoAnswerer
from answer_pipeline.QuestionAnswer.XYQuestionAnswerer import WhenAnswerer
from answer_pipeline.QuestionAnswer.XYQuestionAnswerer import WhyAnswerer
from answer_pipeline.QuestionAnswer.XYQuestionAnswerer import WhereAnswerer
from answer_pipeline.QuestionAnswer.XYQuestionAnswerer import WhoAnswerer
from answer_pipeline.QuestionAnswer.FYQuestionAnswerer import GenericHowQuestionAnswerer
from answer_pipeline.QuestionAnswer.FYQuestionAnswerer import NumberHowQuestionAnswerer
from answer_pipeline.QuestionAnswer.FYQuestionAnswerer import WhatWhichQuestionAnswerer as FYWhatWhichQuestionAnswerer
from answer_pipeline.QuestionAnswer.XYQuestionAnswerer import WhatWhichQuestionAnswerer as XYWhatWhichQuestionAnswerer

def answer_main_pipeline(article:str,questions:list)->list:
    logging.debug("Article: %s" %article)
    preprocessed_article = preprocess(article, no_junk=False, with_coref=True)
    logging.info("Generated %d sentences from the article." % len(preprocessed_article))
    logging.debug("Generated sentences:\n"+str(preprocessed_article))

    logging.info("Started classifying questions.")
    # classifier = MLPClassifier(solver='lbfgs')#, alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
    with open("./Models/question_classifier_bert_adam","rb") as f:
        classifier = pickle.load(f)
    trfc = TransformerSKLearnClassifier(transformer_model_name="bert-base-nli-stsb-mean-tokens",classifier=classifier)
    question_tags = trfc.batch_classify(questions)

    import spacy

    spacy_nlp = spacy.load("en_core_web_sm")

    hard_classify = HardClassifier(spacy_nlp)
    question_tags_hard = hard_classify.classify_question_batch(questions)
    final_question_tags = []
    for i in range(len(question_tags)):
        if len(question_tags_hard[i]) == 0:
            final_question_tags.append(question_tags[i])
        else:
            final_question_tags.append(question_tags_hard[i])
    question_tags = final_question_tags

    # for i in range(len(question_tags)):
    #     print(question_tags[i])
    #     print(questions[i])



    logging.info("Done classifying questions.")

    logging.info("Started finding related sentence for questions.")
    finder = TransformerFinder("bert-base-nli-stsb-mean-tokens", preprocessed_article)
    sentences_lists = finder.batch_get_related_sentences_ranking(questions)
    logging.info("Done finding related sentence for questions.")

    logging.info("Started looking for answers.")
    proposed_answers = []
    answerers = {
        "yesno"     : YesNoAnswerer(spacy_nlp),
        "whatwhich" : XYWhatWhichQuestionAnswerer(spacy_nlp),
        "when"      : WhenAnswerer(spacy_nlp),
        "why"       : WhyAnswerer(spacy_nlp),
        "where"     : WhereAnswerer(spacy_nlp),
        "who"       : WhoAnswerer(spacy_nlp),
        "hownumber" : NumberHowQuestionAnswerer(spacy_nlp),
        "how"       : GenericHowQuestionAnswerer(spacy_nlp),
    }
    debug_info = ""
    for i,question in enumerate(questions):
        answer = answerers[question_tags[i]].answer_one(question=question,sentences=sentences_lists[i])
        if len(answer)==0: answer=[["I suppose you know the answer,right?",0]]
        t = "[%s] %s -> %s (Most possible sentence: %s)\n"% (question_tags[i],question,answer,sentences_lists[i][0])
        debug_info += t
        logging.debug(t)
        proposed_answers.append(answer)
    logging.debug("Generated answers:\n" + debug_info)
    return proposed_answers



# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------
# if __name__ == "__main__":

    # answer_main_pipeline('./data/Development_data/set1/a1.txt',questions:list)

