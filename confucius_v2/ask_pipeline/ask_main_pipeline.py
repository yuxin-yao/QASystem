#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
from ask_pipeline.question_obfuscation import according_obf, reason_obf, please_explain_obf
from ask_pipeline.sort import sort_question
import random
from ask_pipeline.preprocessing import batch_process

from ask_pipeline.preprocessing import preprocess
import logging
from ask_pipeline.QuestionGenerator import *
from ask_pipeline.preprocessing import preprocess

def ask_main_pipeline(article:str,nquestions:int)->list:
    preprocessed_article = preprocess(article,no_junk=True,with_coref=True)
    logging.info("Generated %d sentences from the article." % len(preprocessed_article))
    # logging.debug("Generated sentences:\n"+str(preprocessed_article))

    logging.info("Started generating questions.")
    random.shuffle(preprocessed_article)
    proposed_questions = []
    debug_info=""
    for i in preprocessed_article:
        generator = QuestionGenerator(sent=i.text)
        generated = generator.generate_all()
        debug_info+="%s -> %s\n"%(str(i),str(generated))
        proposed_questions.extend(generated)

        # bad fix for timeout
        if len(proposed_questions)>10*nquestions and nquestions>0:
            logging.info("Stopped generating questions because it is too much.")
            break

    proposed_questions = list(set(proposed_questions))
    logging.info("Adding obfs questions.")
    result = proposed_questions + according_obf(proposed_questions)
    result = result + reason_obf(proposed_questions)
    # result = result + which_obf(proposed_questions)
    proposed_questions = result + please_explain_obf(proposed_questions)

    result= list(set(result))
    logging.debug("Generated questions:\n"+debug_info)
    logging.info("Generated %d questions from the article." % len(proposed_questions))


    logging.info("Started sorting.")
    sorted_question = sort_question(result, nquestions)
    logging.info("Done sorting.")

    # if nquestions == -1: return sorted_question
    # while len(sorted_question)<nquestions:
    #     logging.warning("Not enough question, will repeat questions.")
    #     sorted_question.extend(sorted_question)

    return sorted_question[:nquestions]

