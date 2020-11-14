#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import spacy
import numpy as np
# nlp = spacy.load("en_core_web_sm")
from nltk.corpus import wordnet

import spacy
# nlp = spacy.load("en_core_web_sm")
import logging

from answer_pipeline.QuestionAnswerer import QuestionAnswerer
from nltk.parse import CoreNLPParser
global nlp
nlp = None

# -------------------------------------------------------
#                 Documents & Dev Tools
# -------------------------------------------------------


# -------------------------------------------------------
#                    Answerers go here
# -------------------------------------------------------
class YesNoAnswerer(QuestionAnswerer):
    def __init__(self, spacy_nlp):
        super().__init__()
        global nlp
        nlp = spacy_nlp
        self.nlp = spacy_nlp

    def answer_one(self, question: str, sentences: list) -> list:
        # TODO：这个函数接受一个问题以及该问题相关句子概率列表，返回按概率排序的答案列表
        #  举例：question="What is the meaning of life?"
        #       sentence=[ ["Life is a word.",0.97], ["Life means responsibility", 0.96] ]
        #       return = [ ["Responsibility.",0.533],["A word.", 0.51] ]
        #  请尽量满足return的条件。如果你没有算概率，请保证返回有序的0-1数值；如果你只有一个答案，请也用list包好答案
        Question = question
        result = []
        # for Sentence, prob in sentences:
        Sentence, prob = sentences[0]
        s_dep_tree = get_parse_tree(Sentence)
        # s_dep_tree_word = [s_dep_tree[i][0] for i in range(len(s_dep_tree))]
        s_dep_tree_tag = [s_dep_tree[i][1] for i in range(len(s_dep_tree))]

        s_neg_num = np.sum([np.array(s_dep_tree_tag) == 'neg'])

        q_dep_tree = get_parse_tree(Question)
        # q_dep_tree_word = [q_dep_tree[i][0] for i in range(len(q_dep_tree))]
        q_dep_tree_tag = [q_dep_tree[i][1] for i in range(len(q_dep_tree))]

        q_neg_num = np.sum([np.array(q_dep_tree_tag) == 'neg'])

        answer = False
        if s_neg_num % 2 == q_neg_num % 2:
            answer = True

        s_pos = pos_tagger(Sentence)
        q_pos = pos_tagger(Question)

        s_target_lemma = get_adj_adv_verb(s_pos)
        q_target_lemma = get_adj_adv_verb(q_pos)

        q_different = [lemma for lemma in q_target_lemma if lemma not in s_target_lemma]
        s_different = [lemma for lemma in s_target_lemma if lemma not in q_target_lemma]

        dif_num = check_similarity(q_different, s_different)

        if dif_num > 0:
            answer = not answer

        # s_NN_DATE = get_NN_DATE(s_pos)
        # q_NN_DATE = get_NN_DATE(q_pos)
        #
        # q_different_NN_DATE = [lemma for lemma in q_NN_DATE if lemma not in s_NN_DATE]
        # if len(q_different_NN_DATE) > 0:
        #     answer = False

        if answer == True:
            result_0 = 'Yes.'
        else:
            result_0 = 'No.'
        result.append([result_0, prob])
        return result

# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------

# 如果你有什么需要共用的辅助函数，请写在这里
# Yuxin's functions
def get_parse_tree(sentence):
    doc = nlp(sentence)
    result = []
    for token in doc:
        result.append(
            (token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children]))
    return result

def pos_tagger(txt):
    parsed_text = nlp(txt)
    wordlist = []
    for token in parsed_text:
        # display the token's orthographic representation, lemma, part of speech, and entity type (which is empty if the token is not part of a named entity)
        tagged = token.orth_, token.lemma_, token.tag_, token.ent_type_
        wordlist.append(tagged)

    return wordlist

def check_similarity(q_different, s_different):
    dif = 0
    for word in q_different:
        synonyms = []
        antonyms = []

        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())

        syn = set(synonyms)
        ant = set(antonyms)

        if len(list(set(s_different).intersection(ant))) != 0:
            dif += 1
    return dif

def get_NN_DATE(pos):
    target = []
    for word, lemma, tag, entity in pos:
        if tag[0:2] == 'NN' or entity == 'DATE':
            target.append(lemma)
    return target

def get_adj_adv_verb(pos):
    target = []
    for word, lemma, tag, entity in pos:
        if tag[0:2] == 'VB' or tag == 'RB' or tag == 'JJ':
            target.append(lemma)
    return target


# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------

if __name__ == '__main__':
    # 如果你有直接运行的测试语句，请写在这里
    pass