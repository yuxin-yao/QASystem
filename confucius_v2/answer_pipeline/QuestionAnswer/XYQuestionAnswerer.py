
# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import copy

import spacy

# nlp = spacy.load("en_core_web_sm")

import logging

from answer_pipeline.QuestionAnswerer import QuestionAnswerer
from nltk.parse import CoreNLPParser
import numpy as np

# os.chdir("/Users/hequanhong/Desktop/CS411_Group_Project-master/stanford-corenlp-full-2018-10-05")

# 把下列代码输入含有"stanford-corenlp-full-2018-10-05"包的terminal的directory里
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 1500

parser = CoreNLPParser(url='http://localhost:9001')
global nlp
nlp = None

# -------------------------------------------------------
#                 Documents & Dev Tools
# -------------------------------------------------------
# 把你用到的所有参考资料堆在这里，格式是简介和链接，举例：

# spacy dependency tree visualizer
# https://explosion.ai/demos/displacy

# dependency tree component lookup
# https://blog.csdn.net/lihaitao000/article/details/51812618

# -------------------------------------------------------
#                    Answerers go here
# -------------------------------------------------------
# 把你写的回答类放在这里，请按照下面的模版实现接口：

class WhenAnswerer(QuestionAnswerer):
    def __init__(self, spacy_nlp):
        super().__init__()
        global nlp
        nlp = spacy_nlp
        self.nlp = spacy_nlp

    def answer_one(self,question:str,sentences:list)->list:
        q_spacy = nlp(question)
        q_wh_dep = ""
        q_wh_parent = ""
        for i in range(len(q_spacy)):
            if q_spacy[i].dep_ == "ROOT":
                q_root = q_spacy[i].text
            if q_spacy[i].lemma_ == "when":
                q_wh_dep = str(q_spacy[i].dep_)

        potential_answer_total = []

        for sent_pair in sentences[:5]:
            candidate = dict()
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_spacy = nlp(sent)

            retry = 3
            while retry != 0:
                try:
                    sent_parse_tree = next(parser.raw_parse(sent))
                    PP_finder(sent_parse_tree)
                    break
                except:
                    logging.exception("CoreNLP ERROR!! Retrying count: %d" % retry)
                    retry -= 1
            if retry == 0:
                logging.exception("CoreNLP UNRECOVERABLE ERROR!!")
                continue

            answers = []  # potential when and where answers' list (start, end)
            for answer_tree in PP_tree:
                answer = answer_tree.leaves()
                for i in range(len(sent_spacy)):
                    if str(sent_spacy[i].text) == answer[0]:
                        start = i
                        for j in range(i, i + len(answer)):
                            if str(sent_spacy[j].text) == answer[j - i]:
                                if str(sent_spacy[j].text) == answer[-1]:
                                    end = j + 1
                                    answers.append((start, end))
                                    break
                            else:
                                break

            result = []
            when_index = when_phrase(sent_spacy, answers)

            SBAR_finder_wh(sent_parse_tree)
            clause_answers = []

            for answer_tree in SBAR_tree:
                answer = answer_tree.leaves()
                if answer[0] == "when":
                    for i in range(len(sent_spacy)):
                        if str(sent_spacy[i].text) == answer[0]:
                            start = i
                            for j in range(i, i + len(answer)):
                                if str(sent_spacy[j].text) == answer[j - i]:
                                    if str(sent_spacy[j].text) == answer[-1]:
                                        end = j + 1
                                        clause_answers.append((start, end))
                                        break
                                else:
                                    break
                else:
                    break
            when_index += clause_answers

            potential_answer = []

            for answer_pair in when_index:
                single_answer_dict = dict()
                single_answer_dict["index"] = answer_pair
                single_answer_dict["score"] = 0
                re = ""
                for i in range(answer_pair[0], answer_pair[1]):
                    re += sent_spacy[i].text + " "

                single_answer_dict["answer"] = re.rstrip()
                potential_answer.append(single_answer_dict)

            for item in potential_answer:
                index = item["index"]
                if item["answer"] == "It" or item["answer"] == "it":
                    item["score"] = 0
                    break

                for i in range(index[0], index[1]):
                    # criteria 1: if the answer have the same root as the question
                    if str(sent_spacy[i].head) == q_root:
                        item["score"] += 0.05
                        break

                for i in range(index[0], index[1]):
                    # criteria 2: if the answer have the same dep_ as wh-word in the question
                    if str(sent_spacy[i].dep_) == q_wh_dep:
                        item["score"] += 0.2
                        break

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].text) == q_wh_parent:
                        item["score"] += 0.2
                        break

                for i in range(len(item["answer"])):
                    if item["answer"][i].lower() not in question.lower():
                        item["score"] += 0.3 / len(item["answer"])

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].ent_type_) != "":
                        item["score"] += 0.25
                        break

            max_score = 0
            max_answer = None
            for item in potential_answer:
                if item["score"] > max_score:
                    max_score = item["score"]
                    max_answer = item["answer"]

            candidate["answer"] = max_answer
            candidate["score"] = max_score * sent_prob

            potential_answer_total.append(candidate)

        result = []
        for item in potential_answer_total:
            if item["answer"] != None:
                result.append([item["answer"], item["score"]])

        answer_list = sorted(result, key=lambda i: i[1])[::-1]

        if len(answer_list) == 0:
            whatwhich = WhatWhichQuestionAnswerer(spacy_nlp=self.nlp)
            return whatwhich.answer_one(question, sentences, tested_type="when")

        return answer_post_process(answer_list)

class WhereAnswerer(QuestionAnswerer):
    def __init__(self, spacy_nlp):
        super().__init__()
        nlp = spacy_nlp
        self.nlp = spacy_nlp

    def answer_one(self, question: str, sentences: list) -> list:
        q_spacy = nlp(question)
        q_wh_dep = ""
        q_wh_parent = ""
        for i in range(len(q_spacy)):
            if q_spacy[i].dep_ == "ROOT":
                q_root = q_spacy[i].text
            if q_spacy[i].lemma_ == "where":
                q_wh_dep = str(q_spacy[i].dep_)

        potential_answer_total = []

        for sent_pair in sentences[:5]:
            candidate = dict()
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_spacy = nlp(sent)

            retry = 3
            while retry != 0:
                try:
                    sent_parse_tree = next(parser.raw_parse(sent))
                    PP_finder(sent_parse_tree)
                    break
                except:
                    logging.exception("CoreNLP ERROR!! Retrying count: %d" % retry)
                    retry -= 1
            if retry == 0:
                logging.exception("CoreNLP UNRECOVERABLE ERROR!!")
                continue

            answers = []  # potential when and where answers' list (start, end)
            for answer_tree in PP_tree:
                answer = answer_tree.leaves()
                for i in range(len(sent_spacy)):
                    if str(sent_spacy[i].text) == answer[0]:
                        start = i
                        for j in range(i, i + len(answer)):
                            if str(sent_spacy[j].text) == answer[j - i]:
                                if str(sent_spacy[j].text) == answer[-1]:
                                    end = j + 1
                                    answers.append((start, end))
                                    break
                            else:
                                break

            where_indexes = where_phrase(sent_spacy, answers)

            potential_answer = []

            for answer_pair in where_indexes:
                single_answer_dict = dict()
                single_answer_dict["index"] = answer_pair
                single_answer_dict["score"] = 0
                re = ""
                for i in range(answer_pair[0], answer_pair[1]):
                    re += sent_spacy[i].text + " "

                single_answer_dict["answer"] = re.rstrip()
                potential_answer.append(single_answer_dict)

            for item in potential_answer:
                index = item["index"]
                for i in range(index[0], index[1]):
                    # criteria 1: if the answer have the same root as the question
                    if str(sent_spacy[i].head) == q_root:
                        item["score"] += 0.1
                        break

                for i in range(index[0], index[1]):
                    # criteria 2: if the answer have the same dep_ as wh-word in the question
                    if str(sent_spacy[i].dep_) == q_wh_dep:
                        item["score"] += 0.2
                        break

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].text) == q_wh_parent:
                        item["score"] += 0.2
                        break

                for i in range(len(item["answer"])):
                    if item["answer"][i].lower() not in question.lower():
                        item["score"] += 0.3 / len(item["answer"])

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].ent_type_) != "":
                        item["score"] += 0.2
                        break

            max_score = 0
            max_answer = None
            for item in potential_answer:
                if item["score"] > max_score:
                    max_score = item["score"]
                    max_answer = item["answer"]

            candidate["answer"] = max_answer
            candidate["score"] = max_score * sent_prob

            potential_answer_total.append(candidate)

        result = []
        for item in potential_answer_total:
            if item["answer"] != None:
                result.append([item["answer"], item["score"]])

        answer_list = sorted(result, key=lambda i: i[1])[::-1]

        if len(answer_list) == 0:
            whatwhich = WhatWhichQuestionAnswerer(spacy_nlp=self.nlp)
            return whatwhich.answer_one(question, sentences, tested_type="where")

        return answer_post_process(answer_list)

class WhoAnswerer(QuestionAnswerer):
    def __init__(self, spacy_nlp):
        super().__init__()
        nlp = spacy_nlp
        self.nlp = spacy_nlp

    def answer_one(self,question:str,sentences:list)->list:
        q_spacy = nlp(question)
        q_wh_dep = ""
        q_wh_parent = ""
        for i in range(len(q_spacy)):
            if q_spacy[i].dep_ == "ROOT":
                q_root = q_spacy[i].text
            if q_spacy[i].lemma_ == "who":
                q_wh_dep = str(q_spacy[i].dep_)

        potential_answer_total = []

        for sent_pair in sentences[:5]:
            candidate = dict()
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_spacy = nlp(sent)
            who_indexes = who_phrase(sent_spacy)

            if who_indexes == []:
                who_indexes = whose_phrase(sent_spacy)

            #print(who_indexes)

            potential_answer = []

            for answer_pair in who_indexes:
                single_answer_dict = dict()
                single_answer_dict["index"] = answer_pair
                single_answer_dict["score"] = 0
                re = ""
                for i in range(answer_pair[0], answer_pair[1]):
                    if sent_spacy[i].text == "'s":
                        re = re.rstrip()
                        re += sent_spacy[i].text + " "
                    else:
                        re += sent_spacy[i].text + " "

                single_answer_dict["answer"] = re.rstrip()
                potential_answer.append(single_answer_dict)

            for item in potential_answer:

                index = item["index"]
                for i in range(index[0], index[1]):
                    # criteria 1: if the answer have the same root as the question
                    if str(sent_spacy[i].head) == q_root:
                        item["score"] += 0.05
                        break
                for i in range(index[0], index[1]):
                    # criteria 2: if the answer have the same dep_ as wh-word in the question
                    if str(sent_spacy[i].dep_) == q_wh_dep:
                        item["score"] += 0.2
                        break

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].text) == q_wh_parent:
                        item["score"] += 0.2
                        break

                for i in range(len(item["answer"])):
                    if item["answer"][i].lower() not in question.lower():
                        item["score"] += 0.3 / len(item["answer"])

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].ent_type_) != "":
                        item["score"] += 0.25
                        break
            max_score = 0
            max_answer = None

            for item in potential_answer:
                if type(item["answer"]) == str:
                    if item["score"] > max_score:
                        max_score = item["score"]
                        max_answer = item["answer"]

                candidate["answer"] = max_answer
                candidate["score"] = max_score * sent_prob

                potential_answer_total.append(candidate)

        result = []
        for item in potential_answer_total:
            if item["answer"] != None:
                result.append([item["answer"], item["score"]])

        answer_list = sorted(result, key=lambda i: i[1])[::-1]

        if len(answer_list) == 0:
            whatwhich = WhatWhichQuestionAnswerer(spacy_nlp=self.nlp)
            return whatwhich.answer_one(question, sentences, tested_type="who")

        return answer_post_process(answer_list)


class WhyAnswerer(QuestionAnswerer):
    def __init__(self, spacy_nlp):
        super().__init__()
        nlp = spacy_nlp
        self.nlp = spacy_nlp

    def answer_one(self,question:str,sentences:list)->list:
        q_spacy = nlp(question)
        q_wh_dep = ""
        q_wh_parent = ""
        for i in range(len(q_spacy)):
            if q_spacy[i].dep_ == "ROOT":
                q_root = q_spacy[i].text
            if q_spacy[i].lemma_ == "why":
                q_wh_dep = str(q_spacy[i].dep_)

        potential_answer_total = []

        for sent_pair in sentences[:5]:
            candidate = dict()
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_spacy = nlp(sent)

            retry = 3
            while retry != 0:
                try:
                    sent_parse_tree = next(parser.raw_parse(sent))
                    SBAR_finder_reason(sent_parse_tree)
                    break
                except:
                    logging.exception("CoreNLP ERROR!! Retrying count: %d" % retry)
                    retry -= 1
            if retry == 0:
                logging.exception("CoreNLP UNRECOVERABLE ERROR!!")
                continue

            answers = []  # potential when and where answers' list (start, end)
            for answer_tree in SBAR_tree_reason:
                answer = answer_tree.leaves()
                for i in range(len(sent_spacy)):
                    if str(sent_spacy[i].text) == answer[0]:
                        start = i
                        for j in range(i, i + len(answer)):
                            if str(sent_spacy[j].text) == answer[j - i]:
                                if str(sent_spacy[j].text) == answer[-1]:
                                    end = j + 1
                                    answers.append((start, end))
                                    break
                            else:
                                break

            why_index = answers

            potential_answer = []

            for answer_pair in why_index:
                single_answer_dict = dict()
                single_answer_dict["index"] = answer_pair
                single_answer_dict["score"] = 0
                re = ""
                for i in range(answer_pair[0], answer_pair[1]):
                    re += sent_spacy[i].text + " "

                single_answer_dict["answer"] = re.rstrip()
                potential_answer.append(single_answer_dict)

            for item in potential_answer:
                index = item["index"]
                for i in range(index[0], index[1]):
                    # criteria 1: if the answer have the same root as the question
                    if str(sent_spacy[i].head) == q_root:
                        item["score"] += 0.1
                        break

                for i in range(index[0], index[1]):
                    # criteria 2: if the answer have the same dep_ as wh-word in the question
                    if str(sent_spacy[i].dep_) == q_wh_dep:
                        item["score"] += 0.2
                        break

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].text) == q_wh_parent:
                        item["score"] += 0.2
                        break

                for i in range(len(item["answer"])):
                    if item["answer"][i].lower() not in question.lower():
                        item["score"] += 0.3 / len(item["answer"])

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].ent_type_) != "":
                        item["score"] += 0.2
                        break

            max_score = 0
            max_answer = None
            for item in potential_answer:
                if item["score"] > max_score:
                    max_score = item["score"]
                    max_answer = item["answer"]

            candidate["answer"] = max_answer
            candidate["score"] = max_score * sent_prob

            potential_answer_total.append(candidate)

        result = []
        for item in potential_answer_total:
            if item["answer"] != None:
                result.append([item["answer"], item["score"]])

        answer_list = sorted(result, key = lambda i: i[1])[::-1]

        if len(answer_list) == 0:
            whatwhich = WhatWhichQuestionAnswerer(spacy_nlp=self.nlp)
            return whatwhich.answer_one(question, sentences, tested_type="why")

        return answer_post_process(answer_list)


class WhatWhichQuestionAnswerer(QuestionAnswerer):
    def __init__(self, spacy_nlp):
        super().__init__()
        nlp = spacy_nlp
        self.nlp = spacy_nlp

    def answer_one(self, question:str, sentences:list, tested_type = None)->list:
        q_spacy = self.nlp(question)
        q_wh_parent = ""
        q_wh_dep = ""
        for i in range(len(q_spacy)):
            if q_spacy[i].dep_ == "ROOT":
                q_root = q_spacy[i].text
            if q_spacy[i].lemma_ == "what" or q_spacy[i].lemma_ == "which":
                q_wh_dep = str(q_spacy[i].dep_)
            if len([children for children in q_spacy[i].children]) > 0:
                if (str([children for children in q_spacy[i].children][0]) == "what" or str(
                        [children for children in q_spacy[i].children][0]) == "What") or (
                        str([children for children in q_spacy[i].children][0]) == "which" or str(
                        [children for children in q_spacy[i].children][0]) == "Which"):
                    q_wh_parent = q_spacy[i].text

        potential_answer_total = []

        for sentence in sentences[:5]:#only want to check the first 5 sentence
            candidate = dict()
            sent = sentence[0]
            sent_prob = sentence[1]
            sent_spacy = self.nlp(sent)


            potential_answer = []

            if tested_type != "who" or tested_type == None:
                who_indexes = who_phrase(sent_spacy)
                if who_indexes == []:
                    who_indexes = whose_phrase(sent_spacy)

                # print(who_indexes)
                for answer_pair in who_indexes:
                    single_answer_dict = dict()
                    single_answer_dict["index"] = answer_pair
                    single_answer_dict["score"] = 0
                    re = ""
                    for i in range(answer_pair[0], answer_pair[1]):
                        if sent_spacy[i].text == "'s":
                            re = re.rstrip()
                            re += sent_spacy[i].text + " "
                        else:
                            re += sent_spacy[i].text + " "

                    single_answer_dict["answer"] = re.rstrip()
                    potential_answer.append(single_answer_dict)

            retry = 3
            while retry != 0:
                try:
                    sent_parse_tree = next(parser.raw_parse(sent))
                    break
                except:
                    logging.exception("CoreNLP ERROR!! Retrying count: %d" % retry)
                    retry -= 1
            if retry == 0:
                logging.exception("CoreNLP UNRECOVERABLE ERROR!!")
                continue

            PP_finder(sent_parse_tree)
            answers = []  # potential when and where answers' list (start, end)
            for answer_tree in PP_tree:
                answer = answer_tree.leaves()
                for i in range(len(sent_spacy)):
                    if str(sent_spacy[i].text) == answer[0]:
                        start = i
                        for j in range(i, i + len(answer)):
                            if str(sent_spacy[j].text) == answer[j - i]:
                                if str(sent_spacy[j].text) == answer[-1]:
                                    end = j + 1
                                    answers.append((start, end))
                                    break
                            else:
                                break

            if tested_type != "when" or tested_type == None:
                when_index = when_phrase(sent_spacy, answers)

                SBAR_finder_wh(sent_parse_tree)
                clause_answers = []

                for answer_tree in SBAR_tree:
                    answer = answer_tree.leaves()
                    if answer[0] == "when":
                        for i in range(len(sent_spacy)):
                            if str(sent_spacy[i].text) == answer[0]:
                                start = i
                                for j in range(i, i + len(answer)):
                                    if str(sent_spacy[j].text) == answer[j - i]:
                                        if str(sent_spacy[j].text) == answer[-1]:
                                            end = j + 1
                                            clause_answers.append((start, end))
                                            break
                                    else:
                                        break
                    else:
                        break
                when_index += clause_answers

                for answer_pair in when_index:
                    single_answer_dict = dict()
                    single_answer_dict["index"] = answer_pair
                    single_answer_dict["score"] = 0
                    re = ""
                    for i in range(answer_pair[0], answer_pair[1]):
                        re += sent_spacy[i].text + " "

                    single_answer_dict["answer"] = re.rstrip()
                    potential_answer.append(single_answer_dict)

            if tested_type != "where" or tested_type == None:

                where_indexes = where_phrase(sent_spacy, answers)

                for answer_pair in where_indexes:
                    single_answer_dict = dict()
                    single_answer_dict["index"] = answer_pair
                    single_answer_dict["score"] = 0
                    re = ""
                    for i in range(answer_pair[0], answer_pair[1]):
                        re += sent_spacy[i].text + " "

                    single_answer_dict["answer"] = re.rstrip()
                    potential_answer.append(single_answer_dict)

            if tested_type != "why" or tested_type == None:
                SBAR_finder_reason(sent_parse_tree)
                answers = []
                for answer_tree in SBAR_tree_reason:
                    answer = answer_tree.leaves()
                    for i in range(len(sent_spacy)):
                        if str(sent_spacy[i].text) == answer[0]:
                            start = i
                            for j in range(i, i + len(answer)):
                                if str(sent_spacy[j].text) == answer[j - i]:
                                    if str(sent_spacy[j].text) == answer[-1]:
                                        end = j + 1
                                        answers.append((start, end))
                                        break
                                else:
                                    break

                why_index = answers

                for answer_pair in why_index:
                    single_answer_dict = dict()
                    single_answer_dict["index"] = answer_pair
                    single_answer_dict["score"] = 0
                    re = ""
                    for i in range(answer_pair[0], answer_pair[1]):
                        re += sent_spacy[i].text + " "

                    single_answer_dict["answer"] = re.rstrip()
                    potential_answer.append(single_answer_dict)

            # print(potential_answer)
            #start what
            what_index = what_org_phrase(sent_spacy)
            for index_pair in what_index:
                single_answer_dict = dict()
                single_answer_dict["index"] = index_pair
                single_answer_dict["score"] = 0

                re = []
                for i in range(index_pair[0], index_pair[1]):
                    re.append(sent_spacy[i].text)
                single_answer_dict["answer"] = re
                potential_answer.append(single_answer_dict)

            for np in sent_spacy.noun_chunks:
                single_answer_dict = dict()
                single_answer_dict["answer"] = str(np.text)
                single_answer_dict["score"] = 0

                for i in range(len(sent_spacy)):
                    an = single_answer_dict["answer"].split(" ")

                    if sent_spacy[i].text == an[0]:
                        start = i
                        for j in range(len(an)):

                            if i + j == len(sent_spacy):
                                break

                            if sent_spacy[i + j].text != an[j]:
                                break

                            elif sent_spacy[i + j].text == an[-1]:
                                end = i + j
                                single_answer_dict["index"] = (start, end + 1)
                                if check_inclusive_tuple(what_index, (start, end + 1)):
                                    potential_answer.append(single_answer_dict)
                                    break

            for item in potential_answer:

                index = item["index"]
                for i in range(index[0], index[1]):
                    # criteria 1: if the answer have the same root as the question
                    if str(sent_spacy[i].head) == q_root:
                        item["score"] += 0.05
                        break

                for i in range(index[0], index[1]):
                    # criteria 2: if the answer have the same dep_ as wh-word in the question
                    if str(sent_spacy[i].dep_) == q_wh_dep:
                        item["score"] += 0.25
                        break

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].text) == q_wh_parent:
                        item["score"] += 0.2
                        break

                for i in range(len(item["answer"])):
                    if i == len(item["answer"]) - 1:
                        item["score"] += 0.3
                    elif item["answer"][i].lower() not in question.lower() and i < len(item["answer"]) - 1:
                        continue
                    else:
                        break

                for i in range(index[0], index[1]):
                    # criteria 3:
                    if str(sent_spacy[i].ent_type_) != "":
                        item["score"] += 0.2
                        break

            # print(potential_answer)
            max_score = 0
            max_answer = None
            for item in potential_answer:
                if item["score"] > max_score:
                    max_score = item["score"]
                    max_answer = item["answer"]

            candidate["answer"] = max_answer
            candidate["score"] = max_score * sent_prob

            potential_answer_total.append(candidate)

        result = []
        for item in potential_answer_total:
            if item["answer"] != "":
                result.append([item["answer"], item["score"]])

        answer_list = sorted(result, key = lambda i: i[1])[::-1]
        if len(answer_list) == 0:
            return answer_post_process([sentences[0][0], 0])

        return answer_post_process(answer_list)


#answer question if there is no Entity




# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------

# TODO: 如果你有什么需要共用的辅助函数，请写在这里
#
# for when question in dependent clause
NP_leaf_tree = []


def NP_leaf_finder(parse_tree):
    global NP_leaf_tree
    NP_leaf_tree = []
    NP_leaf_tree_helper(parse_tree)


def NP_leaf_tree_helper(parse_tree):
    for subtree in parse_tree:
        if type(subtree) == nltk.tree.Tree:
            if subtree.label().strip() == 'NP':
                count_sub = 0
                for sub_subtree in subtree:
                    count_sub += 1
                    # print(count_sub, sub_subtree)
                    if sub_subtree.label().strip() == "NP" and count_sub == 1:
                        NP_leaf_tree_helper(subtree)
                    elif sub_subtree.label().strip() == "PP" and count_sub == 2:
                        NP_leaf_tree_helper(subtree)

                    elif count_sub >= 2:
                        break

                    else:
                        # global NP_leaf_tree
                        global NP_leaf_tree
                        if subtree not in NP_leaf_tree:
                            NP_leaf_tree.append(subtree)
                            break

            else:
                NP_leaf_tree_helper(subtree)
        else:
            break



SBAR_tree = []

def SBAR_finder_wh(parse_tree):
    global SBAR_tree
    SBAR_tree = []
    SBAR_finder_helper_wh(parse_tree)


def SBAR_finder_helper_wh(parse_tree):
    for subtree in parse_tree:
        if type(subtree) == nltk.tree.Tree:
            if subtree.label().strip() == 'SBAR':
                count = 0
                for sub_subtree in subtree:
                    count += 1
                    if sub_subtree.label().strip() == 'WHADVP' and count == 1:
                        global SBAR_tree
                        SBAR_tree.append(subtree)
                    elif sub_subtree.label().strip() == 'S' and count == 2:
                        SBAR_finder_helper_wh(sub_subtree)
                    elif count >= 2:
                        break
            else:
                SBAR_finder_helper_wh(subtree)
        else:
            break


# for generating and answering why questions
SBAR_tree_reason = []


def SBAR_finder_reason(parse_tree):
    global SBAR_tree_reason
    SBAR_tree_reason = []
    SBAR_finder_helper_reason(parse_tree)


def SBAR_finder_helper_reason(parse_tree):
    for subtree in parse_tree:
        if type(subtree) == nltk.tree.Tree:
            if subtree.label().strip() == 'SBAR':
                count = 0
                for sub_subtree in subtree:
                    count += 1
                    if sub_subtree.label().strip() == 'IN' and count == 1:
                        global SBAR_tree_reason
                        SBAR_tree_reason.append(subtree)
                    elif sub_subtree.label().strip() == 'S' and count == 2:
                        SBAR_finder_helper_reason(sub_subtree)
            else:
                SBAR_finder_helper_reason(subtree)
        else:
            break


# find PP
PP_tree = []


def PP_finder(parse_tree):
    global PP_tree
    PP_tree = []
    PP_finder_helper(parse_tree)


def PP_finder_helper(parse_tree):
    for subtree in parse_tree:
        if type(subtree) == nltk.tree.Tree:
            if subtree.label().strip() == 'PP':
                count_sub = 0
                for sub_subtree in subtree:
                    count_sub += 1
                    # print(count_sub, sub_subtree)
                    if sub_subtree.label().strip() == "NP" and count_sub == 2:

                        global PP_tree
                        PP_tree.append(subtree)

                    elif sub_subtree.label().strip() == "PP" and count_sub == 1:
                        PP_finder_helper(subtree)

                    elif sub_subtree.label().strip() == "S" and count_sub == 2:
                        PP_finder_helper(sub_subtree)
            else:
                PP_finder_helper(subtree)
        else:
            break


def check_inclusive_tuple(result_list, pair):
    for item in result_list:
        if pair[0] >= item[0] and pair[1] <= item[1]:
            return False
    return True


def what_phrase(sent_spacy, sent):
    result_answer = []
    result = []
    #parse_tree = next(parser.raw_parse(sent))  # expecting parse tree to be VB(is/are...) NP NP
    retry = 3
    while retry >= 0:
        if retry == 0:
            logging.error("CoreNLP unrecoverable ERROR!!")
            # TODO: 如果corenlp一直用不了，那该怎么处理？在这里实现解决方案
            return result
        else:
            try:
                parse_tree = next(parser.raw_parse(sent))  # get parse tree
                break
            except:
                logging.exception("CoreNLP ERROR!! Retrying.")
                retry -= 1

    for subtree in parse_tree:  # 只有一个loop
        if subtree.label().strip() == "NP":
            count = 0
            for sub_subtree in subtree:
                if count == 2:
                    break
                if sub_subtree.label().strip() == "NP":
                    count += 1
                if (count < 2 and sub_subtree.label().strip() != "NP") or (count > 2):  # 没有合适的 what answer
                    return result

    for subtree in parse_tree:  # getting into the first subtree
        if subtree.label().strip() == "NP":
            for sub_subtree in subtree:
                if sub_subtree.label().strip() == "NP":
                    result_answer.append(sub_subtree.leaves())
        else:
            break

    # match the answer to the index of the tree
    for answer in result_answer:
        # print(answer)
        for i in range(len(sent_spacy)):
            if str(sent_spacy[i].text) == answer[0]:
                start = i
                for j in range(i, i + len(answer)):
                    if str(sent_spacy[j].text) == answer[j - i]:
                        if str(sent_spacy[j].text) == answer[-1]:
                            end = j + 1
                            result.append((start, end))
                            break
                    else:
                        break

    return result

def what_org_phrase(sent_spacy):
    result = []
    i = 0
    while i < len(sent_spacy):
        if sent_spacy[i].ent_type_ == "ORG" and sent_spacy[i].pos_ != "PROPN":
            start = i
            end = i + 1
            for j in range(i + 1, len(sent_spacy)):
                if not (sent_spacy[j].ent_type_ == "ORG" and sent_spacy[i].pos_ != "PROPN"):
                    end = j
                    result.append((start, end))
                    i = end
                    break
            i += 1
        else:
            i += 1
    return result

def who_phrase(sent_spacy): #return who_phrase index pairs
    result = []
    i = 0
    check_while = 0
    check_index = 0
    while i < len(sent_spacy):
        if check_index == i and check_while > 1:
            break
        if check_index == i and check_while <= 1:
            check_while += 1
        if check_index != i:
            check_while = 0
            check_index = i
        if (sent_spacy[i].ent_type_ == "PERSON" and sent_spacy[i].pos_ == "PROPN"):
            start = i
            end = i + 1
            for j in range(i + 1, len(sent_spacy)):
                if not ((sent_spacy[j].ent_type_ == "PERSON" and sent_spacy[j].pos_ == "PROPN")):
                    end = j
                    result.append((start, end))
                    i = end
                    break
        else:
            i += 1
    return result

def whose_phrase(sent_spacy):
    result = []
    for i in range(len(sent_spacy)):
        if sent_spacy[i].lemma_ == "'s":
            for j in range (0, i):
                if not (sent_spacy[i - 1 - j].ent_type_ == "PERSON" or sent_spacy[i - 1 - j].text[0].isupper()):
                    if check_inclusive_tuple(result, (i - j, i+1)):
                        result.append((i - j, i+1))
                        break
    return result

# answer_index is a list
# return an index of when phrase
def when_phrase(sent_spacy, answers_index):
    result = []
    for answer_pair in answers_index:
        start, end = answer_pair
        for i in range(start, end):
            if sent_spacy[i].ent_type_ == "DATE" or sent_spacy[i].ent_type_ == "TIME":
                if (start, end) not in result:
                    result.append((start, end))
                    break

    # check if there is a when-phrase without 介词
    for i in range(len(sent_spacy)):
        if (sent_spacy[i].ent_type_ == "TIME" or sent_spacy[i].ent_type_ == "DATE") and sent_spacy[i].pos_ != 'AUX' and \
                sent_spacy[i].pos_ != 'VERB':
            for j in range(i + 1, len(sent_spacy)):
                if (sent_spacy[i].ent_type_ != "TIME" or sent_spacy[i].ent_type_ != "DATE" or sent_spacy[
                    i].pos_ == 'AUX' or sent_spacy[i].pos_ == 'VERB'):
                    if check_inclusive_tuple(result, (i, j)):
                        result.append((i, j))
                        break

    return result


# The answer phrase is a prepositional phrase whose object is tagged noun.location and whose preposition is one of the following: on, in, at, over, to
# answer_index is a list
# return the index of where phrase
def where_phrase(sent_spacy, answers_index):
    result = []
    for answer_pair in answers_index:
        start, end = answer_pair
        for i in range(start, end):
            if (sent_spacy[i].ent_type_ == "GPE" or sent_spacy[i].ent_type_ == "LOC"):
                if sent_spacy[start].lemma_ == "of":
                    if (start, end) not in result:
                        result.append((start, end))  # need to include of in the where phrase
                        break
                elif (sent_spacy[start].lemma_ == "on" or sent_spacy[start].lemma_ == "in" or sent_spacy[
                    start].lemma_ == "at" or sent_spacy[start].lemma_ == "over" or sent_spacy[start].lemma_ == "to" or
                      sent_spacy[start].lemma_ == "under"):
                    if (start + 1, end) not in result:
                        result.append((start + 1, end))  # need not to preserve介词
                        break

    # check if there is the where-phrase without介词
    for i in range(len(sent_spacy)):
        if sent_spacy[i].ent_type_ == "GPE" or sent_spacy[i].ent_type_ == "LOC":
            for j in range(i + 1, len(sent_spacy)):
                if sent_spacy[j].ent_type_ != "GPE" and sent_spacy[j].ent_type_ != "LOC":
                    if check_inclusive_tuple(result, (i, j)):
                        result.append((i, j))
                    break

    return result

def answer_post_process(answer_list):
    result = []
    for pair in answer_list:
        answer, prob = pair
        if type(answer) == str:
            ans = ""
            ans_list = answer.split(" ")
            for i in range(len(ans_list)):
                if i == 0:
                    ans += ans_list[i].capitalize() + " "
                else:
                    ans += ans_list[i] + " "
            ans = ans.rstrip() + "."
            result.append([ans, prob])
    return result



# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------

# TODO：如果你有与测试你的代码相关的函数，请写在这里

if __name__ == '__main__':
    # TODO：如果你有直接运行的测试语句，请写在这里
    who = WhoAnswerer()
    where = WhereAnswerer()
    why = WhyAnswerer()
    when = WhenAnswerer()
    # print(test.answer_when("When will Igglybuff evolve?",
    #                        [["Igglybuff evolves when it reaches a certain point of happiness.", 0.95]]))
    print(who.answer_one("Who marries John?", [["Mary Smith marries John and Cindy eats apple.", 0.95]]))
    print(where.answer_one("Where will MP3 files most commonly be found?", [["MPEG-1 or MPEG-2 Audio Layer III, more commonly referred to as MP3, is an audio coding format for digital audio.", 0.95]]))
    print(why.answer_one("Why does John eat apple?", [["John eats apple because apple makes him healthy.", 0.95]]))
    print(where.answer_one("Where is Pittsburgh?", [["Pittsburgh is in Pennsylvania, USA.", 0.95]]))
    print(when.answer_one("when will JJ envolve?", [["JJ evolves when it reaches certain stage of happiness.", 0.95]]))
    pass

