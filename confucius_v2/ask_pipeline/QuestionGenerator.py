# -*- coding: utf-8 -*-
# import spacy
# import pyinflect

import logging
#logging.DEBUG
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level= logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")

from nltk.parse import CoreNLPParser
from nltk.corpus import wordnet
from ask_pipeline.preprocessing import batch_process

# os.chdir("/Users/hequanhong/Desktop/CS411_Group_Project-master/stanford-corenlp-full-2018-10-05")
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 1500
parser = CoreNLPParser(url='http://localhost:9001')


# -------------------------------------------------------
#                 Documents & Dev Tools
# -------------------------------------------------------
# TODO: 把你用到的所有参考资料堆在这里，格式是简介和链接，举例：

# spacy dependency tree visualizer
# https://explosion.ai/demos/displacy

# dependency tree component lookup
# https://blog.csdn.net/lihaitao000/article/details/51812618


# -------------------------------------------------------
#               Question generators go here
# -------------------------------------------------------
#
# # 基类
# class QuestionGenerator:
#     def __init__(self):
#         #  写你的初始化函数，比如载入模型啥的
#         #  不用载入spacy，直接使用全局的nlp()即可
#         #  如果你载入了什么其他的东西，请存进self里
#         #  你可以自定义这里init的参数，比如模型名称，你需要的其他数据等
#         pass
#     def generate_one(self,sentence:str)->list:
#         #  这个函数接受一个句子，！！根据你的类的类型！！返回可以出的问题以及问题可靠度（0到1之间的float）的列表
#         #  举例，假设这个类是generate who question的：
#         #       sentence="John's father has two sons, the first son's name is Josh."
#         #       return = [ ["Who is the first son?",0.533],["Who is the second son?", 0.1] ]
#         #  请尽量满足return的条件。如果你没有算概率，请保证返回有序的0-1数值；如果你只有一个答案，请也用list包好答案
#         #  如果你的文章句子不适合出这个类型的问题，请返回空list
#         pass
#
#     def batch_generate(self,sentences:list)->list:
#         #  这个函数和上面answer_one类似，区别在于这个函数是批量回答问题
#         #  函数的参数就是list形式的answer_one所传参数；return的东西也是用list包起来的answer_one输出
#         #  举例，假设这个类是generate who question的：
#         #  sentences = ["John's father has two sons.","The first son's name is Josh."]
#         #  return = [ ["Who is the first son?",0.533],["Who is the second son?", 0.1] ]
#         #  如果你的文章的所有句子不适合出这个类型的问题，请返回空list
#
#         #  下面是默认实现，原理是批量调用generate one
#         return batch_process(self.generate_one,sentences)
#
# # who what where when why how yesno
# class WhoGenerator(QuestionGenerator):
#     def __init__(self):
#         pass
#
#     def answer_one(self,sentence:str)->list:
#         pass
#
#
# class WhatGenerator(QuestionGenerator):
#     def __init__(self):
#         pass
#
#     def answer_one(self, sentence: str) -> list:
#         pass
#
# class WhereGenerator(QuestionGenerator):
#     def __init__(self):
#         pass
#
#     def answer_one(self, sentence: str) -> list:
#         pass
#
# class WhenGenerator(QuestionGenerator):
#     def __init__(self):
#         pass
#
#     def answer_one(self, sentence: str) -> list:
#         pass
#
#

# class WhyGenerator(QuestionGenerator):
#     def __init__(self):
#         pass
#
#     def answer_one(self, sentence: str) -> list:
#         pass
#
#
# class HowGenerator(QuestionGenerator):
#     def __init__(self):
#         pass
#
#     def answer_one(self, sentence: str) -> list:
#         pass
#
#
# class YesNoGenerator(QuestionGenerator):
#     def __init__(self):
#         pass
#
#     def answer_one(self, sentence: str) -> list:
#         pass
#

class QuestionGenerator:
    def __init__(self, sent: str):
        #print(sent)
        self.sent = sent
        self.dead = False
        sub_aux_in = sub_aux_inversion(sent)

        logging.debug("Sentence: "+sent)

        is_suitable = self.is_suitable_qgsent(sent)
        # self.parse is assigned in is_quitable_qgsent

        if (not is_suitable) or sub_aux_in == '' or len(sent)<5:
            logging.debug("Not a suitable sentence, will return empty list.")
            self.dead = True
            return

        self.sub_aux_inverse = sub_aux_in
        # print(self.sub_aux_inverse)

        self.orig_spacy = nlp(self.sent)
        self.sent_spacy = nlp(self.sub_aux_inverse) #sub_aux reversed sent

        for i in range(len(self.sent_spacy)):
            if self.sent_spacy[i].dep_ == "ROOT":
                self.root = self.sent_spacy[i].text
                self.root_index = i
                break

        # must call when and where question first to make the what question perform well
        PP_finder(self.parse)
        answers = []  # potential when and where answers' list (start, end)
        answers_phrase = []
        for answer_tree in PP_tree:
            answer = answer_tree.leaves()
            answers_phrase.append(answer_tree.leaves())
            for i in range(len(self.sent_spacy)):
                if str(self.sent_spacy[i].text) == answer[0]:
                    start = i
                    for j in range(i, i + len(answer)):
                        if str(self.sent_spacy[j].text) == answer[j - i]:
                            if str(self.sent_spacy[j].text) == answer[-1]:
                                end = j + 1
                                answers.append((start, end))
                                break
                        else:
                            break

        self.when_index = when_phrase(self.sent_spacy, answers)
        self.where_index = where_phrase(self.sent_spacy, answers)

        self.answers = answers
        self.answers_phrase = answers_phrase
        self.whose_index = []
        self.who_index = []
        pass

    # only want S that is NP + VP / NP + ADVP + VP
    def is_suitable_qgsent(self, sent):
        if len(sent.split(" ")) > 50:
            return False
        else:
            retry = 3
            while retry >= 0:
                if retry == 0:
                    logging.error("CoreNLP unrecoverable ERROR!!")
                    # TODO: 如果corenlp一直用不了，那该怎么处理？在这里实现解决方案
                    return False
                else:
                    try:
                        test_parse = next(parser.raw_parse(sent))  # get parse tree
                        break
                    except:
                        logging.exception("CoreNLP ERROR!! Retrying.")
                        retry -= 1
            for tree in test_parse:
                count = 0
                if type(tree) == nltk.tree.Tree:
                    advp = False
                    for subtree in tree:
                        if type(subtree) == nltk.tree.Tree:
                            # print(subtree)
                            count += 1
                            if subtree.label().strip() == "NP" and count == 1:
                                continue
                            if subtree.label().strip() == "ADVP" and count == 2:
                                advp = True
                                continue
                            if (subtree.label().strip() == "VP" and count == 2) or (
                                    subtree.label().strip() == "VP" and count == 3 and advp == True):
                                self.parse = test_parse
                                return True

                            else:
                                return False
                else:
                    break
            return False

    def generate_all(self):
        r = []
        if not self.dead: #do not change the order, things will be effected
            r.extend(self.generate_whose())
            r.extend(self.generate_who())
            r.extend(self.generate_when())
            r.extend(self.generate_where())
            r.extend(self.generate_what())
            r.extend(self.generate_which_compare())
            r.extend(self.generate_why())
            r.extend(self.generate_how())
            r.extend(self.generate_bi())
        logging.debug("Questions: " + str(r))
        return r
    # generate binary yes/no question
    def generate_bi(self):
        inversed_sent = self.sub_aux_inverse
        yes_q = []
        yes_q.append(inversed_sent)
        no_q = []
        adj_syn = []
        adj_anto = []
        for i in range(len(self.sent_spacy)):
            # use adjs
            if self.sent_spacy[i].pos_ == 'ADJ':
                adj = self.sent_spacy[i].text
                adj_index = i
                adj_syn = []
                adj_anto = []
                for syn in wordnet.synsets(adj):    # BUG will occur here if debug mode is on
                    if len(adj_syn) == 1 and len(adj_anto) == 1:
                        break
                    else:
                        for l in syn.lemmas():
                            # print(syn.lemmas())
                            if l.name() != adj and (l.name() not in adj_syn) and len(
                                    adj_syn) < 1:  # only want one to save time
                                adj_syn.append(l.name())
                            if l.antonyms():
                                if l.antonyms()[0].name() not in adj_anto and len(
                                        adj_anto) < 1:  # only want one to save time
                                    adj_anto.append(l.antonyms()[0].name())
                break  # only use one adj to save time

        # print(adj_syn, adj_anto)
        for word in adj_syn:
            question = ""
            for i in range(len(self.sent_spacy)):
                if i != adj_index:
                    question += self.sent_spacy[i].text + " "
                else:
                    question += word.replace("_", "-") + " "
            yes_q.append(question.rstrip())

        for word in adj_syn:
            question = ""
            for i in range(len(self.sent_spacy)):
                if i != adj_index:
                    question += self.sent_spacy[i].text + " "
                else:
                    question += "not" + " " + word.replace("_", "-") + " "
            no_q.append(question.rstrip())

        for word in adj_anto:
            question = ""
            for i in range(len(self.sent_spacy)):
                if i != adj_index:
                    question += self.sent_spacy[i].text + " "
                else:
                    question += word.replace("_", "-") + " "
            no_q.append(question.rstrip())

        for word in adj_anto:
            question = ""
            for i in range(len(self.sent_spacy)):
                if i != adj_index:
                    question += self.sent_spacy[i].text + " "
                else:
                    question += "not" + " " + word.replace("_", "-") + " "
            yes_q.append(question.rstrip())

        r = post_processing(yes_q)
        r.extend(post_processing(no_q))
        return r

    def generate_whose(self):
        if "who" not in self.sent:
            result = []
            orig_spacy = self.orig_spacy
            whose_index = whose_phrase(orig_spacy)
            self.whose_index = whose_index
            for pair in whose_index:
                whose_sent = ""
                i = 0
                while i < len(orig_spacy):
                    if i == pair[0]:
                        whose_sent += "whose" + " "
                        i += pair[1] - pair[0]
                    else:
                        whose_sent += orig_spacy[i].text + " "
                        i += 1
                result.append(whose_sent.rstrip())
            return post_processing(result)
        return []

    def generate_who(self):
        if "who" not in self.sent:
            result_f = []
            # check if there is any plausible who
            who_index = who_phrase(self.sent_spacy)
            result = []
            for pair in who_index:
                if (pair[0], pair[1] + 1) not in self.whose_index:
                    who_sent = ""
                    i = 0
                    while i < len(self.sent_spacy):
                        if i == pair[0]:
                            who_sent += "who" + " "
                            i += pair[1] - pair[0]
                        elif self.sent_spacy[i].text == "who":
                            break
                        else:
                            who_sent += self.sent_spacy[i].text + " "
                            i += 1
                    if "what's" not in who_sent.rstrip():
                        result.append(who_sent.rstrip())
                result_f += wh_movement(result, self.root_index)
                return post_processing(result_f)
        return []


    def generate_why(self):
        # why questions
        result = []
        SBAR_finder_reason(self.parse)
        why_index = []
        for answer_tree in SBAR_tree_reason:
            answer = answer_tree.leaves()
            if answer[0] == "because" or answer[0] == "since":
                for i in range(len(self.sent_spacy)):
                    if str(self.sent_spacy[i].text) == answer[0]:
                        start = i
                        for j in range(i, i + len(answer)):
                            if str(self.sent_spacy[j].text) == answer[j - i]:
                                if str(self.sent_spacy[j].text) == answer[-1]:
                                    end = j + 1
                                    why_index.append((start, end))
                                    break
                            else:
                                break
            else:
                break

        for pair in why_index:
            why_sent = ""
            i = 0
            while i < len(self.sent_spacy):
                if i == pair[0]:
                    why_sent += "why" + " "
                    i += pair[1] - pair[0]
                else:
                    why_sent += self.sent_spacy[i].text + " "
                    i += 1
            result.append(why_sent.rstrip())
        return post_processing(wh_movement(result, self.root_index))

    def generate_when(self):
        result = []
        when_index = self.when_index

        # additional when questions
        SBAR_finder_wh(self.parse)
        clause_answers = []

        for answer_tree in SBAR_tree:
            answer = answer_tree.leaves()
            if answer[0] == "when":
                for i in range(len(self.sent_spacy)):
                    if str(self.sent_spacy[i].text) == answer[0]:
                        start = i
                        for j in range(i, i + len(answer)):
                            if str(self.sent_spacy[j].text) == answer[j - i]:
                                if str(self.sent_spacy[j].text) == answer[-1]:
                                    end = j + 1
                                    clause_answers.append((start, end))
                                    break
                            else:
                                break
            else:
                break
        when_index += clause_answers
        for pair in when_index:
            when_sent = ""
            i = 0
            while i < len(self.sent_spacy):
                if i == pair[0]:
                    when_sent += "when" + " "
                    i += pair[1] - pair[0]
                else:
                    when_sent += self.sent_spacy[i].text + " "
                    i += 1
            result.append(when_sent.rstrip())

        result_f = post_processing(wh_movement(result, self.root_index))
        output = post_processing(wh_movement(result, self.root_index))
        for q in result_f:
            re = "What time" + q.replace("When", '')
            output.append(re)
        return output

    def generate_where(self):
        result = []
        where_index = self.where_index
        where_index_update = []
        for pair in where_index:
            if (pair not in self.who_index) and ((pair[0], pair[1] + 1) not in self.whose_index):
                where_sent = ""
                i = 0
                while i < len(self.sent_spacy):
                    if i == pair[0]:
                        where_sent += "where" + " "
                        i += pair[1] - pair[0]
                    else:
                        where_sent += self.sent_spacy[i].text + " "
                        i += 1
                result.append(where_sent.rstrip())
        result_f = post_processing(wh_movement(result, self.root_index))
        output = post_processing(wh_movement(result, self.root_index))
        for q in result_f:
            re = "What location" + q.replace("Where", '')
            output.append(re)
        return output

    def generate_what(self):
        result = []
        what_index = []
        root_index = self.root_index
        root = self.root
        what_index += what_org_phrase(self.sent_spacy)

        if root_index == 0 and (root == "is" or root == "are" or root == "was" or root == "were"):
            if "who" not in self.sent:
                what_tmp_index = what_phrase(self.sent_spacy, self.parse)

                for pair in what_tmp_index:
                    if (pair not in self.when_index) and (pair not in self.where_index) and (check_inclusive_tuple(what_index, pair)):
                        what_index.append(pair)

        for pair in self.answers:
            if (pair not in self.when_index):
                if (pair[0], pair[1]) not in self.where_index:
                    if self.sent_spacy[pair[0]].text == "of":
                        if check_inclusive_tuple(what_index, pair):
                            what_index.append((pair[0], pair[1]))
                if (pair[0] + 1, pair[1]) not in self.where_index and self.sent_spacy[pair[0]].text != "of":
                    what_index.append((pair[0] + 1, pair[1]))
        for pair in what_index:
            if ((pair[0], pair[1] + 1) not in self.whose_index) and (pair not in self.who_index) and (pair not in self.when_index) and (pair not in self.where_index):
                what_sent = ""
                i = 0
                is_valid = True
                while i < len(self.sent_spacy):
                    if i == pair[0]:
                        for j in range(pair[0], pair[1]):
                            if (self.sent_spacy[j].text == "this") or (self.sent_spacy[j].text == "that"):
                                is_valid = False
                                break
                        what_sent += "what" + " "
                        i += pair[1] - pair[0]
                    # check the 's
                    elif i == pair[1] and pair[1] < len(self.sent_spacy):
                        if self.sent_spacy[pair[1]].text == "'s":
                            is_valid = False
                            break
                        else:
                            what_sent += self.sent_spacy[i].text + " "
                            i += 1
                    else:
                        if i == len(self.sent_spacy):
                            break
                        else:
                            what_sent += self.sent_spacy[i].text + " "
                            i += 1
                if is_valid:
                    result.append(what_sent.rstrip())
        return post_processing(wh_movement(result, self.root_index))

    def generate_which_compare(self):
        if "," in self.sent:
            return []
        than_index = None
        sent_orig_spacy = self.orig_spacy
        sent_orig_parse = self.parse

        answers_index = []  # potential when and where answers' list (start, end)
        for answer in self.answers_phrase:
            for i in range(len(sent_orig_spacy)):
                if str(sent_orig_spacy[i].text) == answer[0]:
                    start = i
                    for j in range(i, i + len(answer)):
                        if str(sent_orig_spacy[j].text) == answer[j - i]:
                            if str(sent_orig_spacy[j].text) == answer[-1]:
                                end = j + 1
                                answers_index.append((start, end))
                                break
                        else:
                            break

        for answer_pair in answers_index:
            start, end = answer_pair
            for i in range(start, end):
                if sent_orig_spacy[i].text == "than":
                    than_index = i

        if than_index == None:
            return []

        else:
            than_sub_prior = None
            than_sub_post = None

            NP_leaf_finder(sent_orig_parse)

            for answer_tree in NP_leaf_tree:
                answer = answer_tree.leaves()
                for i in range(len(sent_orig_spacy)):
                    if str(sent_orig_spacy[i].text) == answer[0]:
                        start = i
                        end = i + len(answer)
                        if end <= than_index and not (sent_orig_spacy[start].pos_ == "ADV" or (
                                sent_orig_spacy[start].tag_ == "JJR" and sent_orig_spacy[start].pos_ == "ADJ")):
                            re = ""
                            for word in answer:
                                re += word + " "
                            than_sub_prior = re.rstrip()
                        elif start > than_index and than_sub_post == None:
                            re = ""
                            for word in answer:
                                re += word + " "
                            than_sub_post = re.rstrip()

            if than_sub_prior != None and than_sub_post != None:
                result = []
                re = "Which one "
                for i in range(than_index):
                    if sent_orig_spacy[i].text not in than_sub_prior and sent_orig_spacy[i].text not in than_sub_post:
                        re += sent_orig_spacy[i].text + " "
                re = re.rstrip() + "? "
                re += than_sub_prior + " or " + than_sub_post + "."
                result.append(re.rstrip())
                return result
            return []

    def generate_how(self):  # 没有写表程度的how question

        result = []
        how_many_index = []
        how_much_index = []
        how_index = []
        # genenrate how many and how much question
        for i in range(len(self.sent_spacy)):
            if self.sent_spacy[i].pos_ == "NUM" and self.sent_spacy[i].ent_type_ == "CARDINAL":
                how_many_index.append(i)
            if self.sent_spacy[i].pos_ == "PUNCT" and self.sent_spacy[i].ent_type_ == "CARDINAL":
                how_many_index.append(i)
            if self.sent_spacy[i].pos_ == "NUM" and self.sent_spacy[i].ent_type_ == "MONEY":
                how_much_index.append(i)
            if self.sent_spacy[i].pos_ == "PUNCT" and self.sent_spacy[i].ent_type_ == "MONEY":
                how_much_index.append(i)
            if self.sent_spacy[i].lemma_ == 'by':
                how_index.append(i)

        last_idx = 0
        invalid_how_many_index = []
        for index in how_many_index:
            if index == last_idx + 1:
                last_idx = index
                invalid_how_many_index.append(index)
                continue
            last_idx = index

        last_idx = 0
        for index in how_many_index:
            how_many_sent = ""

            if index == last_idx + 1:
                last_idx = index
                continue
            for i in range(len(self.sent_spacy)):
                if i in invalid_how_many_index:
                    continue

                if i == index:
                    how_many_sent += "how many" + " "

                else:
                    how_many_sent += self.sent_spacy[i].text + " "
            last_idx = index

            result.append(how_many_sent.rstrip())

        last_idx = 0
        invalid_how_much_index = []
        for index in how_much_index:
            if index == last_idx + 1:
                last_idx = index
                invalid_how_much_index.append(index)
                continue
            last_idx = index

        last_idx = 0
        for index in how_much_index:
            how_much_sent = ""

            if index == last_idx + 1:
                last_idx = index
                continue
            for i in range(len(self.sent_spacy)):
                if i in invalid_how_much_index:
                    continue

                if i == index:
                    how_much_sent += "how much" + " "

                else:
                    how_much_sent += self.sent_spacy[i].text + " "
            last_idx = index
            result.append(how_much_sent.rstrip())

        for index in how_index:
            how_sent = ""
            for i in range(len(self.sent_spacy)):
                if i == index:
                    how_sent += "how" + " "
                    break
                else:
                    how_sent += self.sent_spacy[i].text + " "
            result.append(how_sent.rstrip())


        return post_processing(wh_movement(result, self.root_index))


####################################### step 1: is suitable sentence

###################################step 2: subject-auxiliary inversion
# input: simplified sentences, Decompose the main verb, output: inversed sub-auxi

def sub_aux_inversion(sent):

    aux = None
    sub = None
    sub_index = None
    root_index = None
    root = None

    result = ""

    if sent[-1].strip() != ".":
        sent += "."

    sent_spacy = nlp(sent)

    i = -1
    for token in sent_spacy:
        i += 1
        if token.dep_ == "ROOT":
            root = token.text
            root_index = i
            root_token = token
        if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
            sub = token.text
            sub_index = i
    # if the sentence has a aux 或被动
    i = -1
    for token in sent_spacy:
        i += 1
        if (token.dep_ == "aux" and i < root_index) or (token.dep_ == "ROOT" and token.head.pos_ == "AUX") or (
                token.dep_ == "auxpass" and token.head.text == root):
            aux_index = i
            if token.text == "'s":
                aux = "is"
            elif token.text == "'re":
                aux = "are"
            elif token.text == "'ve":
                aux = "have"
            elif token.text == "'d":
                aux = "would"
            else:
                aux = token.text
            break

    if aux != None:
        result = aux + " "
        i = -1
        for token in sent_spacy:
            i += 1
            if i != aux_index:
                if token.lemma_ == "'s":
                    result = result.rstrip()
                    result += token.text.lower() + " "
                elif i == 0 and (token.pos_ == "ADV" or token.pos_ == "ADJ" or\
                                 (token.pos_ == "NOUN" and token.ent_type_ == "") or \
                                 (token.lemma_ == "PRON" and token.text != "I") or\
                                 token.pos_ == "DET"):
                    result += token.text.lower() + " "
                elif token.dep_ == "det" or token.dep_ == "ROOT" or token.lemma_ == "be":
                    result += token.text.lower() + " "
                else:
                    result += token.text + " "
        result = result.rstrip()

    # if the sentence does not have an aux, decompose the main verb
    else:
        if root_index != None:
            if root_token.tag_ == 'VBD' or root_token.tag_ == 'VBN':
                aux = "did"
            elif root_token.tag_ == 'VBZ':
                aux = "does"
            elif root_token.tag_ == 'VB' or root_token.tag_ == 'VBP':
                aux = "do"

        if aux != None:
            result = aux + " "
            i = -1
            for token in sent_spacy:
                i += 1
                if token.lemma_ == "'s":
                    result = result.rstrip()
                    result += token.text.lower() + " "
                elif i == 0 and (token.pos_ == "ADV" or token.pos_ == "ADJ" or \
                                 (token.pos_ == "NOUN" and token.ent_type_ == "") or \
                                 (token.lemma_ == "PRON" and token.text != "I") or \
                                 token.pos_ == "DET"):
                    result += token.text.lower() + " "
                elif token.pos_ == "VERB" and token.text == root:
                    result += token.lemma_ + " "
                elif token.dep_ == "det" or token.lemma_ == "be":
                    result += token.text.lower() + " "
                else:
                    result += token.text + " "
            result = result.rstrip()
    return result


##############################step 3: answer-to-question phrase matching
# input: valid sentence
import string  # do not delete this function
#step 3: a_q matching
##############################################step 4: WH movement
# step 4: WH movement
def wh_movement(sent_list, root_index):  # include how many/how much
    # print(root_index)
    question = []
    root_index = root_index

    for sent in sent_list:
        wh_index = []
        wh_index_noaux = []
        wh_index_be = []
        how_index = []
        how_regular_index = []

        sent_spacy = nlp(sent)

        wh_count = 0
        for i in range(len(sent_spacy)):
            if i > root_index and (sent_spacy[i].tag_ == "WRB" or sent_spacy[i].tag_ == "WP") and wh_count < 1 and \
                    sent_spacy[i].text != "how":
                wh_count += 1
                wh_index.append(i)

            elif i < root_index and sent_spacy[0].lemma_ == "be" and (
                    sent_spacy[i].tag_ == "WRB" or sent_spacy[i].tag_ == "WP") and wh_count < 1 and sent_spacy[
                i].text != "how":
                wh_count += 1
                wh_index_be.append(i)  # 被动语态

            elif i < root_index and (sent_spacy[i].tag_ == "WRB" or sent_spacy[i].tag_ == "WP") and wh_count < 1 and \
                    sent_spacy[i].text != "how":
                wh_count += 1
                wh_index_noaux.append(i)

        for i in range(len(sent_spacy)):
            if i + 2 > len(sent_spacy):
                continue
            if sent_spacy[i + 1].text != "many" and sent_spacy[i + 1].text != "much":
                continue
            if sent_spacy[i].text == "how":  # how many# using sent_pos_spacy
                noun_idx = i
                while (True):
                    noun_idx += 1
                    if noun_idx + 1 > len(sent_spacy):
                        break
                    if sent_spacy[noun_idx].pos_ == 'NOUN':
                        t = ()
                        for j in range(i, noun_idx + 1, 1):
                            t += (j,)
                        how_index.append(t)
                        break

        #                 if i + 2 < len(sent_spacy):
        #                     # print(i, sent_pos_spacy[i])
        #                     if sent_spacy[i + 2].pos_ == 'ADJ':
        #                         how_index.append((i, i + 1, i + 2, i + 3))
        #                     else:
        #                         how_index.append((i, i + 1, i + 2))

        for i in range(len(sent_spacy)):
            if sent_spacy[i].text == "how":  # how many# using sent_pos_spacy
                if i + 2 > len(sent_spacy):
                    how_regular_index.append(i)
                elif sent_spacy[i+1].text != 'many' and sent_spacy[i+1].text != 'much':
                    how_regular_index.append(i)

        for index in wh_index:
            wh_sent = sent_spacy[index].text + " "
            for i in range(len(sent_spacy)):
                if i != index:
                    if sent_spacy[i].lemma_ == "'s":
                        wh_sent = wh_sent.rstrip()
                        wh_sent += sent_spacy[i].text.lower() + " "
                    else:
                        wh_sent += sent_spacy[i].text + " "
            question.append(wh_sent.rstrip())

        for index in wh_index_be:
            wh_sent = sent_spacy[index].text + " "
            for i in range(len(sent_spacy)):
                if i != index:
                    if sent_spacy[i].lemma_ == "'s":
                        wh_sent = wh_sent.rstrip()
                        wh_sent += sent_spacy[i].text.lower() + " "
                    else:
                        wh_sent += sent_spacy[i].text + " "
            question.append(wh_sent.rstrip())

        for index in wh_index_noaux:
            wh_sent = sent_spacy[index].text + " "
            for i in range(len(sent_spacy)):
                if i != index:
                    if sent_spacy[i].lemma_ == "'s":
                        wh_sent = wh_sent.rstrip()
                        wh_sent += sent_spacy[i].text.lower() + " "
                    else:
                        wh_sent += sent_spacy[i].text + " "
            question.append(wh_sent.rstrip())

        for tuples in how_index:
            # print(tuples)

            how_sent = "how" + " "

            for index in tuples[1:]:
                how_sent += sent_spacy[index].text + " "

            for i in range(len(sent_spacy)):
                if i not in tuples:
                    if sent_spacy[i].lemma_ == "'s":
                        how_sent = how_sent.rstrip()
                        how_sent += sent_spacy[i].text.lower() + " "
                    else:
                        how_sent += sent_spacy[i].text + " "
            question.append(how_sent.rstrip())

        for idx in how_regular_index:
            how_sent_regular = "how" + " "
            for i in range(len(sent_spacy)):
                if sent_spacy[i].lemma_ == 'how':
                    pass
                else:
                    how_sent_regular += sent_spacy[i].text + " "
            question.append(how_sent_regular.rstrip())

    return question

# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------
# TODO: 如果你有什么需要共用的辅助函数，请写在这里

#############################################
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
                    elif count >= 2:
                        break

            else:
                SBAR_finder_helper_reason(subtree)
        else:
            break

# find PP in corenlp
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


####################################### phrase detection
def check_inclusive_tuple(result_list, pair):
    for item in result_list:
        if pair[0] >= item[0] and pair[1] <= item[1]:
            return False
    return True

def whose_phrase(sent_spacy):
    result = []
    for i in range(len(sent_spacy)):
        if sent_spacy[i].lemma_ == "'s":
            if i > 0:
                if not (sent_spacy[i - 1].text[0].isupper()):
                    break
            for j in range (0, i):
                if not (sent_spacy[i - 1 - j].ent_type_ == "PERSON" or sent_spacy[i - 1 - j].text[0].isupper()):
                    if check_inclusive_tuple(result, (i - j, i + 1)):
                        result.append((i - j, i + 1))
                        break
    return result

def who_phrase(sent_spacy): #return who_phrase index pairs
    result = []
    i = 0
    check_while = 0
    check_index = 0
    while i < len(sent_spacy):
        if check_index == i and check_while > 1:
            break
        elif check_index == i and check_while <= 1:
            check_while += 1
        elif check_index != i:
            check_while = 0
            check_index = i
        elif (sent_spacy[i].ent_type_ == "PERSON" and sent_spacy[i].pos_ == "PROPN"):
            start = i
            end = i + 1
            for j in range(i + 1, len(sent_spacy)):
                if not ((sent_spacy[j].ent_type_ == "PERSON" and sent_spacy[j].pos_ == "PROPN")):
                    end = j
                    if sent_spacy[end].text != "'s":
                        result.append((start, end))
                        i = end
                    break
        else:
            i += 1
    return result


def what_org_phrase(sent_spacy):
    result = []
    i = 0
    while i < len(sent_spacy):
        if sent_spacy[i].ent_type_ == "ORG" and sent_spacy[i].pos_ != "PROPN":
            start = i
            end = i + 1

            if i - 1 >= 0:
                if sent_spacy[i - 1].text == '"' or sent_spacy[i - 1].text == "'":
                    break
            if i + 1 < end:
                if sent_spacy[i + 1].text == "'s":
                    break

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


##answer_index is a list
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
                if (sent_spacy[i].ent_type_ != "TIME" and sent_spacy[i].ent_type_ != "DATE") or sent_spacy[
                    i].pos_ == 'AUX' or sent_spacy[i].pos_ == 'VERB':
                    if check_inclusive_tuple(result, (i, j)):
                        result.append((i, j))
                        break

    return result


# The answer phrase is a prepositional phrase whose object is tagged noun.location and whose preposition is one of the following: on, in, at, over, to
# answer_index is a list
# return the index of where phrase
# The answer phrase is a prepositional phrase whose object is tagged noun.location and whose preposition is one of the following: on, in, at, over, to
# answer_index is a list
# return the index of where phrase
def where_phrase(sent_spacy, answers_index):
    result = []
    for answer_pair in answers_index:
        start, end = answer_pair
        for i in range(start, end):
            if (sent_spacy[i].ent_type_ == "GPE" or sent_spacy[i].ent_type_ == "LOC"):
                if i - 1 >= 0:
                    if sent_spacy[i - 1].text == '"' or sent_spacy[i - 1].text == "'":
                        break
                if i + 1 < end:
                    if sent_spacy[i + 1].text == "'s":
                        break
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
            if i - 1 >= 0:
                if sent_spacy[i - 1].text == '"' or sent_spacy[i - 1].text == "'":
                    break
            if i + 1 < len(sent_spacy):
                if sent_spacy[i + 1].text == "'s":
                    break
            for j in range(i + 1, len(sent_spacy)):
                if sent_spacy[j].ent_type_ != "GPE" and sent_spacy[j].ent_type_ != "LOC":
                    if check_inclusive_tuple(result, (i, j)):
                        result.append((i, j))
                    break

    return result


# sent is inversed
# return a list of index
def what_phrase(sent_spacy, sent_orig_parse):
    result_answer = []
    result = []
    parse_tree = sent_orig_parse

    # expecting parse tree to be NP VP(VP = VB + NP)
    for subtree in parse_tree:  # 只有一个loop
        if subtree.label().strip() == "S":
            count = 0
            for sub_tree in subtree:
                if sub_tree.label().strip() == "NP" and count == 0:
                    count += 1
                    result_answer.append(sub_tree.leaves())
                elif sub_tree.label().strip() == "VP" and count == 1:
                    sub_count = 0
                    for sub_vp in sub_tree:
                        if sub_count == 0 and (
                                sub_vp.label().strip() == "VBD" or sub_vp.label().strip() == "VBZ" or sub_vp.label().strip() == "VBP"):
                            sub_count += 1
                        elif sub_count == 1 and sub_vp.label().strip() == "NP":
                            result_answer.append(sub_vp.leaves())
                            break

        else:
            break

    # match the answer to the index of the tree
    for answer in result_answer:
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


def post_processing(question_list) -> list:
    output = []
    for question in question_list:

        # question = question.capitalize()
        q_content = question.split(" ")
        result = q_content[0].capitalize() + " "
        i = 1
        if len(q_content) > 1:
            quota_count = 0
            while i < len(q_content):
                if q_content[i] == "" and i != len(q_content) - 1:
                    i += 1
                    continue

                if q_content[i] == "," and i >= 1:
                    if q_content[i - 1] == ",":
                        i += 1
                        continue
                    else:
                        result = result.strip()
                        result += q_content[i] + " "
                        i += 1

                elif q_content[i] == "°":
                    result += q_content[i]
                    i += 1
                elif q_content[i] == "/":
                    result = result.rstrip()
                    result += q_content[i]
                    i += 1
                elif q_content[i] == "." and i < len(q_content) - 2:
                    result = result.rstrip()
                    result += "."
                    i += 1

                elif q_content[i] == "These":
                    result += q_content[i].lower() + " "
                    i += 1

                elif q_content[i] == "This":
                    result += q_content[i].lower() + " "
                    i += 1
                elif q_content[i] == "That":
                    result += q_content[i].lower() + " "
                    i += 1
                elif q_content[i] == "Those":
                    result += q_content[i].lower() + " "
                    i += 1
                elif q_content[i] == "It":
                    result += q_content[i].lower() + " "
                    i += 1
                elif q_content[i] == "One":
                    result += q_content[i].lower() + " "
                    i += 1
                elif q_content[i] == "Two":
                    result += q_content[i].lower() + " "
                    i += 1
                elif q_content[i] == "i":
                    result += q_content[i].upper() + " "
                    i += 1

                elif ('"' in q_content[i]) and (q_content[i] != '"'):
                    quota_count += q_content[i].count('"')
                    if quota_count % 2 == 1:
                        result += q_content[i] + " "
                        i += 1
                    else:
                        if q_content[i].count('"') % 2 == 0:
                            result += q_content[i] + " "
                            i += 1
                        else:
                            result = result.rstrip()
                            result += q_content[i] + " "
                            i += 1

                elif ("'" in q_content[i]) and (q_content[i] != "'") and (q_content[i] != '"') and ("'s" not in q_content[i]):
                    quota_count += q_content[i].count("'")
                    if quota_count % 2 == 1:
                        result += q_content[i]
                        i += 1
                    else:
                        if q_content[i].count("'") % 2 == 0:
                            result += q_content[i] + " "
                            i += 1
                        else:
                            result = result.rstrip()
                            result += q_content[i] + " "
                            i += 1

                elif q_content[i] == " " and i != len(q_content) - 1:
                    i += 1
                    continue

                elif q_content[i] in (',', ':', "!") and i == len(q_content) - 1:
                    if quota_count % 2 == 0:
                        result = result[:-1] + '?'
                    else:
                        result = result[:-1] + '"' + '?'
                    break

                elif q_content[i] in (',', ':', "!") and i == len(q_content) - 2 and q_content[i + 1] == ".":
                    result = result.rstrip()
                    result = result[:-1] + '?'
                    break

                elif q_content[i] == "." and i == len(q_content) - 2 and q_content[i + 1] in (',', ':', "!", '"'):
                    if q_content[i + 1] == '"':
                        quota_count += 1
                        if quota_count % 2 == 1:
                            result = result.rstrip()
                            result = result[:-1] + '"' + "?"
                            break
                        else:
                            result = result.rstrip()
                            result = result[:-1] + '?'
                            break

                    else:
                        if quota_count % 2 == 1: #make up the '"'
                            result = result.rstrip()
                            result = result[:-1] + '"' + '?'
                            break
                        else:
                            result = result.rstrip()
                            result = result[:-1] + '?'
                            break
                    break


                elif q_content[i] == "." and i == len(q_content) - 1:
                    if quota_count % 2 == 1:
                        result = result.rstrip()
                        result += '"' + "?"
                    else:
                        result = result.rstrip()
                        result += "?"
                    break

                elif q_content[i] == "," or q_content[i] == "'s" or q_content[i] == ":" or q_content[i] == "%" \
                        or q_content[i] == ";":
                    result = result.rstrip()
                    result += q_content[i] + " "
                    i += 1
                elif q_content[i] == "-":
                    result = result.strip()
                    result += q_content[i]
                    i += 1
                elif q_content[i] == '"' or q_content[i] == "'":
                    quota_count += 1
                    if quota_count % 2 == 1:
                        result += q_content[i]
                        i += 1
                    else:
                        result = result.rstrip()
                        result += q_content[i] + " "
                        i += 1

                elif i == len(q_content) - 1 and quota_count % 2 == 1:
                    result += q_content[i] + '"' + "?"
                    break

                elif i == len(q_content) - 1 and ("," in q_content[i] or "." in q_content[i]):
                    if quota_count % 2 == 0:
                        result += q_content[i][:len(q_content[i]) - 1] + "?"
                    else:
                        result += q_content[i][:len(q_content[i]) - 1] + '"' + "?"
                    break

                elif i == len(q_content) - 1 and "?" not in q_content[i]:
                    if quota_count % 2 == 0:
                        result += q_content[i] + "?"
                    else:
                        result += q_content[i] + '"' + "?"
                    break

                else:
                    result += q_content[i] + " "
                    i += 1
        result = result.replace('""', '')
        output.append(result.rstrip())
    return output



# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------

# TODO：如果你有与测试你的代码相关的函数，请写在这里

if __name__ == '__main__':
    #generator = QuestionGenerator("Ashoka helped convene the Third Buddhist Council of India's and South Asia's Buddhist orders near his capital, a council.")
    #print(generator.generate_all())
    #print(sub_aux_inversion("Ashoka helped convene the Third Buddhist Council of India's and South Asia's Buddhist orders near his capital, a council."))
    #print(post_processing(["Where was an important person during the reign of Djoser an important person during the reign of Djoser's vizier,?"]))

    print(post_processing(['Where were the Vedic people pursued by the Iranians "across to the Levant, across Iran into India']))
    # pass

