# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import spacy

nlp = spacy.load("en_core_web_sm")

# start server
from nltk.parse import CoreNLPParser
import numpy as np

# os.chdir("/Users/hequanhong/Desktop/CS411_Group_Project-master/stanford-corenlp-full-2018-10-05")

# 把下列代码输入含有"stanford-corenlp-full-2018-10-05"包的terminal的directory里
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 1500

parser = CoreNLPParser(url='http://localhost:9001')
import logging

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


#  记得更改类名
#  我放弃治疗了，这玩意太难拆了
#  就让他成为一次性的吧
class QuestionAnswerer:
    def __init__(self):
        # TODO: 写你的初始化函数，比如载入模型啥的
        #  如果你载入了什么东西，请存进self里
        #  你可以自定义这里init的参数，比如模型名称，你需要的其他数据等
        #  article是preprocess过的整篇文章，是list of str，可用可不用
        self.PP_tree = []
        self.SBAR_tree = []
        # for generating and answering why questions
        self.SBAR_tree_reason = []

        pass
    def answer_one(self,question:str,qtype:str,sentence:list):
        if qtype=='whatwhich':
            pass
        elif qtype=='where':
            pass
        elif qtype=='when':
            pass
        elif qtype=='who':
            pass
        elif qtype=='multiple':
            pass
        elif qtype=='yesno':
            return self.answer_yesno(question=question,sentence=sentence)
        elif qtype=='hownumber':
            pass
        elif qtype=='how':
            pass
        elif qtype=='why':
            pass
        else:
            logging.info("WTF")

    def PP_finder(self, parse_tree):
        # test_parse = next(parser.raw_parse(sent))
        self.PP_finder_helper(parse_tree)

    def PP_finder_helper(self,parse_tree):
        for subtree in parse_tree:
            if type(subtree) == nltk.tree.Tree:
                if subtree.label().strip() == 'PP':
                    count_sub = 0
                    for sub_subtree in subtree:
                        count_sub += 1
                        # print(count_sub, sub_subtree)
                        if sub_subtree.label().strip() == "NP" and count_sub == 2:

                            self.PP_tree.append(subtree)

                        elif sub_subtree.label().strip() == "PP" and count_sub == 1:
                            self.PP_finder_helper(subtree)

                        elif sub_subtree.label().strip() == "S" and count_sub == 2:
                            self.PP_finder_helper(sub_subtree)
                else:
                    self.PP_finder_helper(subtree)
            else:
                break

    def answer_yesno(self, question: str, sentence: list) -> list:
        # TODO：这个函数接受一个问题以及该问题相关句子概率列表，返回按概率排序的答案列表
        #  举例：question="What is the meaning of life?"
        #       sentence=[ ["Life is a word.",0.97], ["Life means responsibility", 0.96] ]
        #       return = [ ["Responsibility.",0.533],["A word.", 0.51] ]
        #  请尽量满足return的条件。如果你没有算概率，请保证返回有序的0-1数值；如果你只有一个答案，请也用list包好答案
        Question = question
        result = []
        for Sentence, prob in sentence:
            s_dep_tree = get_parse_tree(Sentence)
            s_dep_tree_word = [s_dep_tree[i][0] for i in range(len(s_dep_tree))]
            s_dep_tree_tag = [s_dep_tree[i][1] for i in range(len(s_dep_tree))]

            s_neg_num = np.sum([np.array(s_dep_tree_tag) == 'neg'])

            q_dep_tree = get_parse_tree(Question)
            q_dep_tree_word = [q_dep_tree[i][0] for i in range(len(q_dep_tree))]
            q_dep_tree_tag = [q_dep_tree[i][1] for i in range(len(q_dep_tree))]

            q_neg_num = np.sum([np.array(q_dep_tree_tag) == 'neg'])

            answer = False
            if s_neg_num % 2 == q_neg_num % 2:
                answer = True
            print(answer)

            s_pos = pos_tagger(Sentence)
            q_pos = pos_tagger(Question)

            s_target_lemma = get_adj_adv_verb(s_pos)
            q_target_lemma = get_adj_adv_verb(q_pos)

            q_different = [lemma for lemma in q_target_lemma if lemma not in s_target_lemma]
            s_different = [lemma for lemma in s_target_lemma if lemma not in q_target_lemma]

            dif_num = check_similarity(q_different, s_different)

            for i in range(dif_num):
                answer = not answer
            print(answer)

            s_NN_DATE = get_NN_DATE(s_pos)
            q_NN_DATE = get_NN_DATE(q_pos)

            q_different_NN_DATE = [lemma for lemma in q_NN_DATE if lemma not in s_NN_DATE]
            if len(q_different_NN_DATE) > 0:
                answer = False
            print(answer)
            if answer == True:
                result_0 = 'Yes.'
            else:
                result_0 = 'No.'
            result.append([result_0, prob])
        return result

    def answer_when(self, question, sent_list):
        for sent_pair in sent_list:
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_parse_tree = next(parser.raw_parse(sent))
            self.PP_finder(sent_parse_tree)
            sent_spacy = nlp(sent)

            answers = []  # potential when and where answers' list (start, end)
            for answer_tree in self.PP_tree:
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

            # additional when questions
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
            # print(clause_answers)

            for pair in when_index:
                re = ""
                start, end = pair[0], pair[1]
                for j in range(start, end):
                    re += sent_spacy[j].text + " "
                re = re.rstrip()
                if re not in question:
                    result.append(re)

            if len(result) > 0:
                result.append(sent_prob)
                return result

    def answer_where(self, question, sent_list):
        for sent_pair in sent_list:
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_parse_tree = next(parser.raw_parse(sent))
            self.PP_finder(sent_parse_tree)
            sent_spacy = nlp(sent)

            answers = []  # potential when and where answers' list (start, end)
            for answer_tree in self.PP_tree:
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
            where_index = where_phrase(sent_spacy, answers)
            for pair in where_index:
                re = ""
                start, end = pair[0], pair[1]
                for j in range(start, end):
                    re += sent_spacy[j].text + " "
                re = re.rstrip()
                if re not in question:
                    result.append(re)

            if len(result) > 0:
                result.append(sent_prob)
                return result

    def answer_who(self, question, sent_list):
        for sent_pair in sent_list:
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_spacy = nlp(sent)
            who_indexes = who_phrase(sent_spacy)
            result = []
            i = 0
            for pair in who_indexes:
                re = ""
                start, end = pair[0], pair[1]
                for j in range(start, end):
                    re += sent_spacy[j].text + " "
                re = re.rstrip()
                if re not in question:
                    result.append(re)
            if len(result) > 0:
                result.append(sent_prob)
                return result

    def answer_why(self, question, sent_list):
        for sent_pair in sent_list:
            sent = sent_pair[0]
            sent_prob = sent_pair[1]
            sent_parse_tree = next(parser.raw_parse(sent))
            SBAR_finder_reason(sent_parse_tree)
            sent_spacy = nlp(sent)

            answers = []  # potential why answers' list (start, end)
            for answer_tree in SBAR_tree_reason:
                answer = answer_tree.leaves()
                if answer[0] == "because" or answer[0] == "since":
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
            for pair in answers:
                re = ""
                start, end = pair[0], pair[1]
                for j in range(start, end):
                    re += sent_spacy[j].text + " "
                re = re.rstrip()
                if re not in question:
                    result.append(re)

            if len(result) > 0:
                result.append(sent_prob)
                return result


# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------

# TODO: 如果你有什么需要共用的辅助函数，请写在这里
# find pp in the sentence, use to generate where and when questions and answers









# for when question in dependent clause(addition to when)


def SBAR_finder_wh(parse_tree):
    global SBAR_tree
    SBAR_tree = []
    # test_parse = next(parser.raw_parse(sent))
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
            else:
                SBAR_finder_helper_wh(subtree)
        else:
            break





def SBAR_finder_reason(parse_tree):
    global SBAR_tree_reason
    SBAR_tree_reason = []
    # test_parse = next(parser.raw_parse(sent))
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


def check_inclusive_tuple(result_list, pair):
    for item in result_list:
        if pair[0] >= item[0] and pair[1] <= item[1]:
            return False
    return True


# phrase detection
def who_phrase(sent_spacy):
    result = []
    i = 0
    while i < len(sent_spacy):
        if sent_spacy[i].ent_type_ == "PERSON" and sent_spacy[i].pos_ == "PROPN":
            start = i
            end = i + 1
            for j in range(i + 1, len(sent_spacy)):
                if not (sent_spacy[j].ent_type_ == "PERSON" and sent_spacy[j].pos_ == "PROPN"):
                    end = j
                    result.append((start, end))
                    i = end
                    break
        else:
            i += 1
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
            if (sent_spacy[i].ent_type_ == "GPE" or sent_spacy[i].ent_type_ == "LOC") and (
                    sent_spacy[i - 1].lemma_ == "on" or sent_spacy[i - 1].lemma_ == "in" or sent_spacy[
                i - 1].lemma_ == "at" or sent_spacy[i - 1].lemma_ == "over" or sent_spacy[i - 1].lemma_ == "to" or
                    sent_spacy[i - 1].lemma_ == "under"):
                if (start + 1, end) not in result:
                    result.append((start + 1, end))  # need not to preserve介词
                    break

    # check if there is the where-phrase without介词
    for i in range(len(sent_spacy)):
        if sent_spacy[i].ent_type_ == "GPE" or sent_spacy[i].ent_type_ == "LOC":
            for j in range(i + 1, len(sent_spacy)):
                if sent_spacy[j].ent_type_ != "GPE" or sent_spacy[j].ent_type_ != "LOC":
                    if check_inclusive_tuple(result, (i, j)):
                        result.append((i, j))
                    break

    return result

# Yuxin's functions
def get_parse_tree(sentence):
    doc = nlp(sentence)
    result = []
    for token in doc:
        result.append((token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children]))
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

class QuestionTemplate:
    def __init__(self, article: list = []):
        # 写你的初始化函数，比如载入模型啥的
        #  如果你载入了什么东西，请存进self里
        #  你可以自定义这里init的参数，比如模型名称，你需要的其他数据等
        #  article是preprocess过的整篇文章，是list of str，可用可不用
        pass

    def answer_one(self,question:str,sentence:list)->list:
        # 这个函数接受一个问题以及该问题相关句子概率列表，返回按概率排序的答案列表
        #  举例：question="What is the meaning of life?"
        #       sentence=[ ["Life is a word.",0.97], ["Life means responsibility", 0.96] ]
        #       return = [ ["Responsibility.",0.533],["A word.", 0.51] ]
        #  请尽量满足return的条件。如果你没有算概率，请保证返回有序的0-1数值；如果你只有一个答案，请也用list包好答案
        pass

    def answer_batch(self,questions:list,sentences:list)->list:
        # 这个函数和上面answer_one类似，区别在于这个函数是批量回答问题
        #  函数的参数就是list形式的answer_one所传参数；return的东西也是用list包起来的answer_one输出
        #  设立这个函数的其中原因在于，有些人的回答方式能借助库来加速批量处理
        pass

    # 如果你有什么不需要共用的辅助函数，请写在class里，不然则可以写在下面的utilities里


# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------

# TODO：如果你有与测试你的代码相关的函数，请写在这里

if __name__ == '__main__':
    # TODO：如果你有直接运行的测试语句，请写在这里
    test = QuestionAnswerer()
    # print(test.answer_when("When will Igglybuff evolve?",
    #                        [["Igglybuff evolves when it reaches a certain point of happiness.", 0.95]]))
    print(test.answer_when("When will John get up?", [["John usually gets up at 12:00 o'clock.", 0.95]]))
    print(test.answer_who("who marries Maggie?", [["John Smith marries Maggie.", 0.95]]))
    print(test.answer_where("Where did you arrive at?", [["I arrived at Pittsburgh.", 0.95]]))
    print(test.answer_why("why do you like eating apple?",
                          [["I like eating apple because apple makes me healthy.", 0.95]]))
    pass

