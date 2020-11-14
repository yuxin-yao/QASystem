#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import spacy

nlp = spacy.load('en_core_web_sm')  # load spaCy's built-in English models


def sort_question(questions: list, length=10):
    import math
    sorted_question = sorted(zip(questions, get_score(questions, length)), key=lambda x: x[1], reverse=True)
    # print(sorted_question)
    sorted_question = [row[0] for row in sorted_question]
    # print(sorted_question)
    question_type = classify(sorted_question)
    return_que = []
    where_count = 0
    what_count = 0
    which_count = 0
    who_count = 0
    when_count = 0
    how_count = 0
    other_count = 0
    hownumber_count = 0

    where_up = question_type.count('where')
    what_up = question_type.count('what')
    which_up = question_type.count('which')
    who_up = question_type.count('who')
    when_up = question_type.count('when')
    how_up = question_type.count('how')
    other_up = question_type.count('other')
    hownumber_up = question_type.count('hownumber_up')
    all_up = where_up+what_up+which_up+who_up+when_up+how_up+other_up+hownumber_up

    # where_limit = min(where_up, int(1 * length / 10))
    # what_limit = min(what_up, int(1 * length / 10))
    # which_limit = min(which_up, int(1 * length / 10))
    # who_limit = min(who_up, int(1 * length / 10))
    # when_limit = min(when_up, int(1 * length / 10))
    # how_limit = min(how_up, int(1 * length / 10))
    # other_limit = min(other_up, int(1 * length / 10))
    # hownumber_limit = min(hownumber_up, int(1 * length / 10))
    where_limit = min(where_up, math.ceil(length * where_up/all_up))
    what_limit = min(what_up, math.ceil(length * what_up/all_up))
    which_limit = min(which_up, math.ceil(length * which_up/all_up))
    who_limit = min(who_up, math.ceil(length * who_up/all_up))
    when_limit = min(when_up, math.ceil(length * when_up/all_up))
    how_limit = min(how_up, math.ceil(length * how_up/all_up))
    other_limit = min(other_up, math.ceil(length * other_up/all_up))
    hownumber_limit = min(hownumber_up, math.ceil(length * hownumber_up/all_up))

    # remain_num = length - (where_limit + what_limit + which_limit + who_limit + when_limit + how_limit + other_limit + hownumber_limit)
    # if (where_up - where_limit > remain_num):
    #     where_limit += math.ceil(remain_num / 2)
    # if (when_up - when_limit > remain_num):
    #     when_limit += int(remain_num / 2)
    other_limit += length - (where_limit + what_limit + which_limit + who_limit + when_limit + how_limit + other_limit + hownumber_limit)

    for i, que in enumerate(sorted_question):
        if len(return_que) == length:
            break
        if question_type[i] == 'where' and where_count < where_limit:
            return_que.append(que)
            where_count += 1
        elif question_type[i] == 'what' and what_count < what_limit:
            return_que.append(que)
            what_count += 1
        elif question_type[i] == 'which' and which_count < which_limit:
            return_que.append(que)
            which_count += 1
        elif question_type[i] == 'who' and who_count < who_limit:
            return_que.append(que)
            who_count += 1
        elif question_type[i] == 'when' and when_count < when_limit:
            return_que.append(que)
            when_count += 1
        elif question_type[i] == 'how' and how_count < how_limit:
            return_que.append(que)
            how_count += 1
        elif question_type[i] == 'other' and other_count < other_limit:
            return_que.append(que)
            other_count += 1
        elif question_type[i] == 'hownumber' and hownumber_count < hownumber_limit:
            return_que.append(que)
            hownumber_count += 1

    return return_que


def classify(Questions):
    question_type = []
    for i, question in enumerate(Questions):

        q_list = question.lower().split(' ')

        if q_list[0] == 'what':
            question_type.append('what')  # 1
        elif q_list[0] == 'where':
            question_type.append('where')  # 2
        elif q_list[0] == 'which':
            question_type.append('which')  # 1
        elif q_list[0] == 'who':
            question_type.append('who')  # 1
        elif q_list[0] == 'when':
            question_type.append('when')  # 2
        elif q_list[0] == 'how' and q_list[1] != 'many' and q_list[1] != 'much':
            question_type.append('how')  # 1
        elif q_list[1] == 'many' or q_list[1] == 'much':
            # print('-------------------------------------')
            question_type.append('hownumber')
        elif q_list[0] == 'whose':
            question_type.append('who')
        else:
            question_type.append('other')  # 2

    return question_type


def which_reduce(Questions, score, add_score):
    for i, question in enumerate(Questions):
        q_list = question.lower().split(' ')
        if 'which' in q_list:
            if q_list.index('which') > 0:
                score[i] -= add_score
    return score


def length_bonus(Questions, score, upper_len, lower_len, add_score):
    for i, question in enumerate(Questions):
        q_list = question.split(' ')
        if len(q_list) > lower_len and len(q_list) < upper_len:
            score[i] += add_score
    return score


def pos_tagger(txt):
    # print(txt)

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    # stop_words = set(stopwords.words('english'))

    # sent_tokenize is one of instances of
    # PunktSentenceTokenizer from the nltk.tokenize.punkt module

    parsed_text = nlp(txt)
    wordlist = []
    for token in parsed_text:
        tagged = (token.orth_, token.lemma_, token.tag_, token.ent_type_, token.pos_, token.dep_)
        wordlist.append(tagged)

    return wordlist


def punct_bonus(pos_return, score, upper_cont, add_score):
    for i, pos in enumerate(pos_return):
        pos_list = [tup[4] for tup in pos]
#         print(pos_list)
        count_punc = [1 for word in pos_list if word == 'PUNCT']
#         print(count_punc)
        if len(count_punc) < upper_cont:
            score[i] += add_score
    return score


def pronn_bonus(pos_return, score, lower_cont, add_score):
    for i, pos in enumerate(pos_return):
        pos_list = [tup[1] for tup in pos]
        que_list = [tup[0].lower() for tup in pos]
        count_punc = [1 for word in pos_list if word == '-PRON-']
        that_list = ['that', 'these', 'this', 'those', 'a']
        count_that = [1 for word in que_list if word in that_list]
        if len(count_punc)+len(count_that) < lower_cont:
            score[i] += add_score*2
    return score


def get_parse_tree(sentence):
    doc = nlp(sentence)
    result = []
    for token in doc:
        result.append((token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children]))
    return result


# grammar bonus

def grammar_bonus(ptree_return, score, lower_cont, add_score):
    for i, dep in enumerate(ptree_return):
        dep_list = [tup[1] for tup in dep]
        count_obj = [1 for word in dep_list if word.endswith('obj')]
        count_subj = [1 for word in dep_list if word.endswith('subj')]
#         print(len(count_punc))
        if len(count_obj) > lower_cont and len(count_subj) > lower_cont:
            score[i] += add_score
    return score



def get_score(Questions, length):
    if length > 20:
        # Wh bonus
        score = [0] * len(Questions)
        score = which_reduce(Questions, score, 1)

        # len bonus
        score = length_bonus(Questions, score, 20, 6, 1)

        pos_return = []
        for question in Questions:
            pos_return.append(pos_tagger(question))

        # punctuations count bonus
        score = punct_bonus(pos_return, score, 2, 1)

        # pronoun bonus
        score = pronn_bonus(pos_return, score, 1, 2)

        ptree_return = []
        for question in Questions:
            ptree_return.append(get_parse_tree(question))

        score = grammar_bonus(ptree_return, score, 0, 3)
    else:
        # Wh bonus
        score = [0] * len(Questions)
        score = which_reduce(Questions, score, 100)

        # len bonus
        score = length_bonus(Questions, score, 25, 6, 10)

        pos_return = []
        for question in Questions:
            pos_return.append(pos_tagger(question))

        # punctuations count bonus
        score = punct_bonus(pos_return, score, 2, 30)

        # pronoun bonus
        score = pronn_bonus(pos_return, score, 1, 20)

        ptree_return = []
        for question in Questions:
            ptree_return.append(get_parse_tree(question))

        score = grammar_bonus(ptree_return, score, 0, 30)
    return score


if __name__=='__main__':
    # TODO：如果你有直接运行的测试语句，请写在这里
    # s = get_score(["Is Jack Gay?","Who is Jack?","Hi Jack."])
    r = sort_question(Questions)
    pass