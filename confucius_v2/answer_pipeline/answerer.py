#!/opt/conda/bin/python
# -*- coding: utf-8 -*-


# -------------------------------------------------------
#                 Documents & Dev Tools
# -------------------------------------------------------
# TODO: 把你用到的所有参考资料堆在这里，格式是简介和链接，举例：

# spacy dependency tree visualizer
# https://explosion.ai/demos/displacy

# dependency tree component lookup
# https://blog.csdn.net/lihaitao000/article/details/51812618

# -------------------------------------------------------
#                    Answerers go here
# -------------------------------------------------------
# TODO: 把你写的回答类放在这里，请按照下面的模版实现接口：
#  记得更改类名

class AnswerYesNoQuestion:
    def __init__(self, article: list = []):
        # TODO: 写你的初始化函数，比如载入模型啥的
        #  如果你载入了什么东西，请存进self里
        #  你可以自定义这里init的参数，比如模型名称，你需要的其他数据等
        #  article是preprocess过的整篇文章，是list of str，可用可不用
        pass

    def answer_one(self, question: str, sentence: list) -> list:
        # TODO：这个函数接受一个问题以及该问题相关句子概率列表，返回按概率排序的答案列表
        #  举例：question="What is the meaning of life?"
        #       sentence=[ ["Life is a word.",0.97], ["Life means responsibility", 0.96] ]
        #       return = [ ["Responsibility.",0.533],["A word.", 0.51] ]
        #  请尽量满足return的条件。如果你没有算概率，请保证返回有序的0-1数值；如果你只有一个答案，请也用list包好答案
        Question = question
        result = []
        for Sentence, prob in sentence:

            import spacy
            import numpy as np

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

    def answer_batch(self, questions: list, sentences: list) -> list:
        # TODO: 这个函数和上面answer_one类似，区别在于这个函数是批量回答问题
        #  函数的参数就是list形式的answer_one所传参数；return的东西也是用list包起来的answer_one输出
        #  设立这个函数的其中原因在于，有些人的回答方式能借助库来加速批量处理
        result_list = []
        for i, Question in enumerate(questions):
            sentences_one = sentences[i]
            result_list.append(self.answer_one(Question, sentences_one))

        return result_list

class QuestionTemplate:
    def __init__(self, article: list = []):
        # TODO: 写你的初始化函数，比如载入模型啥的
        #  如果你载入了什么东西，请存进self里
        #  你可以自定义这里init的参数，比如模型名称，你需要的其他数据等
        #  article是preprocess过的整篇文章，是list of str，可用可不用
        pass

    def answer_one(self,question:str,sentence:list)->list:
        # TODO：这个函数接受一个问题以及该问题相关句子概率列表，返回按概率排序的答案列表
        #  举例：question="What is the meaning of life?"
        #       sentence=[ ["Life is a word.",0.97], ["Life means responsibility", 0.96] ]
        #       return = [ ["Responsibility.",0.533],["A word.", 0.51] ]
        #  请尽量满足return的条件。如果你没有算概率，请保证返回有序的0-1数值；如果你只有一个答案，请也用list包好答案
        pass

    def answer_batch(self,questions:list,sentences:list)->list:
        # TODO: 这个函数和上面answer_one类似，区别在于这个函数是批量回答问题
        #  函数的参数就是list形式的answer_one所传参数；return的东西也是用list包起来的answer_one输出
        #  设立这个函数的其中原因在于，有些人的回答方式能借助库来加速批量处理
        pass

    # TODO: 如果你有什么不需要共用的辅助函数，请写在class里，不然则可以写在下面的utilities里


# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------

# TODO: 如果你有什么需要共用的辅助函数，请写在这里

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
    from nltk.corpus import wordnet
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

# TODO：如果你有与测试你的代码相关的函数，请写在这里

if __name__=='__main__':
    # TODO：如果你有直接运行的测试语句，请写在这里
    pass
