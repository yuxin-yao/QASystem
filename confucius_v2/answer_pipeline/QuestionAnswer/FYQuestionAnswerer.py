#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
import spacy
from answer_pipeline.QuestionAnswer.QuestionAnswerer import QuestionAnswerer

# pip install pattern3
# from pattern.en import sentiment
# nlp = spacy.load("en_core_web_sm")
# nlp1 = spacy.load("en_core_web_md")

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
class GenericHowQuestionAnswerer(QuestionAnswerer):
    def __init__(self,spacy_nlp):
        # 写你的初始化函数，比如载入模型啥的
        #  如果你载入了什么东西，请存进self里
        #  你可以自定义这里init的参数，比如模型名称等
        #  article是preprocess过的整篇文章，是list of str，可用可不用
        super().__init__()
        self.nlp = spacy_nlp

    def answer_one(self, question: str, sentences: list) -> list:
        # 这个函数接受一个问题以及该问题相关句子概率列表，返回按概率排序的答案列表
        #  举例：question="What is the meaning of life?"
        #       sentence=[ ["Life is a word.",0.97], ["Life means responsibility", 0.96] ]
        #       return = [ ["Responsibility.",0.533],["A word.", 0.51] ]
        #  请尽量满足return的条件。如果你没有算概率，请保证返回有序的0-1数值；如果你只有一个答案，请也用list包好答案
        question = question.lower()
        pos, tokens, dependency = get_ner(spacy_nlp=self.nlp,sentence = question)
        sentences_tokens = []
        sentences_pos = []
        for s in sentences:
            s = s[0]
            s1 = get_ner(self.nlp,s)
            sentences_pos.append(s1[0])
            sentences_tokens.append(s1[1])
        if tokens[0] == 'how':
            # answer how question
            # 1. how do/does/did/are/is
            if pos[1] == 'AUX':
                for i in range(0, len(sentences_tokens)):
                    curr_p = sentences_pos[i]
                    curr_token = sentences_tokens[i]
                    if 'ADV' in curr_p:
                        return [sentences[i]]

            # 3. how friendly/ha
            elif pos[1] == 'ADJ':
                # Fixed get word similarity using spacy word embedding
                for i in range(0, len(sentences)):
                    curr_p = sentences_pos[i]
                    curr_token = sentences_tokens[i]
                    if 'ADV' in curr_p:
                        index = curr_p.index('ADV')
                        if index < len(curr_p) - 1 and curr_p[index + 1] == 'ADJ':
                            return [[curr_token[index] + " " + curr_token[index + 1], 1]]
                    elif 'ADJ' in curr_p:
                        index = curr_p.index('ADJ')
                        question_vector = self.nlp(tokens[1] + "")
                        curr_token_vector = self.nlp(curr_token[index] + "")
                        sim = question_vector.similarity(curr_token_vector)
                        if sim > 0.5:
                            return [['very', 1]]
                        else:
                            return [['no', 1]]
                        # sentiment_question = sentiment(question).assessments[0][1]
                        # sentiment_sentence = sentiment(curr_token).assessments[0][1]
                        # if sentiment_question * sentiment_sentence < 0:
                        #     return [['no', 1]]
                        # else:
                        # return [['very', 1]]
            return [sentences[0]]

        else:
            return [sentences[0]]


class WhatWhichQuestionAnswerer(QuestionAnswerer):
    def __init__(self):
        super().__init__()

    def answer_one(self, question: str, sentences: list) -> list:
        # print("What question is called")
        question = question.lower()
        pos, tokens, dependency = get_ner(question)
        sentences_tokens = []
        sentences_pos = []
        for s in sentences:
            s = s[0]
            s1 = get_ner(s)
            sentences_pos.append(s1[0])
            sentences_tokens.append(s1[1])
        if tokens[0] == 'what' or tokens[0] == 'which':
            for i in range(0, len(pos)):
                curr_pos = pos[i]
                if curr_pos == 'NOUN':
                    if i < len(pos) - 1 and pos[i + 1] == "AUX":
                        curr_object = tokens[i]
                        for s in sentences:
                            if curr_object in s[0]:
                                return [s]
            return [sentences[0]]
        else:
            return [sentences[0]]



class NumberHowQuestionAnswerer(QuestionAnswerer):
    def __init__(self, spacy_nlp):
        # 写你的初始化函数，比如载入模型啥的
        #  如果你载入了什么东西，请存进self里
        #  你可以自定义这里init的参数，比如模型名称等
        #  article是preprocess过的整篇文章，是list of str，可用可不用
        super().__init__()
        self.nlp = spacy_nlp

    def get_ner(self, sentence):
        tokens = []
        tags = []
        dependency = []

        doc7 = self.nlp(sentence)
        for token in doc7:
            tokens.append(token.text)
            tags.append(token.pos_)
            dependency.append(token.dep_)
        return tags, tokens, dependency

    def answer_one(self, question: str, sentences: list) -> list:
        question = question.lower()
        pos, tokens, depedency = self.get_ner(question)
        sentences_tokens = []
        sentences_pos = []
        sentences_dependency = []
        number_list = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
        for s in sentences:
            s = s[0]
            s1 = self.get_ner(s)
            sentences_pos.append(s1[0])
            sentences_tokens.append(s1[1])
            sentences_dependency.append(s1[2])
        for i in range(0, len(sentences)):
            curr_p = sentences_pos[i]
            curr_token = sentences_tokens[i]
            curr_dep = sentences_dependency[i]
            # Fixed: number not return in full
            if 'NUM' in curr_p:
                index = curr_p.index('NUM')
                curr_d = curr_dep[index]
                if index < len(curr_p) - 1 and curr_p[index + 1] == 'NOUN':
                    return [[curr_token[index] + " " + curr_token[index + 1], 1]]
                elif curr_d == 'compound':
                    res = ' '.join(curr_token[index:])
                    return [[res, 1]]
                else:
                    return [[curr_token[index], 1]]
            else:
                for num in number_list:
                    if num in curr_token or num.lower() in curr_token:
                        index = curr_token.index(num)
                        return [[curr_token[index]], 1]



    def answer_batch(self, questions: list, sentences: list) -> list:
        # TODO: 这个函数和上面answer_one类似，区别在于这个函数是批量回答问题
        #  函数的参数就是list形式的answer_one所传参数；return的东西也是用list包起来的answer_one输出
        #  设立这个函数的其中原因在于，有些人的回答方式能借助库来加速批量处理
        answers = []
        for question in questions:
            # sentences and sentence??
            answers.append(self.answer_one(question, sentences))
        return answers



    # TODO: 如果你有什么不需要共用的辅助函数，请写在class里，不然则可以写在下面的utilities里


# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------

# TODO: 如果你有什么需要共用的辅助函数，请写在这里
def get_ner(spacy_nlp, sentence):
    tokens = []
    tags = []
    dependency = []

    doc7 = spacy_nlp(sentence)
    for token in doc7:
        tokens.append(token.text)
        tags.append(token.pos_)
        dependency.append(token.dep_)
    return tags, tokens, dependency

# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------

# TODO：如果你有与测试你的代码相关的函数，请写在这里

if __name__ == '__main__':
    # TODO：如果你有直接运行的测试语句，请写在这里
    pass
