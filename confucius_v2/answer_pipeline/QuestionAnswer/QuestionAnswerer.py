#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

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
class QuestionAnswerer:
    def __init__(self, article: list = []):
        # 写你的初始化函数，比如载入模型啥的
        #  如果你载入了什么东西，请存进self里
        #  你可以自定义这里init的参数，比如模型名称，你需要的其他数据等
        #  article是preprocess过的整篇文章，是list of str，可用可不用
        # self.nlp = spacy_nlp
        pass

    def answer_one(self,question:str,sentences:list)->list:
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
        #  下面是默认实现
        pass

    # 如果你有什么不需要共用的辅助函数，请写在class里，不然则可以写在下面的utilities里

# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------

# 如果你有什么需要共用的辅助函数，请写在这里


# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------

if __name__ == '__main__':
    # 如果你有直接运行的测试语句，请写在这里
    pass