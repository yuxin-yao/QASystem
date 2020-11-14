#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

import numpy
from sentence_transformers import SentenceTransformer
import logging
# import pyserini
import scipy.spatial.distance as distance
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

# model = SentenceTransformer('bert-base-nli-mean-tokens') #  BERT-base model with mean-tokens pooling. Performance: STSbenchmark: 77.12
# model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens') # Performance: STSbenchmark: 86.31
# model = SentenceTransformer('bert-base-nli-stsb-mean-tokens') # Performance: STSbenchmark: 85.14
# model = SentenceTransformer('bert-base-wikipedia-sections-mean-tokens') # 80.42% accuracy on Wikipedia sections test set.


class TransformerFinder:
    def __init__(self,model_name:str = "bert-base-nli-stsb-mean-tokens",sentences:list=[]):
        self.model_name = model_name
        logging.info("Initializing TransformerFinder:" + model_name)
        self.model = model = SentenceTransformer(model_name)
        logging.info("Initialized TransformerFinder:" + model_name)
        self.sentences = sentences
        self.embeddings = self.calculate_article_embedding(sentences)
        pass

    def calculate_article_embedding(self,sentences):
        logging.info("Generating sentence embedding...")
        sentence_embeddings = self.model.encode(sentences)
        logging.info("Sentence embedding generated!")
        return sentence_embeddings
        pass

    def calculate_sentence_embedding(self,sentence:str) -> numpy.ndarray:
        sentence_embedding = self.model.encode([sentence])
        return sentence_embedding[0]

    def get_related_sentences_ranking(self,sentence:str,sentence_embedding:numpy.ndarray=None):
        if not sentence_embedding:
            sentence_embedding = self.calculate_sentence_embedding(sentence)
        similarities = [self.cosine_similarity(sentence_embedding,x) for x in self.embeddings]
        similarity_pairs = zip(range(len(sentence_embedding)),similarities)
        index_ranking = sorted(similarity_pairs,key=lambda x:x[1],reverse = True)
        return [[self.sentences[x[0]],x[1]] for x in index_ranking]
        pass

    # Use this function if possible. Batch process will always be faster
    def batch_get_related_sentences_ranking(self,sentences:list):
        r=[]
        sentence_embeddings = self.calculate_article_embedding(sentences)
        for i in sentence_embeddings:
            r.append(self.get_related_sentences_ranking(i))
        return r

    def cosine_similarity(self,vector_a, vector_b):
        # # num = float(vector_a.T * vector_b)  # 若为行向量则 A * B.T
        # num = float(vector_a * vector_b.T)
        # denom = numpy.linalg.norm(vector_a) * numpy.linalg.norm(vector_b)
        # cos = num / denom  # 余弦值
        # return cos
        return 1 - distance.cosine(vector_a, vector_b)


class BM25RM3LRFinder:
    # TODO: Implement this: https://arxiv.org/pdf/1904.08861.pdf
    #  It seems that bert is way more accurate than BM25RM3: https://www.aclweb.org/anthology/D19-3004.pdf
    #  So I am thinking about not implementing it.
    #  https://github.com/castorini/pyserini
    #  https://github.com/castorini/anserini

    pass

class MixFinder:
    # TODO: mix the two finders
    pass

# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------

if __name__=='__main__':
    from ask_pipeline.preprocessing import *
    with open("../data/Development_data/set1/a1.txt","r") as f:
        doc = f.read()
        seprated_sentences = preprocess(doc)
        original_sentences = sentence_tokenization(doc) # add the unpreprocessed sentences just in case

    article = [x.text for x in seprated_sentences]
    article.extend([x.text for x in original_sentences])
    article = list(set(article))

    question = 'Who was succeeded by his son, Khufu?' # this example would find a correct sentence.
    question = 'Who built the Great Pyramid of Giza?' # this example would find an incorrect sentence if preprocessed sentences are not added
    question = 'Is the Great Pyramid of Giza build by Khufu?' # this example would find a correct sentence.
    question = 'Who the fuck was succeeded by his son, Khufu?' # this example would find a correct sentence.
    question = 'Who the fuck was succeeded by his son?'  # this example would find a correct sentence.

    # finder = TransformerFinder("bert-base-wikipedia-sections-mean-tokens",article)
    finder = TransformerFinder("bert-base-nli-stsb-mean-tokens", article)
    t = finder.get_related_sentences_ranking(question)
    print(t[0])

    pass