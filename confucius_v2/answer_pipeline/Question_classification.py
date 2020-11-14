import h5py
from answer_pipeline.SentenceFinder import TransformerFinder
from sklearn.neural_network import MLPClassifier
import numpy as np

#TODO
# question types:
#     what
#     how
#         generic how
#         how many
#         how long
#     who
#     which
#     why: http://ceur-ws.org/Vol-2036/T6-1.pdf         # this category will not be detailly classfied
#         informational why: Why are rabbits eyes red? Why is Indiglo called Indiglo? Why do scuba divers go into the water backwards?
#         historical why: Why were people recruited for the Vietnam War? Why did the Globe Theatre burn down? c. Why were medieval castles built?
#         situational/contextural why: hy do the clouds darken when it rains? Why do you say "God bless you" when people sneeze? Why does the moon turn orange?
#         opinionated why: Why was my payment in a message cancelled? Why are some people ’doublejointed’? Why do we laugh?
#     when
#     where
#     yesno

# https://www.lawlessenglish.com/learn-english/grammar/questions-yes-no/

# simple rule-based classifier with spacy
class SimpleQuestionClassifier:
    def __init__(self):
        pass

    def train(self):
        pass
    def fit(self,s):
        pass

# classifier with word embedding
class TransformerSKLearnClassifier:
    def __init__(self,classifier,transformer_model_name=""):

        from sklearn.ensemble import RandomForestClassifier
        self.tff = TransformerFinder(transformer_model_name)
        self.classifier = classifier
        self.embeddings = None
        pass

    def train(self,train,train_labels):
        if self.embeddings is None: self.embeddings = self.tff.calculate_article_embedding(train)
        self.classifier.fit(self.embeddings,train_labels)
        pass

    def classify(self,s):
        embedding = self.tff.calculate_sentence_embedding(s)
        return self.classifier.predict([embedding])
        pass

    # Use this function if possible. Batch process will always be faster
    def batch_classify(self,questions):
        embeddings = self.tff.calculate_article_embedding(questions)
        return self.classifier.predict(embeddings)
        pass

    def score(self,test,test_tag):
        embeddings = self.tff.calculate_article_embedding(test)
        return self.classifier.score(embeddings,test_tag)

class HardClassifier:
    def __init__(self, spacy_nlp):
        self.nlp = spacy_nlp


    def get_parse_tree(self, sentence):

        doc = self.nlp(sentence)
        result = []
        for token in doc:
            result.append((token.text, token.lemma_, token.dep_, token.tag_, token.head.text, token.head.pos_,
                           [child for child in token.children], token.pos_))
        return result

    def what_classify(self, i, pos_tag, denp, ques_list, head_pos_tag):
        type_que = str()
        what_idx = i #ques_list.index('what')
        root_idx = denp.index('ROOT')
        if denp[what_idx] == 'attr':
            type_que = 'whatwhich'
        elif pos_tag[what_idx] == 'WDT':
            if ques_list[what_idx + 1] == 'location':
                type_que = 'where'
            elif ques_list[what_idx + 1] == 'time':
                type_que = 'when'
        elif head_pos_tag[what_idx] == 'AUX' or head_pos_tag[what_idx] == 'VERB':
            type_que = 'whatwhich'
        if root_idx < what_idx:
            type_que = ''
        return type_que

    def which_classify(self, i, pos_tag, denp, ques_list, head_pos_tag):
        type_que = str()
        root_idx = denp.index('ROOT')
        which_idx = i #ques_list.index('which')
        if pos_tag[which_idx] == 'WDT':
            if ques_list[which_idx + 1] == 'location':
                type_que = 'where'
            elif ques_list[which_idx + 1] == 'time':
                type_que = 'when'
            elif head_pos_tag[which_idx] == 'AUX' or head_pos_tag[which_idx] == 'VERB':
                type_que = 'whatwhich'
        if root_idx < which_idx:
            type_que = ''
        return type_que

    def where_classify(self, i, pos_tag, denp, ques_list, head_pos_tag):
        type_que = str()
        root_idx = denp.index('ROOT')
        where_idx = i #ques_list.index('where')
        # if head_pos_tag[where_idx] == 'AUX' or head_pos_tag[where_idx] == 'VERB':
        type_que = 'where'
        if root_idx < where_idx:
            type_que = ''
        return type_que

    def when_classify(self, i, pos_tag, denp, ques_list, head_pos_tag):
        type_que = str()
        root_idx = denp.index('ROOT')
        when_idx = i #ques_list.index('when')
        # if head_pos_tag[when_idx] == 'AUX' or head_pos_tag[when_idx] == 'VERB':
        type_que = 'when'
        if root_idx < when_idx:
            type_que = ''
        return type_que



    def who_classify(self, i, pos_tag, denp, ques_list, head_pos_tag):
        # type_que = str()
        root_idx = denp.index('ROOT')
        who_idx = i #ques_list.index('who')
        # if head_pos_tag[who_idx] == 'AUX' or head_pos_tag[who_idx] == 'VERB':
        type_que = 'who'
        if root_idx < who_idx:
            type_que = ''
        return type_que

    def whose_classify(self, i, pos_tag, denp, ques_list, head_pos_tag, pos_tag_tag):
        type_que = str()

        whose_idx = i  # ques_list.index('whose')
        # if head_pos_tag[who_idx] == 'AUX' or head_pos_tag[who_idx] == 'VERB':
        count = 0
        #         for i in range(whose_idx, len(ques_list), 1):
        #             if pos_tag_tag[i] == 'VERB':
        #                 count += 1
        #         if count == 0:
        #             type_que = 'who'
        type_que = 'whose'
        return type_que

    def how_classify(self, i, pos_tag, denp, ques_list, head_pos_tag, pos_tag_tag):
        type_que = str()
        root_idx = denp.index('ROOT')
        how_idx = i #ques_list.index('how')
        if ques_list[how_idx + 1] == 'many' or ques_list[how_idx + 1] == 'much':
            type_que = 'hownumber'
        else:
            type_que = 'how'
        if root_idx < how_idx:
            type_que = ''
        return type_que

    def yesno_classify(self, aux_idx, pos_tag, denp, ques_list, head_pos_tag):
        type_que = str()
        root_idx = denp.index('ROOT')
        type_que = 'yesno'

        if root_idx < aux_idx:
            type_que = ''
        return type_que



    def classify_question_one(self, Question):
        parse = self.get_parse_tree(Question)
        pos_tag = [row[3] for row in parse]
        denp = [row[2] for row in parse]
        ques_list = [row[0].lower() for row in parse]
        head_pos_tag = [row[5] for row in parse]
        pos_tag_tag = [row[-1] for row in parse]
        type_que = []
        for i, word_que in enumerate(ques_list):
            if word_que == 'what':
                type_que.append(self.what_classify(i, pos_tag, denp, ques_list, head_pos_tag))
            elif word_que == 'which':
                type_que.append(self.which_classify(i, pos_tag, denp, ques_list, head_pos_tag))
            elif word_que == 'where':
                type_que.append(self.where_classify(i, pos_tag, denp, ques_list, head_pos_tag))
            elif word_que == 'when':
                type_que.append(self.when_classify(i, pos_tag, denp, ques_list, head_pos_tag))
            elif word_que == 'who':
                type_que.append(self.who_classify(i, pos_tag, denp, ques_list, head_pos_tag))
            elif word_que == 'whose':
                type_que.append(self.whose_classify(i, pos_tag, denp, ques_list, head_pos_tag, pos_tag_tag))
            elif word_que == 'how':
                type_que.append(self.how_classify(i, pos_tag, denp, ques_list, head_pos_tag, pos_tag_tag))
            elif pos_tag_tag[i] == 'AUX' or denp[i] == 'aux':
                type_que.append(self.yesno_classify(i, pos_tag, denp, ques_list, head_pos_tag))

        result = [tag for tag in type_que if len(tag) > 0]

        if len(result) == 0:
            return ''
        else:
            return result[0]

    def classify_question_batch(self, Questions):

        result = [self.classify_question_one(Question) for Question in Questions]
        return result


    # -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------



if __name__=='__main__':
    # https://cogcomp.seas.upenn.edu/Data/QA/QC/
    def read_dataset(path):
        with open(path, "r", errors='ignore') as f:
            data = f.read() \
                .strip() \
                .replace(" ?", "?") \
                .replace("``", "") \
                .replace("''", "") \
                .split("\n")
            data = [x.split(' ',1) for x in data]
        train = [x[1] for x in data]
        labels = [x[0] for x in data]
        return train,labels

    classifier = MLPClassifier(solver='lbfgs')#, alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
    trfc = TransformerSKLearnClassifier(transformer_model_name="bert-base-nli-stsb-mean-tokens",classifier=classifier)

    train_set,train_labels = read_dataset("/tmp/train_5500.label")
    test_set, test_labels = read_dataset("/tmp/TREC_10.label")
    trfc.train(train=train_set,train_labels=train_labels)
    s = "What is the meaning of life?"
    trfc.classify(s)

    pass