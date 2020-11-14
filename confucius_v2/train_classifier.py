import flair

import spacy
import neuralcoref
import pickle

from sklearn.neural_network import MLPClassifier

from answer_pipeline.Question_classification import TransformerSKLearnClassifier


def read_dataset(path):
    with open(path, "r", errors='ignore') as f:
        data = f.read() \
            .strip() \
            .replace(" ?", "?") \
            .replace("``", "\"") \
            .replace("''", "\"") \
            .split("\n")
        data = (tuple(x.split(' ', 1)) for x in data)
    train = (x[1] for x in data)
    labels = (x[0] for x in data)
    return data

d = {
    "ENTY:veh":"whatwhich",
    "NUM:ord":"whatwhich",
    "NUM:temp":"hownumber",
    "NUM:volsize":"hownumber",
    "DESC:desc":"whatwhich",
    "ENTY:product":"whatwhich",
    "ENTY:dismed":"whatwhich",
    "ENTY:word":"whatwhich",
    "ENTY:currency":"whatwhich",
    "NUM:dist":"hownumber",
    "HUM:title":"what",
    "ABBR:exp":"whatwhich",
    "ENTY:religion":"where",
    "NUM:date":"when",
    "ENTY:sport":"whatwhich",
    "LOC:state":"whatwhich",
    "ENTY:techmeth":"", # how
    "ENTY:animal":"whatwhich",
    "NUM:weight":"hownumber",
    "ENTY:event":"whatwhich",
    "DESC:manner":"how",
    "ENTY:color":"whatwhich",
    "DESC:def":"whatwhich",
    "HUM:ind":"whatwhich",
    "ENTY:food":"whatwhich",
    "NUM:code":"whatwhich",
    "NUM:period":"when",
    "ENTY:symbol":"whatwhich",
    "LOC:city":"where", # what where
    "NUM:perc":"hownumber",
    "ENTY:letter":"whatwhich",
    "NUM:money":"hownumber",
    "DESC:reason":"why",
    "HUM:desc":"who",
    "LOC:other":"", # what where
    "ENTY:instru":"whatwhich",
    "NUM:other":"hownumber",
    "ENTY:termeq":"whatwhich",
    "ENTY:plant":"whatwhich",
    "ENTY:cremat":"whatwhich",
    "HUM:gr":"",    #who what
    "NUM:count":"hownumber",
    "ENTY:lang":"whatwhich",
    "ENTY:body":"whatwhich",
    "ENTY:substance":"whatwhich",
    "LOC:mount":"whatwhich",
    "NUM:speed":"hownumber",
    "ABBR:abb":"whatwhich",
    "LOC:country":"whatwhich",
    "ENTY:other":"whatwhich",
    "yesno:":"yesno"
}


#
# for k,v in d.items():
#     for i in pairs:
#         if i[0] == k:
#             print(i)
#     input("asdf")

train=[]
train_tag = []
test=[]
test_tag = []

pairs = []
flist=["/tmp/train_1000.label","/tmp/train_2000.label","/tmp/train_3000.label","/tmp/train_4000.label","/tmp/train_5500.label","/tmp/yesno"]
for i in flist:pairs.extend(read_dataset(i))
pairs = list(set(tuple(pairs)))
new_pairs = []
for i in pairs:
    if d[i[0]] !="":
        train_tag.append(d[i[0]])
        train.append(i[1])

for i in read_dataset("/tmp/TREC_10.label"):
    if d[i[0]] != "":
        test_tag.append(d[i[0]])
        test.append(i[1])



classifier = MLPClassifier(solver='adam',max_iter=1000)#, alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1)
trfc = TransformerSKLearnClassifier(transformer_model_name="bert-base-nli-stsb-mean-tokens",classifier=classifier)

trfc.train(train=train,train_labels=train_tag)
s = "What is the meaning of life?"
trfc.classify(s)
embeddings = trfc.tff.calculate_article_embedding(test)
print(trfc.classifier.score(embeddings,test_tag))

with open("question_classifier_bert_adam","wb+") as f:
    pickle.dump(trfc.classifier,f)
i=1
pass