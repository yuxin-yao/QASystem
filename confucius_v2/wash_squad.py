

base="./data/Development_data/"

import os
import json

article_list = []
for root, dirs, files in os.walk(base):
    for name in files:
        fp=os.path.join(root, name)
        if fp[-3:].lower()=='txt':
            article_list.append(fp)

squad = "./data/squad/train-v2.0.json"

with open(squad,"r") as f:
    squad=f.read()
j = json.loads(squad)["data"]

for i in article_list:
    with open(i,"r") as f:
        c = f.read()
    title=c.split("\n",1)[0].strip()
    for k in j:
        if k['title'].strip()==title:
            print(title)



pass