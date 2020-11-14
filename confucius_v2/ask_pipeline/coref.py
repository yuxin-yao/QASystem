#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
import sys
import spacyold as spacy
import neuralcorefwitholdspacy as neuralcoref
import logging
import pickle
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')
# from ask_pipeline.preprocessing import sentence_tokenization

COREF_THRESHOLD = 2
sys.path.append("/tmp")
model_name = "en_core_web_sm_210"

logging.info("loading %s"%model_name)
nlp = spacy.load(model_name)
logging.info("loaded %s"%model_name)
coref = neuralcoref.NeuralCoref(nlp.vocab)
# nlp.add_pipe(coref, name='neuralcoref')
# nlp.remove_pipe("tagger")
# nlp.remove_pipe("parser")
# nlp.remove_pipe("ner")
neuralcoref.add_to_pipe(nlp,max_dist=100)#, greedyness=0.6,max_dist = 100)

def do_coref(article:str)->str:
    logging.info("Doing coref...")
    doc_list = nlp_with_coref(article)
    newdoc = ""
    for doc in doc_list:
        for i in doc:
            # Try to replace corefed word
            # TODO: dont replace or find better result for stuff like:
            #  it->it
            #  their  -> Indians's
            #  his  -> His's
            #  his  -> An important person during the reign of Djoser
            #  themselves -> they
            #  him -> His
            #   it  -> the period in the third millennium  also known as the 'Age of the Pyramids' or 'Age of the Pyramid Builders' as it includes the great 4th Dynasty when King Sneferu perfected the art of pyramid building and the pyramids of Giza were constructed under the kings Khufu, Khafre, and Menkaure
            try:
                if i.tag_ in ("PRP", "PRP$", "WP", "WP$"):
                    if len(i._.coref_clusters) == 1 \
                            and max(i._.coref_scores[0].values()) > COREF_THRESHOLD :
                        # newword = max(i._.coref_scores[0].items(),key=lambda x:x[1])[0].string

                        if (len(i._.coref_clusters[0].main) == 1 and (i._.coref_clusters[0].main[0].tag_ in ("PRP$","PRP","DT","PRON","WP", "WP$"))) \
                                or len(i._.coref_clusters[0])>20:
                            newdoc += i.string
                            continue

                        # select the real name
                        newword = i._.coref_clusters[0].main.text.strip().split("\n")[0].strip()
                        # add 's for whose dep_ is poss
                        if i.dep_ == 'poss':
                            newword += "'s"
                            jjj = 1
                        newdoc += newword
                        logging.debug("Coref: " + i.string + " -> " + newword)
                        newdoc += " "
                        pass
                    else:  # not ref
                        newdoc += i.string
                else:  # not ref
                    newdoc += i.string
            except:
                logging.exception("Coref BUG!!")
                newdoc += i.string
        pass
    article = newdoc
    # if no_crlf: article.replace("\n", " Fuck. ")
    logging.info("Done coref.")
    return article


def nlp_with_coref(article:str)->list:
    # neuralcoref has a unknown bug which makes it impossible to treat article longer than approximately 40KB (it crashes the whole pipeline)
    # :( so sad having realizing it before spending an hour
    # break the article and treat separately if too long

    if len(article) > 45000:
        breakpoints_count = int(len(article) / 30000)
        logging.warning("Article too long, splitting for coref. "+ str(breakpoints_count))
        unprocessed_break_indexes = []
        for i in range(breakpoints_count): unprocessed_break_indexes.append(
            int(len(article) * (i + 1) / (breakpoints_count + 1)))
        article_list = []
        break_indexes = [0]
        # there is an exploit for the following code, but this code works anyway with wikipedia articles
        unprocessed_break_indexes.append(len(article))
        for i in range(len(unprocessed_break_indexes) - 1):
            for break_index in range(unprocessed_break_indexes[i], unprocessed_break_indexes[i + 1]):
                if article[break_index] == '\n': #and article[break_index + 1] == '\n':  # splits with 2 \n
                    break_indexes.append(break_index)
                    break

        break_indexes.append(len(article))
        for i in range(len(break_indexes) - 1):
            article_list.append(article[break_indexes[i]:break_indexes[i + 1]])

        ret = []
        for i in article_list:
            ret.extend(nlp_with_coref(i))
            logging.info("Done processing one segment.")
        return ret

    doc = nlp(article)
    return [doc]

if __name__=='__main__':
    # TODO：如果你有直接运行的测试语句，请写在这里
    with open("../data/Development_data/set4/a9.txt","r") as f:
        # doc=f.read()
        doc = "In 1970, NASA established two committees, one to plan the engineering side of the space telescope project, and the other to determine the scientific goals of the mission"
    f = do_coref(doc)
    i=1
    pass