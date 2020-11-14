# -*- coding: utf-8 -*-
import spacy
import re
from ask_pipeline.coref import do_coref
import pickle
# from spacy.pipeline import
import pyinflect
import flair
from flair.data import Sentence
from flair.models import SequenceTagger

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S')

# tagger = SequenceTagger.load('ner')
nlp = spacy.load("en_core_web_sm")
merge_ents = nlp.create_pipe("merge_entities")
nlp.add_pipe(merge_ents)
merge_nps = nlp.create_pipe("merge_noun_chunks")
nlp.add_pipe(merge_nps)
# from spacy.tokens import Token


# -------------------------------------------------------
#                 Documents & Dev Tools
# -------------------------------------------------------

# spacy dependency tree visualizer
# https://explosion.ai/demos/displacy

# dependency tree component lookup
# https://blog.csdn.net/lihaitao000/article/details/51812618
# https://nlp.stanford.edu/software/dependencies_manual.pdf

# pos tags
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# -------------------------------------------------------
#                    Exposed Functions
# -------------------------------------------------------
# TODO FIX BAD:
#  MPEG-1 or MPEG-2 Audio Layer III, more commonly referred to as MP3, is an audio coding format for digital audio which uses a form of lossy data compression.
#   ->


def preprocess(article_str:str, no_junk=True,with_coref=True,more_sentences=True) -> list:
    logging.info("Started preprocessing")
    proposed_question = []
    useful_fragments = []
    processed_sentence = []

    article_str = bad_fix(article_str)

    # processed_sentence.extend(extract_parenthesis_from_article(article_str,with_coref=with_coref,no_junk=no_junk))

    if not check_parentheses_valid(article_str):
        logging.warning("Parenthesis not match for this article!")
    article_str = remove_parentheses(article_str, max_search_span=200)

    # article_str = article_str.replace("\n"," ")#re.sub("\n"," ",article_str)     # remove \n
    article_tokenized = sentence_tokenization(article=article_str,with_coref=with_coref,no_junk=no_junk,no_crlf=True)
    logging.info("Tokenized %d sentences"%len(article_tokenized))
    proposed_question.extend(extract_existed_questions(article_tokenized))
    processed_sentence.extend(batch_process(single_sentence_splitting_pipeline,article_tokenized))

    if more_sentences:processed_sentence.extend(article_tokenized)
    return list(set(processed_sentence))

def single_sentence_splitting_pipeline(sentence:spacy.tokens.doc.Doc,more_sentences=True) -> list:
    return_sentence_list=[]

    return_sentence_list.extend(extract_relcl(sentence))
    return_sentence_list.extend(extract_advcl(sentence))

    sentence_without_clauses = extract_main_sentence(sentence)
    if sentence_without_clauses is None:
        return return_sentence_list

    if more_sentences: return_sentence_list.append(sentence_without_clauses)

    sentence_list_1 = split_semicolon(sentence_without_clauses)
    sentence_list_2 = []
    for i in sentence_list_1:
        sentence_list_2.extend(split_cc(i))

    return_sentence_list.extend(sentence_list_2)
    # return_sentence_list.extend(batch_process(appositive_removal,sentence_list_2))



    #Early in his reign, Amenemhet I was compelled to campaign in the Delta region, which had not received as much attention as upper Egypt during the 11th Dynasty.
    return batch_process(lambda x:nlp(fix_sentence(x.text)),return_sentence_list)

    # for sentence in tokenized:
    #     sentence = Sentence(sentence)
    #     for entity in sentence.get_spans('ner'):
    #         print(entity)
    #     pass

# -------------------------------------------------------
#                    Pipeline elements
# -------------------------------------------------------


def bad_fix(article:str)->str:
    return article\
        .replace(")—",") - ")\
        .replace("—"," - ")\
        .replace(" ",' ')   \
        .replace("as well as","and")\
        .replace(" 's ","'s ")\
        .replace("s's ","s' ") \
        .replace("Also,", "") \
        .replace("However,", "") \
        .replace(" also ", " ")\
        .strip()
def appositive_removal(sentence:spacy.tokens.doc.Doc):
    # 同位语切换
    # sentence in doc type
    # print("sentence is", sentence.text)
    result = []
    appoStart = 0
    appoEnd = 0
    available_dep = ["verb", "aux", "ROOT", "cc"]
    clauses = []
    for i in range(0, len(sentence)):
        # print(sentence[i].text, " ", sentence[i].pos_, " ", sentence[i].dep_)
        # print(sentence[i].text + " " + sentence[i].dep_)
        if sentence[i].text == ',' and appoStart == 0:
            appoStart = i
        elif sentence[i].text == ',' and appoEnd == 0 and sentence[i+1].dep_ in available_dep:
            appoEnd = i
    if appoEnd == 0:
        appoEnd = len(sentence)
    # print("appo start is", appoStart)
    # print("appo end is", appoEnd)
    # print(sentence[appoStart+1])
    result.append(sentence[:appoStart])
    result.append(sentence[appoEnd+1:])
    # in case of multiple appositives
    final_result = []
    for r in result:
        for word in r:
            if word.dep_ == 'appos':
                continue
        final_result.append(r)
    concatenated = ""
    for r in final_result:
        concatenated += r.text + " "
    # return
    # s1: remove appositive
    # s2: with appositive to replace main clauses
    s2 = sentence[appoStart+1:appoEnd]
    s2_nextP =  sentence[appoEnd+1:]
    final_result2 = []
    for word in s2:
        final_result2.append(word)
    for word in s2_nextP:
        final_result2.append(word)
    return [r, final_result2]


def is_junk_sentence(sentence:spacy.tokens.doc.Doc,strict = False) -> bool:
    # junk sentence is shorter than 4 words
    if len(sentence)<5: return True
    # junk sentence does not have root
    has_root = False
    # junk sentence does not have verb
    verb = 0
    # junk sentence does not have subj
    has_subj = False
    # junk sentence does have more than 1 tokens with \n
    crlf = 0

    accepted_verb_dep = ("nsubj","nsubjpass","aux","auxpass")

    for i in sentence:
        if i.dep_ == "ROOT":
            has_root = True
        elif i.dep_ in ("nsubj", "nsubjpass"):
            has_subj = True
        if not strict and i.dep_ in accepted_verb_dep:
            verb+=1
        elif i.pos_ == "VERB":
            verb+=1
        if "\n" in i.text:
            crlf +=1
    if not has_root \
            or verb==0 \
            or not has_subj \
            or crlf>1:

        return True
    # normal sentence in an article should not be a question
    if sentence[-1].text.strip()=="?":return True
    # TODO: no colon; must have noun phrase?
    return False
    pass

def sentence_tokenization(article:str,with_coref = False,no_junk = True,no_crlf=False) -> list:
    logging.info("Started tokenizing")
    tokenized = []
    if with_coref:
        article = do_coref(article)
    if no_crlf: article = article.replace("\n", " Fuck. ")
    doc = nlp(article)
    for sent in doc.sents:
        tokenized.append(nlp(sent.text.strip().strip("Fuck.")))
    logging.info("Done tokenizing")
    if no_junk: return list(filter(lambda x:not is_junk_sentence(x),tokenized))
    return tokenized

def extract_existed_questions(sentences:list)->list:
    ret = []
    for i in sentences:
        if len(i)<5:continue
        if (i[-1].string.strip() in ("?","？")) or (i[-2].string.strip() in ("?","？")):
            ret.append(i)
    return ret

def extract_from_parentheses(sentence: spacy.tokens.doc.Doc)->list:
        return_list = []
        spans = locate_parentheses(sentence)

        for i in spans:
            # find root of this fragment
            # assume there is only one root
            # exits if there is multiple root
            root = None
            for j in i:
                for k in j.children:
                    if k.i < i.start and k.dep_ !='punct':
                        if not root:
                            root = j
                        else:
                            logging.warning("multipal root found while extracting parentheses:"+i.string.strip())
                            return []

            # filter rootless fragment
            if not root: continue

            # extract if it is a full sentence (have nsubj and root is a verb of any kind)
            children_kinds = [x.dep_ for x in root.children]
            if ('nsubj' in children_kinds or 'nsubjpass' in children_kinds) and root.tag_ in ('VB','VBD','VBG','VBN','VBP','VBZ'):
                return_list.append(fix_sentence(i.string))

        pass



        # extract if it is a sentence without nsubj

        return return_list

def extract_relcl(sentence: spacy.tokens.doc.Doc,no_junk = True)->list:
        # input: However, Jefferson did not believe the Embargo Act, which restricted trade with Europe, would hurt the American economy.
        # output: The Embargo Act restricted trade with Europe.
        relcl_list=[]
        return_list=[]
        for i in sentence:
            if i.dep_ == "relcl":
                relcl_list.append(i)
        for i in relcl_list:
            # assumed word pointed by relcl will only have one nsubj
            if 'nsubj' not in [x.dep_ for x in list(i.children)] and 'nsubjpass' not in [x.dep_ for x in list(i.children)]:
                continue
            t = extend_node(i, without_dep=['punct', 'nsubj', 'nsubjpass'],with_token=[")",'"',"'"])#which=nsubj
            subject = extend_node(i.head, without_dep=['punct', 'relcl'],with_token=[")",'"',"'"])

            # TODO: add aux verb support (done but need more testing)
            # FIXIT:The period comprises two phases, the 11th Dynasty, which ruled from Thebes and the 12th Dynasty onwards which was centered on el-Lisht.
            # FIXIT:The northern skies have constellations that have mostly survived since Antiquity, whose common names are based on Classical Greek legends or those whose true origins have now been lost.
            # FIXIT: After toppling the last rulers of the 10th Dynasty, Mentuhotep began consolidating his power over all Egypt, a process which he finished by his 39th regnal year.
            # r=nlp(fix_sentence(subject.string.strip()+" "+t.doc[i.i:t.end].string.strip()))
            if len(set(range(subject.start,subject.end)) & set(range(t.start,t.end)) ) != 0:
                logging.error("Overlapping found while extracting relcl:"+str(sentence))
            r = nlp(fix_sentence(subject.string.strip() + " " + t.doc[t.start:t.end].string.strip()))
            if len(r)>3:return_list.append(r)
            pass
        for i in sentence:
            # assert len(i._.coref_clusters)<2
            pass
        return return_list


        pass

def extract_advcl(sentence: spacy.tokens.doc.Doc,no_junk = True)->list:
    # TODO: fix broken advcl instead of filtering them: Having thus acquired royal power, Sandracottos possessed India at the time Seleucos was preparing future glory."
    # another example: The empire was the largest to have ever existed in the Indian subcontinent, spanning over 5 million square kilometres.

    # TODO: give proposed questions (having xxxxx ->when did xxxx)

    advcl_list = []
    return_list = []
    for i in sentence:
        if i.dep_ == "advcl":
            advcl_list.append(i)
    for i in advcl_list:
        t = extend_node(i, without_dep=['punct', 'mark','advmod','ccomp'], with_token=[")", '"', "'"])

        # skip if sentence
        # Fixed for It is popularly suggested that cockroaches will "inherit the earth" if humanity destroys itself in a nuclear war.
        # TODO: here should have a proposed question
        if t[0].i!=0 and (sentence[t[0].i-1].pos_ == 'SCONJ' or sentence[t[0].i-1].lemma_.lower() in ('if','while','once','when')):continue

        fragment = nlp(t.text)
        potential_nsubj = find_main_subject(fragment)


        if len(potential_nsubj)==0:
            # try to fix advcl by adding nsubj
            root = i.head
            # fix: open clausal complement
            # The Western Kshatrapa ruler Nahapana is known to have ruled the former Satavahana territory, as attested by the inscriptions of his governor and son-in-law, Rishabhadatta.
            # redirect verb
            if root.dep_ == 'xcomp':
                root=root.head
            main_subject = find_main_subject(t,root = root)
            if len(main_subject) >1:
                logging.error("more than one main subject for advcl:"+i.text)
                continue

            # tense = nlp.vocab.morphology.tag_map[i.tag_]['Tense']
            # if tense != 'past':
            #     logging.error("root not VBN for advcl:"+i.text)
            #     continue

            # get verb
            verb = None
            for j in t:
                if j.pos_ == 'VERB':
                    verb = j
                    break
            if not verb:
                logging.error("no verb for advcl:"+i.text)
                continue

            # change verb tense accordingly
            # FIXed for The earliest cockroach-like fossils ("blattopterans" or "roachids") are from the Carboniferous period 320 million years ago, as are fossil roachoid nymphs.
            verb_new = None
            if root.pos_ == 'VERB':
                # fix for 'am' in It is popularly suggested that cockroaches will "inherit the earth" if humanity destroys itself in a nuclear war.
                # use original aux if found
                if root.lemma_=='be':
                    verb_new = root.text
                else:
                    verb_new = verb._.inflect(root.tag_)
            if not verb_new:
                verb_new = verb._.inflect("VBN")
            if not verb_new:
                verb_new = verb.text
            verb_new = verb_new.lower()


            if len(main_subject) == 0:
                logging.error("no main subj found while fixing advcl:"+sentence.text)
                continue
            r = main_subject[0].text

            if len(set(range(main_subject[0].start,main_subject[0].end)) & set(range(t.start,t.end)) ) != 0:
                logging.error("Overlapping found while extracting advcl:"+str(sentence))
                continue

            for j in t:
                if j == verb:
                    r += " " + verb_new
                else:
                    r += " " + j.text

            r = nlp(fix_sentence(r))

        else:
            r = nlp(fix_sentence(t.string.strip()))



        if len(r)>3:
            if not is_junk_sentence(r,strict=True) or not no_junk:return_list.append(r)
        # print("test advcl")
        # for s11 in return_list:
        #     print(s11)
        # EDIT: also added main to the return list, instead of avcel
        has_advcl = False
        advcl_index = -1
        for i in range (0, len(sentence)):
            if sentence[i].dep_ == 'advcl':
                has_advcl = True
            if sentence[i].dep_ == 'ROOT':
                advcl_index = i
        if has_advcl:
            main_subject_s = 0
            main_subject_e = 0
            for i in range(advcl_index, -1, -1):
                if sentence[i].text == ',':
                    main_subject_s = i + 1
                    break
            for i in range(advcl_index, len(sentence), 1):
                if sentence[i].text == ',' or sentence[i].dep_ == 'punct':
                    main_subject_e = i
                    break
            if main_subject_e == 0:
                main_subject_e = len(sentence)
        # print("main start is here", main_subject_s, main_subject_e)
        # print(sentence[main_subject_s:main_subject_e])
        return_list.append(sentence[main_subject_s:main_subject_e])

    return return_list

def split_semicolon(sentence: spacy.tokens.doc.Doc)->list:
    split_list = sentence.text.split(";")
    if len(split_list) <2 : return [nlp(fix_sentence(sentence.text))]
    return_list = []
    for i in split_list:
        t = nlp(fix_sentence(i))
        if not is_junk_sentence(t):
            return_list.append(t)
        else:       # if one of the split result is not valid, return unsplit sentence
            return [nlp(fix_sentence(sentence.text))]
    return return_list

def split_cc(sentence: spacy.tokens.doc.Doc)->list:
    if len(sentence)<4: return [sentence]
    main_subject = find_main_subject(sentence)
    if len(main_subject)!= 1: return [sentence]
    main_subject = main_subject[0]


    split_tokens = [sentence[0]]
    roots_of_fragments = []
    for i in sentence:
        if i.dep_ == 'ROOT': roots_of_fragments.append(i)
        if i.dep_ == 'cc':
            for j in sentence[i.i:len(sentence)]:
                if j.dep_ =='conj' and j.head == i.head and j.pos_ =='VERB':
                    split_tokens.append(i)
                    roots_of_fragments.append(j)
    split_tokens.append(sentence[-1])
    # FIXed the problem of directly returning sentence without splitting it
    if len(split_tokens) == 2:
        clause = []
        # for s in sentence:
        #     print(s, " ", s.dep_)
        for root in roots_of_fragments:
            for child in root.children:
                if child.dep_ == 'prep':
                    for s in sentence[child.i:len(sentence)]:
                        if s.dep_ == 'cc':
                            clause.append(sentence[: s.i])
                            if s.i+2< len(sentence) and (sentence[s.i+1].dep_ == 'conj' or sentence[s.i+2].dep_ == 'conj'):
                                str = ""
                                for s1 in sentence[:child.i+1]:
                                    str += s1.text + " "
                                for s1 in sentence[s.i+1:]:
                                    str += s1.text + " "
                                #
                                # str += [s1 for s1 in sentence[s.i+1:]]
                                # print(str)
                                clause.append(nlp(str))
                            break
                                # clause.append(sentence[:child.i+1] + sentence[s.i+1:])
        result = []
        for c in clause:
            # print(c)
            result.append(c)
        if len(result) == 0:
            result.append(sentence)
        return result

    fragments = []
    for i in range(len(split_tokens)-1):
        fragment = sentence[split_tokens[i].i:split_tokens[i+1].i]
        # remove cc
        if len(fragment)>2 and fragment[0].dep_ == 'cc':
            fragment = fragment[1:]
        # remove punct
        if len(fragment)>2 and fragment[-1].dep_ == 'punct':
            fragment = fragment[:-1]
        fragments.append(fragment)

    return_sentences = []
    for i in fragments:
        root = None
        for j in roots_of_fragments:
            if j in i: root = j
        if not root:
            logging.warning("No root for fragment:"+i.string)
            continue
        root_children_dep = [x.dep_ for x in root.children]
        if 'nsubj' not in root_children_dep and 'nsubjpass' not in root_children_dep:
            return_sentences.append(nlp(fix_sentence(main_subject.text.strip()+" "+i.text.strip())))
        # TODO: if not using global coreference, we can do a simple coref here with elif
        else:
            return_sentences.append(nlp(fix_sentence(i.text.strip())))
    # for sentence in return_sentences:
    #     print(sentence)
    return return_sentences

def extract_main_sentence(sentence: spacy.tokens.doc.Doc) -> spacy.tokens.doc.Doc:
    # TODO: switch appros to make new sentences: \
    #  Chandragupta then defeated the invasion led by Seleucus I, a Macedonian general from Alexander's army.

    mask = mask_everything_but_clause(sentence)
    return_sentence = ""
    for i in sentence:
        if i.i not in mask:
            return_sentence+=i.string
    return_sentence = nlp(fix_sentence(return_sentence))
    if not is_junk_sentence(return_sentence,strict=False):
        return return_sentence
    return None
    pass

def mask_everything_but_clause(sentence: spacy.tokens.doc.Doc) -> set:
    # FIXIT: His name was Hellenized by later Greek historians as Sesostris, a name which was then given to a conflation of Senusret and several New Kingdom warrior pharaohs.
    clause_root_list = []
    return_str = ""
    mask = set()
    for i in sentence:
        if i.dep_ in ("relcl","advcl"):
            clause_root_list.append(i)
    for i in clause_root_list:
        t = extend_node(i)
        mask.update(range(t.start,t.end))
        if t.start !=0:
            if sentence[t.start-1].dep_=='punct':
                mask.add(t.start-1)
    for i in sentence:
        if i.i not in mask:
            return_str+=i.string
    return mask
    pass

def extract_parenthesis_from_article(article_str:str,with_coref=False,no_junk=True):
    # tokenize before extracting parentheses and remove it
    processed_sentence = []
    article_tokenized = sentence_tokenization(article_str, with_coref=with_coref, no_junk=no_junk)

    for sentence in article_tokenized:
        if check_parentheses_valid(sentence):
            processed_sentence.extend(extract_from_parentheses(sentence))
        else:  # avoid danger sentence with unmatching parentheses
            continue
    pass
    return processed_sentence

# -------------------------------------------------------
#                        Utilities
# -------------------------------------------------------

def batch_process(function,list_to_process)->list:
    ret = []
    for i in list_to_process:
        result = function(i)
        if isinstance(result,list):
            ret.extend(result)
        else:
            ret.append(result)
    return ret


def check_parentheses_valid(sentence) -> bool:
    # !! need sentence processed by a pipeline WITHOUT chunk merging
    # check if the parentheses in a sentence matches one another
    # assumes the parent have only one pair of parentheses, rejects those with 2
    left_parentheses = 0
    if isinstance(sentence,str):
        for i in sentence:
            if i.strip() == '(': left_parentheses += 1
            if i.strip() == ')': left_parentheses -= 1
            if left_parentheses < 0:
                return False
        pass
    else:
        for i in sentence:
            if i.string.strip() == '(': left_parentheses += 1
            if i.string.strip() == ')': left_parentheses -= 1
            if left_parentheses < 0:
                return False
    return True

def locate_parentheses(sentence:spacy.tokens.doc.Doc)->list:
    index = 0
    return_list = []

    def skip(index: int) -> int:
        # returns new index after skip
        start = index
        index += 1
        while index<len(sentence) and sentence[index].string.strip() != ')':
            if sentence[index].string.strip() == '(':
                index = skip(index)
            else:
                index += 1
        return_list.append(sentence[start+1:index])
        return index

    while index < len(sentence):
        if sentence[index].string.strip() == '(':
            index = skip(index)+1
        else:
            index += 1
    return return_list

def remove_parentheses(sentence: str,max_search_span = 200) -> str:
    index = 0
    backup_sentence = sentence
    sentence = sentence\
        .replace("（","(")\
        .replace("）",")")
    r=re.sub("\([\S\s].{0,%d}?\)"%max_search_span,"",sentence)
    return r
    # new_sentence = ""
    #
    # def skip(index: int) -> (bool,int):
    #     # returns new index after skip
    #     index += 1
    #     # Fixed:Richard B. Parkinson and Ludwig D. Morenz write that ancient Egyptian literature—narrowly defined as belles-lettres ("beautiful writing")—were not recorded in written form until the early Twelfth dynasty of the Middle Kingdom.
    #
    #     while sentence[index] != ')':
    #         if sentence[index] == '(':
    #             success,index = skip(index)
    #             if not success: return False,-1
    #         else:
    #             index += 1
    #             if index == len(sentence):
    #                 return (False, -1)
    #
    #     return True,index+1
    #
    # while index < len(sentence):
    #     if sentence[index] == '(':
    #         if index>0:new_sentence=new_sentence[:-1]
    #         success,index = skip(index)
    #         if not success:
    #             logging.error('Unable to pair parentheses:'+sentence)
    #             return backup_sentence
    #         if len(new_sentence) != 0 and new_sentence[-1] != ' ': new_sentence += " "
    #     else:
    #         new_sentence += sentence[index]
    #         index += 1
    #
    # return new_sentence

def fix_sentence(sentence:str)->str:
    sentence=sentence.strip('.')\
        .strip(",")\
        .strip(":")\
        .strip()\
        .replace("  "," ") \
        .replace(" ,", ',')\
        # .replace("\" ","\"")\
        # .replace("\' ","\'")


    if len(sentence)<5:return ""
    if 'a'<=sentence[0]<='z':
        sentence=sentence[0].upper()+sentence[1:]
    if 'a'<=sentence[-1]<='z' or 'A'<=sentence[-1]<='Z':
        sentence+='.'
    if sentence[-2] ==' ':
        sentence = sentence[:-2]+sentence[-1]

    if sentence[-1] in (',',':',"!"):
        sentence = sentence[:-1]+'.'
    else:
        # see if there is period
        t = len(sentence)-1
        while t>0:
            if sentence[t] == '.': break
            if 'a'<=sentence[t]<='z' or 'A'<=sentence[t]<='Z' or '0'<=sentence[t]<'9':
                sentence += '.'
                break
            else:
                t -= 1



    return sentence

def extend_node(node, without_dep=[], with_token=[]):
    i_min = node.i
    i_max = node.i
    to_be_processed = [node]
    while len(to_be_processed) != 0:
        t = to_be_processed.pop()
        if t.string.strip() not in with_token and t.dep_ in without_dep: continue
        # if t.string.strip() in without_token: continue
        to_be_processed.extend(t.children)
        if t.i < i_min:
            i_min = t.i
        elif t.i > i_max:
            i_max = t.i
    r=node.doc[i_min:i_max+1]
    return r

def find_main_subject(sentence,root = None):
    # nlp = spacy.load("en_core_web_sm")
    # doc = nlp("Dempsy plays for Seattle Sounders FC in Major League Soccer but he has served as the captain of the United Stares national team.")
    # doc = nlp("Despite of that, sad Dempsy Djhv plays for Seattle Sounders FC in Major League Soccer but he has served as the captain of the United Stares national team.")
    doc = sentence

    if not root:
        for i in sentence:
            if i.dep_ == "ROOT":
                root = i
                break

    # find potential nsubj
    potential_nsubj = []

    # Fix: when root is a noun
    if root.pos_ == 'NOUN':
        potential_nsubj.append(root)
    else:
        for i in root.children:
            if (i.dep_ == "nsubj" or i.dep_ == "nsubjpass"):
                potential_nsubj.append(i)

    # eliminate unlikely nsubj
    # TODO
    if len(potential_nsubj)>1:
        logging.error("multipal nsubj found:"+str(potential_nsubj)+" "+doc.text)

    # extend nsubj to get noun phrase
    potential_main_subject = []
    for i in potential_nsubj:
        t = extend_node(i, without_dep=['punct'],with_token=[")",'"',"'"])
        if len(t) == 0: continue
        potential_main_subject.append(t)

    # print(potential_main_subject)
    return potential_main_subject
    pass


# -------------------------------------------------------
#                    Testing Functions
# -------------------------------------------------------
def print_spacy_table(sentence):
    # from flair.models import SequenceTagger
    # from flair.data import Sentence
    # tagger = SequenceTagger.load('pos')
    # sentence = Sentence("Dempsy plays for Seattle Sounders FC in Major League Soccer and has served as the captain of the United Stares national team.")
    # tagger.predict(sentence)
    # pass

    from spacy.symbols import nsubj, VERB

    doc = nlp(sentence)
    # doc=nlp("In the Nashik inscription of Gautami Balashri, her son Gautamiputra Satakarni is called \"ekabamhana\", which is interpreted by some as \"unrivaled Brahmana\", thus indicating a Brahmin origin.")
    # doc = nlp("John is handsome (he is a gay).")
    # doc = nlp("However, Jefferson did not believe the Embargo Act, which restricted trade with Europe, would hurt the American economy.")
    # doc = nlp("The apple was eaten by John.")
    # doc = nlp("Dempsy plays for Seattle Sounders FC in Major League Soccer but he has served as the captain of the United Stares national team.")
    # doc = nlp("Despite of that, sad Dempsy Djhv plays for Seattle Sounders FC in Major League Soccer but he has served as the captain of the United Stares national team.")
    # doc = nlp("Despite of that, the computer plays for Seattle Sounders FC in Major League Soccer but he has served as the captain of the United Stares national team.")
    # doc = nlp("However the function of the female figurines in the life of Indus Valley people remains unclear.")
    # doc = nlp("The estimation for the antiquity of Bhirrana as pre-Harappan is based on two calculations of charcoal samples, giving two dates of respectively 7570-7180 BCE, and 6689-6201 BCE.")
    # doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
    # doc = nlp("Dsadf has also playd for Fasd Ldf, Nhel, and Nje.")
    # doc=nlp("Aridification of this region during the 3rd millennium BCE may have been the initial spur for the urbanisation associated with the civilisation, but eventually also reduced the water supply enough to cause the civilisation's demise, and to scatter its population eastward.")
    # doc= nlp("I sleep about 16 hours per day. I fall asleep accidentally. I'm tired, I have headache and I have problems with concentration. I don't eat anything and than I'm eating a lot.")
    # with open("data/Development_data/set2/a1.txt","r") as f:
    #     doc=nlp(f.read())

    from spacy import displacy
    # Finding a verb with a subject from below — good
    sentences = [sent.string.strip() for sent in doc.sents]
    print("TEXT	DEP	HEAD_TEXT	HEAD_POS	CHILDREN")
    for token in doc:
        # if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
        #     verbs.add(possible_subject.head)
    # print(verbs)
        print(token.text,"\t", token.dep_,"\t", token.head.text,"\t", token.head.pos_,"\t",
            [child for child in token.children])
    # displacy.serve(doc)


def test_passage():
    with open("../data/Development_data/set1/a1.txt","r") as f:
        doc=f.read()
    sent_tokenized = sentence_tokenization(doc,with_coref=False)
    for i in sent_tokenized:
        result = test_one(i)
        if type(result)==bool or (result is not None and len(result)!=0):
            print(result)
            print(i)
            print()

def test_one(i):
    return split_semicolon(i)
    pass

def test_pipeline():
    with open("../data/Development_data/set3/a9.txt","r") as f:
        doc=f.read()
    doc = "Sahure was in turn succeeded by Neferirkare Kakai who was Sahure's son. "
    for i in preprocess(doc,with_coref=True):
        print(i)

def test():
    doc = "MPEG-1 or MPEG-2 Audio Layer III, more commonly referred to as MP3, is an audio coding format for digital audio which uses a form of lossy data compression."
    # doc = 'It is a common audio format for consumer audio streaming or storage, and a de facto standard of digital audio compression for the transfer and playback of music on most digital audio players.'

    sentence = nlp(doc)

    # sentence = nlp("They also are able to control blood flow to their extremities, reducing the amount of blood that gets cold, but still keeping the extremities from freezing.")
    # sentence = nlp("Having thus acquired royal power, Sandracottos will possess India at the time Seleucos was preparing future glory.")

    # Having killed that guy, he will die.
    print(find_main_subject(sentence))
    # print(sentence_splitting_pipeline(nlp("John is handsome (he is a gay).")))

    # print(sentence_splitting(nlp("However, Jefferson did not believe the Embargo Act, which restricted trade with Europe, would hurt the American economy.")))
    # print(find_main_subject(nlp("The Indus Valley Civilisation has also been called by some the \"Sarasvati culture\"")))
    # find_main_subject("The estimation for the antiquity of Bhirrana as pre-Harappan is based on two calculations of charcoal samples, giving two dates of respectively 7570-7180 BCE, and 6689-6201 BCE.")
    # find_main_subject("However the function of the female figurines in the life of Indus Valley people remains unclear, and Possehl does not regard the evidence for Marshall's hypothesis to be \"terribly robust\".")
    # print(sentence_splitting('My mother\'s name is Sasha, she likes dogs.'))
    # print(sentence_tokenization("I love my father and my mother. They work hard. She is always nice but he is sometimes rude.",with_coref=True))
    # print(sentence_splitting(nlp("However, Jefferson did not believe the Embargo Act, which restricted trade with Europe, would hurt the American economy.")))


if __name__=='__main__':
    # TODO：如果你有直接运行的测试语句，请写在这里
    test_pipeline()
    doc = "MPEG-1 or MPEG-2 Audio Layer III, more commonly referred to as MP3, is an audio coding format for digital audio which uses a form of lossy data compression."
    # print_spacy_table(doc)
    # test()
    pass