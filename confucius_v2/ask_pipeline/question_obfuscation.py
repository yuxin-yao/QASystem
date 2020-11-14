#!/opt/conda/bin/python
# -*- coding: utf-8 -*-

# TODO: possible obfuscation:
#  where    ->  what location
#  when     ->  what is the time for


#  why      ->  what is the reason for
#  which    ->  what
#  what noun->  How         # What is the speed -> how fast is
#  ANY      ->  Can you explain ANY / Would you please ANY
#  ANY      ->  Please explain XXX.
#  ANY+ANY  ->  Given how/what/why ANY, ANY     # this will trick some of the rule-based classifier

import spacy
nlp = spacy.load('en_core_web_sm')  # load spaCy's built-in English models
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize

def get_parse_tree(sentence):

    doc = nlp(sentence)
    result = []
    for token in doc:
        result.append((token.text, token.dep_, token.head.text, token.head.pos_,[child for child in token.children]))
    return result

def pos_tagger(txt):
    parsed_text = nlp(txt)
    wordlist = []
    for token in parsed_text:
        # display the token's orthographic representation, lemma, part of speech, and entity type (which is empty if the token is not part of a named entity)
        tagged = token.orth_, token.lemma_, token.tag_, token.ent_type_
        wordlist.append(tagged)

    return wordlist
def please_explain_obf(Questions):
    result = []

    for que in Questions:

        que_list = que.split(" ")
        sent_spacy = nlp(que)
        ent_list = [word.ent_type_ for word in sent_spacy]
        que_list_change = []
        for i, word in enumerate(que_list):
            if ent_list[i] == 'ORG' or ent_list[i] == 'GPE' or ent_list[i] == 'PERSON':
                que_list_change.append(word)
            else:
                que_list_change.append(word.lower())
        que_list = que_list_change
        if 'why' in que_list:
            que_list[que_list.index('why')] = 'Please explain why'
            result.append(' '.join(que_list))
        elif 'what' in que_list:
            que_list[que_list.index('what')] = 'Please explain what'
            result.append(' '.join(que_list))
    return result


def reason_obf(Questions):
    result = []
    for que in Questions:
        pos_result = pos_tagger(que)
        parse_tree = [tag[1] for tag in get_parse_tree(que)]
        que_list = que.split(" ")
        sent_spacy = nlp(que)
        ent_list = [word.ent_type_ for word in sent_spacy]
        que_list_change = []
        for i, word in enumerate(que_list):
            if ent_list[i] == 'ORG' or ent_list[i] == 'GPE' or ent_list[i] == 'PERSON':
                que_list_change.append(word)
            else:
                que_list_change.append(word.lower())
        que_list = que_list_change

        if 'why' in que_list:
            root_idx = parse_tree.index('ROOT')
            why_idx = que_list.index('why')
            if pos_result[why_idx + 1][1] == 'do':
                #             if que_list[why_idx+1] == 'were':
                que_list[why_idx] = 'What is the reason for'
                #             que_list[root_idx] = que_list[why_idx+1]+" "+que_list[root_idx]

                que_list[root_idx] = que_list[root_idx] + 'ing'
                que_list[why_idx + 1] = ''
                result.append(' '.join(que_list))
            elif pos_result[why_idx + 1][1] == 'be':

                sub_idx = parse_tree.index('nsubj')
                que_list[sub_idx + 1] = 'being ' + que_list[sub_idx + 1]
                que_list[why_idx + 1] = ''
                que_list[why_idx] = 'What is the reason for'
                result.append(' '.join(que_list))

    return result



#
# def which_obf(Questions):
#     result = []
#
#     for que in Questions:
#         pos_result = pos_tagger(que)
#
#         que_list = que.split(" ")
#         sent_spacy = nlp(que)
#         ent_list = [word.ent_type_ for word in sent_spacy]
#         que_list_change = []
#         for i, word in enumerate(que_list):
#             if ent_list[i] == 'ORG' or ent_list[i] == 'GPE' or ent_list[i] == 'PERSON':
#                 que_list_change.append(word)
#             else:
#                 que_list_change.append(word.lower())
#         que_list = que_list_change
#         if 'what' in que_list:
#             what_idx = que_list.index('what')
#             #             print(pos_result[what_idx+1])
#             if pos_result[what_idx + 1][-2] == 'NN' and que_list[what_idx + 1] != 'time':
#                 que_list[what_idx] = 'Which'
#                 result.append(' '.join(que_list))
#
#     return result

def according_obf(Questions):
    import random

    result = []
    number_ques = 0
    Questions_shuffle = Questions.copy()

    random.shuffle(Questions_shuffle)


    for que in Questions_shuffle:
        que_list = que.split(" ")
        sent_spacy = nlp(que)
        ent_list = [word.ent_type_ for word in sent_spacy]
        que_list_change = []
        for i, word in enumerate(que_list):
            if ent_list[i] == 'ORG' or ent_list[i] == 'GPE' or ent_list[i] == 'PERSON':
                que_list_change.append(word)
            else:
                que_list_change.append(word.lower())

        result.append('According to the article, ' + ' '.join(que_list_change))
        number_ques += 1
        if number_ques > 50:
            break


    return result

if __name__ == '__main__':
    Questions = ['Why does he like playing football?','What is his son named after?','What country does he come from?']

    result = Questions + according_obf(Questions)
    result = Questions + reason_obf(Questions)
    # result = result + which_obf(Questions)
    result = result + please_explain_obf(Questions)

