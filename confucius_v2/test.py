# one sentence-->two sentence(decompose "cc": and, or...)

import string

import spacy
def pos_tagger(txt):
    nlp = spacy.load('en_core_web_sm')  # load spaCy's built-in English models
    #print(txt)

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    #stop_words = set(stopwords.words('english'))


        # sent_tokenize is one of instances of
        # PunktSentenceTokenizer from the nltk.tokenize.punkt module
    sentlist = []

    tokenized = sent_tokenize(txt)
    for i in tokenized:
        parsed_text = nlp(i)
        wordlist = []
        for token in parsed_text:


        # display the token's orthographic representation, lemma, part of speech, and entity type (which is empty if the token is not part of a named entity)
            tagged = (token, token.lemma_, token.pos_, token.ent_type_)
            wordlist.append(tagged)
        sentlist.append(wordlist)
    return sentlist


#parse tree from spacy
import spacy

def get_parse_tree(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    result = []
    for token in doc:
        result.append((token.text, token.dep_, token.head.text, token.head.pos_,[child for child in token.children]))
    return result

#get_parse_tree("Gyarados (ギャラドス, Gyaradosu,  or ) is a Pokémon species in Nintendo and Game Freak's Pokémon franchise.")

# step 1
# input: raw sentence; output: simplified sentence
def sentence_simplifier(sent):
    result = sent
    main_sent_start = 0
    apposite = []
    braket = []
    clause = []
    # Removing Discourse Markers and Adjunct Modifiers: (), ,...,
    sent_pos_spacy = pos_tagger(result)[0]

    sent_parse = get_parse_tree(sent)
    for i in range(len(sent_parse)):
        if sent_parse[i][1] == "ROOT":
            root = sent_parse[i][0]
            root_index = i
        if sent_parse[i][1] == "nsubj" or sent_parse[i][1] == "nsubjpass":
            sub = sent_parse[i][0]
            sub_index = i

    # remove同位语
    for i in range(sub_index, root_index):
        if sent_pos_spacy[i][1] == ',':
            for j in range(i + 1, root_index):
                if sent_pos_spacy[j][1] == ',':
                    apposite.append((i, j))
                    break

    # print(apposite)
    if len(apposite) != 0:
        result = ""
        i = 0
        for pair in apposite:
            for j in range(i, pair[0]):
                result += " "
                result += str(sent_pos_spacy[j][0])
            i = pair[1] + 1

        if i < len(sent_pos_spacy) - 1:
            for k in range(i, len(sent_pos_spacy)):
                result += " "
                result += str(sent_pos_spacy[k][0])

    sent_pos_spacy = pos_tagger(result)[0]

    # remove 前面的从句: (when ...,) I get up
    for i in range(len(sent_pos_spacy)):
        if sent_pos_spacy[i][1] == ',' and i < root_index:
            main_sent_start = i + 1
            break
    if main_sent_start != 0:
        result = ""
        for i in range(len(sent_pos_spacy)):
            if i >= main_sent_start:
                result += " "
                result += str(sent_pos_spacy[i][0])

    sent_pos_spacy = pos_tagger(result)[0]

    # remove braket
    for i in range(len(sent_pos_spacy)):
        if sent_pos_spacy[i][1] == '(':
            for j in range(i + 1, len(sent_pos_spacy)):
                if sent_pos_spacy[j][1] == ')':
                    braket.append((i, j))
                    break

    if len(braket) != 0:
        result = ""
        i = 0
        for pair in braket:
            for j in range(i, pair[0]):
                result += " "
                result += str(sent_pos_spacy[j][0])
            i = pair[1] + 1

        if i < len(sent_pos_spacy) - 1:
            for k in range(i, len(sent_pos_spacy)):
                result += " "
                result += str(sent_pos_spacy[k][0])

    sent_pos_spacy = pos_tagger(result)[0]

    # remove clause and ", verbing..."
    # print(result)
    sent_pos_nltk = postagger(result)[0][0]
    for i in range(len(sent_pos_nltk) - 1):
        if (sent_pos_nltk[i][0] == ',') and (
                sent_pos_nltk[i + 1][1] == "VBG" or sent_pos_nltk[i + 1][0] == 'who' or sent_pos_nltk[i + 1][
            0] == 'where' or sent_pos_nltk[i + 1][0] == 'when' or sent_pos_nltk[i + 1][0] == 'which'):
            if i < len(sent_pos_nltk) - 1:
                for j in range(i + 1, len(sent_pos_nltk)):
                    if sent_pos_nltk[j][0] == ",":
                        clause.append((i, j))
                        break

                    else:
                        if (i, len(sent_pos_nltk) - 2) not in clause:
                            clause.append(
                                (i, len(sent_pos_nltk) - 2))  # clause is at the end of the sentence(we need biaodian)

    # print(clause)
    if len(clause) != 0:
        result = ""
        i = 0
        for pair in clause:
            for j in range(i, pair[0]):
                result += " "
                result += str(sent_pos_nltk[j][0])
            i = pair[1] + 1

        if i < len(sent_pos_nltk):
            for k in range(i, len(sent_pos_nltk)):
                result += " "
                result += str(sent_pos_nltk[k][0])

    return result.lstrip()