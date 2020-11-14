import spacy

nlp = spacy.load("en_core_web_sm")

def wh_bonus(Questions, score, add_score):
    for i, question in enumerate(Questions):
        score_tmp = 0
        q_list = question.lower().split(' ')
        if 'why' in q_list:
            score_tmp += add_score
        elif 'what' in q_list:
            score_tmp += add_score
        elif 'where' in q_list:
            score_tmp += add_score
        elif 'which' in q_list:
            score_tmp += add_score
        elif 'who' in q_list:
            score_tmp += add_score
        elif 'when' in q_list:
            score_tmp += add_score
        elif 'how' in q_list:
            score_tmp += add_score
        score[i] += score_tmp
    return score


def length_bonus(Questions, score, upper_len, lower_len, add_score):
    for i, question in enumerate(Questions):
        q_list = question.split(' ')
        if len(q_list) > lower_len and len(q_list) < upper_len:
            score[i] += add_score
    return score


def pos_tagger(txt):
    # load spaCy's built-in English models
    # print(txt)

    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    # stop_words = set(stopwords.words('english'))

    # sent_tokenize is one of instances of
    # PunktSentenceTokenizer from the nltk.tokenize.punkt module

    parsed_text = nlp(txt)
    wordlist = []
    for token in parsed_text:
        tagged = (token.orth_, token.lemma_, token.tag_, token.ent_type_)
        wordlist.append(tagged)

    return wordlist


def punct_bonus(pos_return, score, upper_cont, add_score):
    for i, pos in enumerate(pos_return):
        pos_list = [tup[2] for tup in pos]
        #         print(pos_list)
        count_punc = [1 for word in pos_list if word == '.' or word == ',']
        if len(count_punc) < upper_cont:
            score[i] += add_score
    return score


def pronn_bonus(pos_return, score, lower_cont, add_score):
    for i, pos in enumerate(pos_return):
        pos_list = [tup[1] for tup in pos]
        count_punc = [1 for word in pos_list if word == '-PRON-']
        #         print(len(count_punc))
        if len(count_punc) < lower_cont:
            score[i] += add_score
    return score


def get_parse_tree(sentence):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    result = []
    for token in doc:
        result.append((token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children]))
    return result


# grammar bonus

def grammar_bonus(ptree_return, score, lower_cont, add_score):
    for i, dep in enumerate(ptree_return):
        dep_list = [tup[1] for tup in dep]
        count_bj = [1 for word in dep_list if word.endswith('bj')]
        #         print(len(count_punc))
        if len(count_bj) > lower_cont:
            score[i] += add_score
    return score


Questions = ['Does he like palying football?', 'What does he like playing?', 'Where is his hometown?',
             'how much is this pencil?', 'Who is he?', 'Does John like palying with other students?',
             'For your information, why does he do this?']


def get_score(Questions):
    # Wh bonus

    score = [0] * len(Questions)
    score = wh_bonus(Questions, score, 1)

    # len bonus
    score = length_bonus(Questions, score, 6, 3, 2)

    pos_return = []
    for question in Questions:
        pos_return.append(pos_tagger(question))

    # punctuations count bonus
    score = punct_bonus(pos_return, score, 2, 2)

    # pronoun bonus
    score = pronn_bonus(pos_return, score, 1, 2)

    ptree_return = []
    for question in Questions:
        ptree_return.append(get_parse_tree(question))

    score = grammar_bonus(ptree_return, score, 1, 1)
    return score