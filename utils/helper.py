import sys
import itertools
import numpy as np
import pandas as pd
from pymagnitude import Magnitude
import spacy
from spacy.matcher import Matcher
from gensim.models import KeyedVectors

nlp = spacy.load('en_core_web_sm')
pattern = [{'LIKE_NUM': True}, {'IS_ALPHA': True}]
matcher = Matcher(nlp.vocab)
matcher.add('num_bigram', None, pattern)


def num_bigram(doc):
    """
    function to merge tokens for {NUM} followed by {TEXT}
    :param doc: spacy doc
    :return: modified spacy doc
    """
    matched_spans = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_spans.append(span)
    for span in matched_spans:  # merge into one token after collecting all matches
        span.merge()
    return doc


nlp.add_pipe(num_bigram, first=True)


def process_text(text):
    """
    Function to return word tokens after removing stopwords
    :param text: text
    :return: word tokens
    """
    return [token.text for token in nlp(text) if not token.is_stop]


def init_model(model_type):
    """
    Function to initialize the pre-trained word embedding model
    :return: model
    """
    if model_type == 'magnitude':
        model = Magnitude('../model/crawl-300d-2M.magnitude')
    elif model_type == 'gensim':
        model = KeyedVectors.load('../model/pre_trained_word2vec_embeddings.bin')
    else:
        print("Invalid model type.")
        sys.exit(1)
    return model, model_type


def text2vec(doc_tok, model, dim=300):
    """
    Function to convert documents to vectors
    :param doc_tok: tokenized document
    :param model: word embedding model
    :param dim: dimension of embedding (default=300)
    :return: embedding vectors
    """
    doc_embedding = np.zeros(dim)
    valid_words = 0
    for word in doc_tok:
        if word in model:
            valid_words += 1
            doc_embedding += model.query(word)
        else:
            continue
    if valid_words > 0:
        return doc_embedding / valid_words
    else:
        return doc_embedding


def get_relevant_words(search_tok, doc_tok, model, model_type):
    """
    Function to reutrn semantically relevant words
    :param search_tok: search tokens
    :param doc_tok: document tokens
    :param model: word embedding model
    :return: relevant words
    """
    search_set = set()
    doc_set = set()
    word_array = set()
    for word in search_tok:
        if word in model:
            search_set.add(word)

    for word in doc_tok:
        if word in model:
            doc_set.add(word)

    for s in itertools.product(search_set, doc_set):
        if model_type == 'magnitude':
            if model.similarity(s[0], s[1]) >= 0.3:
                word_array.add(s[1])
        else:
            if model.wv.similarity(s[0], s[1]) >= 0.3:
                word_array.add(s[1])
    return ', '.join(list(word_array))


def get_docs_embedding(docs_tok, model, dim=300):
    """
    Function to generate document embedding
    :param docs_tok: documents' tokens
    :param model: word embedding model
    :param dim: dimension of embedding (default=300)
    :return: documents' embedding
    """
    all_docs_embedding = []
    for doc in docs_tok:
        all_docs_embedding.append(text2vec(doc, model, dim))
    cols = [str(i) for i in range(dim)]
    embeddings = pd.DataFrame(data=all_docs_embedding)
    embeddings.columns = cols
    embeddings.to_parquet('../model/docs_embeddings.parquet', index=False)
    return np.array(all_docs_embedding)


