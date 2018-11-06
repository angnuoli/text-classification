import numpy as np
from src.data_structure.data_structure import Document


def calculate_tf_idf(tf, df, doc_num):
    """
    :param doc_num: the number of all documents
    :param tf: term frequency
    :param df: document frequency where term appears
    :return: td-idf importance
    """
    idf = np.log(float(doc_num + 1) / (df + 1))
    tf = 1.0 + np.log(tf)
    return tf * idf


def calculate_priority_by_tfidf(documents: []) -> {}:
    term_df = {}
    term_tf = {}
    tf = {}

    for document in documents:
        document: Document
        terms = set(document.words_list)
        for term in terms:
            term_df[term] = term_df.get(term, 0) + 1

        for class_ in document.class_list:
            if class_ not in term_tf.keys():
                term_tf[class_] = {}

            for term in document.words_list:
                term_tf[class_][term] = term_tf[class_].get(term, 0) + 1

    for class_ in term_tf.keys():
        for term, v in term_tf[class_].items():
            tf[term] = max(tf.get(term, 0), v)

    term_importance_pair = {}
    for term in term_df.keys():
        term_importance_pair[term] = calculate_tf_idf(tf[term], term_df[term], len(documents))

    return term_importance_pair
