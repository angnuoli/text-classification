import numpy as np


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
    min_count = 4

    for document in documents:
        # document: Document
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
            if v >= min_count:
                tf[term] = tf.get(term, 0) + v

    term_importance_pair = {}
    _n = len(documents)
    for term in tf.keys():
        term_importance_pair[term] = calculate_tf_idf(tf[term], term_df[term], _n)

    return term_importance_pair

def calculate_priority_by_tf(documents: []) -> {}:
    term_tf = {}

    for document in documents:
        # document: Document
        for term in document.words_list:
            if term not in term_tf.keys():
                term_tf[term] = {}

            for class_ in document.class_list:
                term_tf[term][class_] = term_tf[term].get(term, 0) + 1

    term_importance_pair = {}
    for term in term_tf.keys():
        term_importance_pair[term] = max(term_tf[term].values())

    return term_importance_pair

def calculate_priority_by_chi_square(documents: []) -> {}:
    """Calculate chi square metric to measure the importance of a term to a class.

    :return:
    """
    term_importance_pair = {}
    chi_2_term_class = {}
    df_term_class = {}
    df_term = {}
    df_of_classes = {}
    n_train_documents = len(documents)
    classes = set()

    for document in documents:
        terms = set(document.words_list)
        for term in terms:
            df_term[term] = df_term.get(term, 0) + 1
            if term not in df_term_class.keys():
                df_term_class[term] = {}

            for label in document.class_list:
                df_term_class[term][label] = df_term_class[term].get(label, 0) + 1
                classes.add(label)

        for label in document.class_list:
            df_of_classes[label] = df_of_classes.get(label, 0) + 1

    for term in df_term_class.keys():
        chi_2_term_class[term] = chi_2_term_class.get(term, {})
        for label in df_term_class[term].keys():
            A = df_term_class[term][label]

            if A != 0:
                B = df_term[term] - A
                C = df_of_classes[label] - A
                D = n_train_documents - A - B - C
                N = A + B + C + D

                chi_2_term_class[term][label] = (float(N) * (A * D - C * B) * (A * D - C * B)) \
                                                 / ((A + C) * (B + D) * (A + B) * (C + D))
            else:
                chi_2_term_class[term][label] = 0.0

            term_importance_pair[term] = max(term_importance_pair.get(term, 0), chi_2_term_class[term][label])

    return term_importance_pair


def cal_predict_accuracy(y_pred, test_documents):
    # test
    sum_acc = 0
    for j, pre in enumerate(y_pred):
        if pre in test_documents[j].class_list:
           sum_acc += 1

    print(sum_acc / len(y_pred))
    return sum_acc / len(y_pred)


def calculate_df(document_list):
    df = {}

    for document in document_list:
        for term in document.tf.keys():
            df[term] = df.get(term, 0) + 1

    return df