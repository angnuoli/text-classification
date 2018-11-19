import os
from sys import argv
import types
import pickle

from sklearn import svm
from sklearn.metrics import accuracy_score
from src.data_preprocess.preprocess import DataProcessor, Vectorizer
from src.metric.metric import calculate_priority_by_tfidf, calculate_priority_by_tf
from src.metric.transform_vector import *
import numpy as np
import matplotlib.pyplot as plt
import pylab
import time

def data_preprocess(calculate_priority: types.FunctionType, vocabulary_size):
    """
        Params
        ------------
        calculate_priority: function type, (document) -> {term: importance}
        vocabulary_size: int, limit size of words set

        Return
        ------------
        train_documents: list of Document object
        test_documents: list of Document object
        vocabulary: list of string, sorted by importance
    """

    if len(argv) > 1:
        data_dir = argv[1]
    else:
        data_dir = 'data'

    if not os.path.exists(data_dir):
        raise OSError('Please store original data files in data/ directory or type '
                      '"python3 main.py data_path" to input path of data')

    # =============================================================================
    #   Preprocessing
    # =============================================================================
    print("========== Parse data files ==========")
    if os.path.isfile('data/train.pkl'):
        with open('data/train.pkl', 'rb') as f:
            train_documents = pickle.load(f)
        with open('data/test.pkl', 'rb') as f:
            test_documents = pickle.load(f)
    else:
        data_dir = os.path.abspath(data_dir)
        data_processor = DataProcessor()
        train_documents, test_documents = data_processor.data_preprocess(data_dir)
        with open('data/train.pkl', 'wb') as f:
            pickle.dump(train_documents, f)
        with open('data/test.pkl', 'wb') as f:
            pickle.dump(test_documents, f)


    # test_documents = test_documents[0:10]
    # binarize the class label to class vectors
    print("\n========== Constructing bag of words ==========")
    vectorizer = Vectorizer(max_df=0.9)
    vocabulary = vectorizer.generate_bag_of_words(raw_documents=train_documents,
                                                   calculate_priority=calculate_priority,
                                                   vocabulary_size=vocabulary_size)

    return train_documents, test_documents, vocabulary


def calculate_tf_df(document_list):
    df = {}

    for document in document_list:
        for term in set(document.words_list):
            df[term] = df.get(term, 0) + 1

        for term in document.words_list:
            document.tf[term] = document.tf.get(term, 0) + 1

    return df

def run(filename, calculate_priority):
    train_documents, test_documents, vocabulary = data_preprocess(calculate_priority=calculate_priority_by_tfidf,
                                                    vocabulary_size=-1)

    #print(len(train_documents))
    vocabulary_size = []
    accuracy = []

    train_df = calculate_tf_df(train_documents)
    test_df = calculate_tf_df(test_documents)

    b = np.log10(len(vocabulary)-100) / np.log10(300)
    base = np.log10(300)

    i = 1
    model = svm.SVC()
    while i <= b:
        vocabulary_size.append(int(10 ** (i * base)))

        print('vocabulary_size: {}'.format(vocabulary_size[-1]))
        t = Transform(vocabulary[0:vocabulary_size[-1]])
        train_x, train_y = t.get_feature_tfidf(train_documents, train_df)
        #print(train_x)
        test_x, test_y = t.get_feature_tfidf(test_documents, test_df)
        #svm
        A = time.time()
        model.fit(train_x, train_y)
        print(time.time() - A)
        predict_y = model.predict(test_x)
        #accuracy
        sum_acc = 0
        for j in range(len(predict_y)):
            if predict_y[j] in test_documents[j].class_list:
               sum_acc += 1

        print(sum_acc / len(predict_y))
        accuracy.append(sum_acc / len(predict_y))

        i += (b-1)/10


    plt.plot(vocabulary_size[6:], accuracy[2:], marker='x')
    plt.xlabel('vocabulary_size')
    plt.ylabel('accuracy')
    plt.title('classification accuracy with respect to different vocabulary_size')
    plt.savefig('fig{}.svg'.format(filename))

if __name__ == "__main__":

    run(2, calculate_priority_by_tf)
    run(3, calculate_priority_by_tfidf)
