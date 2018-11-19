import os
from sys import argv
import types
import pickle

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from src.data_preprocess.preprocess import DataProcessor, Vectorizer
from src.metric.metric import calculate_priority_by_tfidf, calculate_priority_by_tf
from src.metric.transform_vector import Transform
import numpy as np
import matplotlib.pyplot as plt
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


def calculate_df(document_list):
    df = {}

    for document in document_list:
        for term in document.tf.keys():
            df[term] = df.get(term, 0) + 1

    return df

def svm_predict(train_documents, test_documents, vocabulary, train_df, tuned_parameters):
    t = Transform(vocabulary)

    train_x, train_y = t.get_feature_tfidf_train(train_documents, train_df)
    test_x = t.get_feature_tfidf_test(test_documents, train_df)

    # train svm via cross validation
    A = time.time()
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='precision_macro')
    clf.fit(train_x, train_y)
    print(time.time() - A)
    print("Best parameters set found on development set:")
    print(clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("{:.3f} (+/-{:.3f}) for {}".format(mean, std * 2, params))

    # test
    predict_y = clf.predict(test_x)

    sum_acc = 0
    for j, pre in enumerate(predict_y):
        if pre in test_documents[j].class_list:
           sum_acc += 1

    print(sum_acc / len(predict_y))
    return sum_acc / len(predict_y)


def run(filename, calculate_priority):
    train_documents, test_documents, vocabulary = data_preprocess(calculate_priority=calculate_priority_by_tfidf,
                                                    vocabulary_size=-1)

    # prepare document term frequency
    train_df = calculate_df(train_documents)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    accuracy = svm_predict(train_documents, test_documents, vocabulary, train_df, tuned_parameters)

#    vocabulary_size = []
#    accuracy = []
#
#    b = np.log10(len(vocabulary)-100) / np.log10(300)
#    base = np.log10(300)
#
#    i = 1
#    while i <= b:
#        vocabulary_size.append(int(10 ** (i * base)))
#
#        print('vocabulary_size: {}'.format(vocabulary_size[-1]))
#
#        accuracy.append(sum_acc / len(predict_y))
#
#        i += (b-1)/10
#
#
#    plt.plot(vocabulary_size, accuracy, marker='x')
#    plt.xlabel('vocabulary_size')
#    plt.ylabel('accuracy')
#    plt.title('classification accuracy with respect to different vocabulary_size')
#    plt.savefig('cur-fig{}.svg'.format(filename))
    with open('file{}.txt'.format(filename), 'w') as f:
        f.writelines('{}'.format(accuracy))

if __name__ == "__main__":

    #run(2, calculate_priority_by_tf)
    run(3, calculate_priority_by_tfidf)
