import os
from sys import argv
import types
import pickle

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from src.data_preprocess.preprocess import DataProcessor, Vectorizer
from src.metric.metric import calculate_priority_by_tfidf, calculate_priority_by_tf, calculate_df, cal_predict_accuracy
from src.metric.transform_vector import Transform
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


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

    # save model
    save_model(clf, 'vocabulary-{}-{}'.format(len(vocabulary), datetime.datetime.now()))

    # test
    y_pred = clf.predict(test_x)
    acc = cal_predict_accuracy(y_pred, test_documents)
    print(acc)
    return acc


def save_model(model, filename):
    joblib.dump(model, 'running_result/{}.joblib'.format(filename))


def run_svm_classification(calculate_priority):
    data_processor = DataProcessor()
    train_documents, test_documents = data_processor.get_train_and_test_documents()

    print("\n========== Constructing bag of words ==========")
    vectorizer = Vectorizer()
    vocabulary = vectorizer.generate_bag_of_words(raw_documents=train_documents,
                                                   calculate_priority=calculate_priority)

    # prepare document term frequency
    train_df = calculate_df(train_documents)

    # cross_validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}, {'kernel': ['linear'], 'C': [1, 10, 100]}]

# =============================================================================
#  Pipeline:
#
#     t = Transform(vocabulary[0:10000])
#
#     train_x, train_y = t.get_feature_tfidf_train(train_documents, train_df)
#     test_x = t.get_feature_tfidf_test(test_documents, train_df)
#
#     clf = svm.SVC()
#     clf.fit(train_x, train_y)
#     save_model(clf, filename='vocabulary-{}'.format(10000))
#
#     acc = cal_predict_accuracy(clf, test_x, test_documents)
#     print(acc)
# =============================================================================

    vocabulary_size = []
    accuracy = []

    b = np.log10(len(vocabulary)-100) / np.log10(300)
    base = np.log10(300)

    i = 1
    while i <= b:
        vocabulary_size.append(int(10 ** (i * base)))

        print('vocabulary_size: {}'.format(vocabulary_size[-1]))

        acc = svm_predict(train_documents, test_documents, vocabulary[0:vocabulary_size[-1]], train_df, tuned_parameters)

        accuracy.append(acc)

        i += (b-1)/10

    plt.plot(vocabulary_size, accuracy, marker='x')
    plt.xlabel('vocabulary_size')
    plt.ylabel('accuracy')
    plt.title('classification accuracy with respect to different vocabulary_size')
    plt.savefig('running_result/cur-fig{}.svg'.format(datetime.datetime.now()))
    with open('running_result/file{}.txt'.format(datetime.datetime.now()), 'w') as f:
        f.writelines('{}'.format(vocabulary_size))
        f.writelines('{}'.format(accuracy))