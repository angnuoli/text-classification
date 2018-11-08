import numpy as np
import string
import sys
import math

from sklearn.feature_extraction.text import TfidfVectorizer
class Transform:

    def __init__(self, vocab):
        self.vocab = vocab

    def vectorize_x(self, document):
        x = []
        for i in range(len(self.vocab)):
            #print(len(document.words_list))
            #for svm
            x.append(1 if (self.vocab[i]) in document.words_list else 0)

        return x

    def vectorize_y(self, document):
        return document.class_list[0]

    def get_feature(self, document_list):
        vector_x = []
        vector_y = []
        for i in range(len(document_list)):
            print(i)
            #if(i > 100):
                #print(vector_x)
                #print(vector_y)
                #sys.exit()
            vector_x.append(self.vectorize_x(document_list[i]))
            vector_y.append(self.vectorize_y(document_list[i]))
        return vector_x, vector_y

    def get_feature_tfidf(self, document_list):
        x = []
        y = []
        for i in (range(len(document_list))):
            print(i)
            x.append(self.cal_dfidf(document_list[i], document_list))
            y.append(self.vectorize_y(document_list[i]))
            #print(x)
        return x, y

    def cal_dfidf(self, document, document_list):
        x =[]
        for i in range(len(self.vocab)):
            if (self.vocab[i]) in document.words_list:
                tf = 0
                idf = 0
                for j in range(len(document.words_list)):
                    tf = (tf + 1) if self.vocab[i] == document.words_list[j] else tf
                for k in range(len(document_list)):
                    idf = (idf + 1) if self.vocab[i] in document_list[k].words_list else idf
                idf = math.log(len(document_list) / idf)
                x.append(tf * idf)
            else:
                x.append(0)
        return x






