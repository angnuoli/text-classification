import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from src.metric.metric import calculate_tf_idf
class Transform:

    def __init__(self, vocab):
        self.vocab = vocab
        self.map = {}
        for i in range(len(vocab)):
            self.map[vocab[i]] = i

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
            #print(i)
            #if(i > 100):
                #print(vector_x)
                #print(vector_y)
                #sys.exit()
            vector_x.append(self.vectorize_x(document_list[i]))
            vector_y.append(self.vectorize_y(document_list[i]))
        return vector_x, vector_y

    def get_feature_tfidf(self, document_list, df):
        x = []
        self.df = df
        y = []

        for document in document_list:
            #print(i)
            x.append(self.cal_dfidf(document, document_list))
            y.append(self.vectorize_y(document))
            #print(x)
        return x, y

    def cal_dfidf(self, document, document_list):
        x = np.zeros(len(self.vocab))
        for term in set(document.words_list):
            if term in self.map.keys():
                j = self.map[term]
                x[j] = calculate_tf_idf(document.tf[term], self.df[term], len(document_list))

        return x.tolist()