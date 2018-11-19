import numpy as np

from src.metric.metric import calculate_tf_idf
class Transform:

    def __init__(self, vocab):
        self.vocab = vocab
        self.map = {}
        for i, word in enumerate(vocab):
            self.map[word] = i

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

    def get_feature_tfidf_train(self, document_list, df):
        x, y = [], []
        n = len(document_list)
        count_label = {}

        for document in document_list:
            tmp_x = self.cal_tfidf(document, df, n)
            # partition multiple labels in multiple single-label training element
            for class_label in document.class_list:
                x.append(tmp_x)
                y.append(class_label)
                count_label[class_label] = count_label.get(class_label, 0) + 1

        tx, ty = [], []
        for i, class_label in enumerate(y):
            if count_label[class_label] >= 5:
                tx.append(x[i])
                ty.append(y[i])

        return tx, ty

    def get_feature_tfidf_test(self, document_list, df):
        x = []
        n = len(document_list)

        for document in document_list:
            tmp_x = self.cal_tfidf(document, df, n)
            # partition multiple labels in multiple single-label training element
            x.append(tmp_x)

        return x

    def cal_tfidf(self, document, df, n):
        x = np.zeros(len(self.vocab))
        for term in document.tf.keys():
            if term in self.map.keys():
                j = self.map[term]
                x[j] = calculate_tf_idf(document.tf[term], df[term], n)

        return x.tolist()
