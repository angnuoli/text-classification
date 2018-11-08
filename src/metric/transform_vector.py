import numpy as np
import sys

class Transform:

    def __init__(self, vocab):
        self.vocab = vocab

    def vectorize_x(self, document):
        x = []
        for i in range(len(self.vocab)):
            #print(len(document.words_list))
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



