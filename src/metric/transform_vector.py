class Transform:

    def __init__(self, vocab):
        self.vocab = vocab

    def vectorize_x(document):
        x = []
        for i in range(len(self.vocab)):
            x.append(1 if (document.words_list[i]) in self.vocab else 0)
        return x

    def vectorize_y(document):
        return document.class_list[0]

    def get_feature(self, document_list):
        vector_x = []
        vector_y = []
        for i in range(len(document_list)):
            vector_x.append(vectorize_x(document_list[i]))
            vector_u.append(vectorize_y(document_list[i]))
        return vector_x, vector_y



