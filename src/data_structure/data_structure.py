class Document:
    """A document instance for further processing.

    The structure of class document:
        @dict['words'] is a dictionary.
        @dict['words']['title'] is a list which contains words of title.
        @dict['words']['body'] is a list which contains words of article body.
        @dict['topics'] is a list of TOPICS class labels.
        @dict['places'] is a list of PLACES class labels.
    """

    def __init__(self):
        self.title = ""
        self.text = ""
        self.words_list = []
        self.class_list = []


class StaticData:
    """ Static Count """
    # set of classes, class -> document frequency, initialized in data_preprocess
    bag_of_classes = set()

    # number of train documents, initialized in data_preprocess
    n_train_documents = 0
    n_classes = 0.0
