import os
from sys import argv
import types

from src.data_preprocess.preprocess import DataProcessor, Vectorizer
from src.metric.metric import calculate_priority_by_tfidf


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
    data_dir = os.path.abspath(data_dir)
    data_processor = DataProcessor()
    train_documents, test_documents = data_processor.data_preprocess(data_dir)

    # test_documents = test_documents[0:10]
    # binarize the class label to class vectors
    print("\n========== Constructing bag of words ==========")
    vectorizer = Vectorizer(max_df=0.9)
    vocabulary = vectorizer.generate_bag_of_words(raw_documents=train_documents,
                                                   calculate_priority=calculate_priority,
                                                   vocabulary_size=vocabulary_size)

    return train_documents, test_documents, vocabulary


if __name__ == "__main__":
    train_documents, test_documents, vocabulary = data_preprocess(calculate_priority=calculate_priority_by_tfidf,
                                                    vocabulary_size=-1)
