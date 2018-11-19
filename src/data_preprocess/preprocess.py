#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gjxhlan
"""
import numbers
import os
import re
import string

from src.data_structure.data_structure import StaticData, Document
from src.metric.metric import calculate_priority_by_tfidf
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

class Vectorizer:
    """Convert a collection of text documents to a matrix of token counts.

    Parameters
    ----------
    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None, default=None
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.
    """

    def __init__(self, max_df=0.9, min_df=3, max_features=None):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features


    def generate_bag_of_words(self,
            raw_documents: [],
            calculate_priority = calculate_priority_by_tfidf,
            vocabulary_size = -1) -> []:
        """Learn the vocabulary dictionary and return vocabulary set.

        Parameters
        ----------
        raw_documents : list of Document object

        calculate_priority : self define function to calculate term - importance pair.
            it must follow the formula

            (raw_documents) -> {term: importance}

            default use document frequency as importance metric.

        vocabulary_size: top n important words.

        Returns
        -------
        list of words sorted by importance.
        """
        max_df = self.max_df
        min_df = self.min_df

        n_doc = len(raw_documents)
        max_doc_count = (max_df
                         if isinstance(max_df, numbers.Integral)
                         else max_df * n_doc)
        min_doc_count = (min_df
                         if isinstance(min_df, numbers.Integral)
                         else min_df * n_doc)
        assert min_doc_count < max_doc_count

        print("\n========== Feature selection ==========")
        term_importance_pair = calculate_priority(raw_documents)
        temp = []
        for term, value in term_importance_pair.items():
            temp.append([term, value])

        print("Sort terms by importance...")
        temp = sorted(temp, key=lambda pair: pair[1], reverse=True)
        vocabulary = [item[0] for item in temp]

        if len(vocabulary) == 0:
            print("Warning! After pruning, no terms remain. Try a lower min_df or a higher max_df.")

        if vocabulary_size >= 0 and vocabulary_size < len(vocabulary):
            return vocabulary[0:vocabulary_size]

        return vocabulary


class DataProcessor:
    """ Class DataProcessor for data pre-processing.

    data_preprocess(): pre-process the raw data into feature vectors
    """

    remove_table = str.maketrans('', '', string.digits + string.punctuation)
    stop_words_ = frozenset(["a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually",
                             "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost",
                             "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an",
                             "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways",
                             "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around",
                             "as",
                             "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be",
                             "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
                             "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between",
                             "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't",
                             "cannot",
                             "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
                             "come",
                             "comes", "concerning", "consequently", "consider", "considering", "contain", "containing",
                             "contains", "corresponding", "could", "couldn't", "course", "currently", "d",
                             "definitely",
                             "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing",
                             "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight",
                             "either",
                             "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever",
                             "every",
                             "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except",
                             "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for",
                             "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get",
                             "gets",
                             "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings",
                             "h",
                             "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he",
                             "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby",
                             "herein",
                             "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully",
                             "how",
                             "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored",
                             "immediate",
                             "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner",
                             "insofar",
                             "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
                             "itself",
                             "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last",
                             "lately",
                             "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked",
                             "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may",
                             "maybe",
                             "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly",
                             "much",
                             "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary",
                             "need",
                             "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non",
                             "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "n't",
                             "o",
                             "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones",
                             "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves",
                             "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per",
                             "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides",
                             "q",
                             "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding",
                             "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same",
                             "saw",
                             "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed",
                             "seeming",
                             "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven",
                             "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some",
                             "somebody",
                             "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere",
                             "soon",
                             "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t",
                             "t's", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that",
                             "that's", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence",
                             "there",
                             "there's", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon",
                             "these",
                             "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough",
                             "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
                             "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try",
                             "trying",
                             "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until",
                             "unto",
                             "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v",
                             "value",
                             "various", "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we",
                             "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were", "weren't", "what",
                             "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter",
                             "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
                             "whither", "who", "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing",
                             "wish", "with", "within", "without", "won't", "wonder", "would", "wouldn't", "x", "y",
                             "yes",
                             "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
                             "yourselves", "z", "zero"])

    def __init__(self):
        # variables for removing stop words, digits, punctuation
        # two class labels are dictionary, key is class, value is list of documents
        self.stemmer = PorterStemmer()
        pass

    def convert_text_to_word_list(self, text: str) -> []:
        """ Return words list and term frequncy """

        word_tokens = word_tokenize(text)
        tokens = []
        tf = {}
        for word in word_tokens:
            # remove punctturation and stop word
            if re.compile(r'[a-z]').search(word) is None or word in DataProcessor.stop_words_ or len(word) <= 3:
                continue

            # stem
            word = self.stemmer.stem(word)
            if len(word) <= 3:
                continue
            tokens.append(word)

            # count term frequency
            tf[word] = tf.get(word, 0) + 1

        return tokens, tf


    def parse_article(self, article: str) -> Document:
        """ Parse the article to generate a document object.

        Args:
            @article: represent an article.

        Returns:
            A document instance.
        """
        document = Document()

        # extract text body
        text_compile = re.compile('<text.*?</text>', re.DOTALL).search(article)
        if text_compile is None:
            return None
        else:
            text = text_compile.group(0)

        text = re.sub(pattern='</?[a-z]*?>', repl='', string=text)
        document.text = text

        # extract words list and term frequency
        document.words_list, document.tf = self.convert_text_to_word_list(text)
        if len(document.words_list) <= 0:
            return None

        # extract class label
        topic_labels = set()
        for topics in re.compile('<topics.*?</topics>').findall(article):
            for topic in re.compile('<d>[a-z]*?</d>').findall(topics):
                topic_labels.add(re.sub(pattern='</?d>', repl='', string=topic))
        if len(topic_labels) <= 0:
            return None
        document.class_list = list(topic_labels)

        # train or test
        document.train = re.search('lewissplit="train"', string=article) is not None

        return document

    def extract_documents(self, directory: str) -> []:
        """ Extract documents from raw data.

        Args:
            @data: raw data read from .sgm file.

        Returns:
            A list of document.
        """

        documents = []
        filecount = 0

        # open each .sgm
        with os.scandir(directory) as it:
            for entry in it:
                if entry.name.startswith('reut2') and entry.is_file():
                    filecount += 1
                    print('Processing file {}...'.format(filecount))

                    with open(entry.path, 'rb') as datafile:
                        data = datafile.read().decode('utf8', 'ignore')
                        soup = re.compile('<REUTERS.*?</REUTERS>', re.DOTALL)
                        for article in soup.findall(data):
                            document = self.parse_article(article.lower())
                            if document is not None:
                                documents.append(document)

                    print('Finished processing file {}...'.format(filecount))

                    assert filecount != 0

        return documents

    def data_preprocess(self, directory: str):
        """ Read data from the /data folder. Transform the data to a list of documents. Generate feature vectors for documents.

        In this case, I choose unique terms to construct feature vector.

        """
        # generate list of document objects for constructing feature vector
        documents = self.extract_documents(directory)
        print("\n========== Construct a list of TOPIC words ==========")
        _train_documents = []
        _test_documents = []
        bag_of_classes = set()
        for document in documents:
            if document.train:
                _train_documents.append(document)
                bag_of_classes = bag_of_classes.union(document.class_list)
            else:
                _test_documents.append(document)

        StaticData.bag_of_classes = bag_of_classes

        print("Finish constructing TOPIC list.")

        return _train_documents, _test_documents

