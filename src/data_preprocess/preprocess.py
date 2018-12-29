#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gjxhlan
"""
import numbers
import os
import re
import string
import pickle

from src.data_structure.data_structure import Document
from src.metric.metric import calculate_priority_by_tfidf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

class DataSet:
    """ Class DataSet for get data from reuter-21578.

    get_train_and_test_documents(): get the data, return x_train, y_train, x_test, y_test
    """

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
        for entry in os.scandir(directory):
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


    def get_train_and_test_documents(self):
        """ Read data from the data/ folder. Transform the data to a list of documents.

            Return
            ------------
            x_train: list of string
            y_train: list of labels
            x_test: list of string
            y_test: list of labels
        """

        data_dir = 'data/'
        if not os.path.exists(data_dir):
            raise OSError('Please store original data files in data/ directory')

        if os.path.isfile('data/train.pkl'):
            print("========== Extract data from pickle files ==========")
            with open('data/train.pkl', 'rb') as f:
                train_documents = pickle.load(f)
            with open('data/test.pkl', 'rb') as f:
                test_documents = pickle.load(f)
        else:
            print("========== Parse data files ==========")
            documents = self.extract_documents(data_dir)
            train_documents, test_documents = [], []
            
            for document in documents:
                if document.train:
                    train_documents.append(document)
                else: 
                    test_documents.append(document)
            
            with open('data/train.pkl', 'wb') as f:
                pickle.dump(train_documents, f)
            with open('data/test.pkl', 'wb') as f:
                pickle.dump(test_documents, f)
        
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        for document in train_documents:
            x_train.append(document.text)
            y_train.append(document.class_list)

        for document in test_documents:
            x_test.append(document.text)
            y_test.append(document.class_list)

        return x_train, y_train, x_test, y_test


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
