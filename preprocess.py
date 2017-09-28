import string

from abc import ABCMeta, abstractmethod
from nltk.corpus import stopwords
from nltk import stem
from nltk.tokenize import word_tokenize, wordpunct_tokenize


class Pipeline(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def preprocess(self, doc):
        pass


class Tokenizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def tokenize(self, doc):
        pass


class Stemmer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def stem(self, doc):
        pass


class Filter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def filter(self, word):
        pass


class Mapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, word):
        pass


class NLTKTokenizer(Tokenizer):
    def tokenize(self, doc):
        return word_tokenize(doc)


class WordPunctTokenizer(Tokenizer):
    def tokenize(self, doc):
        return wordpunct_tokenize(doc)


class PorterStemmer(Stemmer, Mapper):
    def __init__(self):
        self.stemmer = stem.PorterStemmer()

    def map(self, token):
        return self.stem(token)

    def stem(self, token):
        return self.stemmer.stem(token)


class LowerCaseMapper(Mapper):
    def map(self, token):
        return token.lower()


class StopWordFilter(Filter):
    def __init__(self):
        self.stop_words = stopwords.words('english')

    def filter(self, word):
        return word not in self.stop_words


class PunctuationFilter(Filter):
    def filter(self, word):
        return word not in string.punctuation


class LDAPipeline(Pipeline):
    def __init__(self, tokenizer=None, mappers=[], filters=[]):
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = NLTKTokenizer()

        self.mappers = mappers
        if self.mappers is None:
            self.mappers = []

        self.filters = filters
        if self.filters is None:
            self.filters = []

        try:
            assert isinstance(self.tokenizer, Tokenizer)
        except AssertionError:
            raise ValueError('The tokenizer passed should be of type "Tokenizer"')

        try:
            for filter_obj in self.filters:
                assert isinstance(filter_obj, Filter)
        except AssertionError:
            raise ValueError('The filters passed should be of type "Filter')

        try:
            for mapper_obj in self.mappers:
                assert isinstance(mapper_obj, Mapper)
        except AssertionError:
            raise ValueError('The mappers passed should be of type "Mapper')

    def preprocess(self, doc):
        tokens = [token for token in self.tokenizer.tokenize(doc)]
        for filter_obj in self.filters:
            tokens = filter(filter_obj.filter, tokens)
        for mapper_obj in self.mappers:
            tokens = map(mapper_obj.map, tokens)
        return tokens
