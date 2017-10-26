import string
import re
import logging
import unicodedata

from abc import ABCMeta, abstractmethod
from nltk.corpus import stopwords
from nltk import stem
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from tweetokenize import Tokenizer as Tweetokenizer

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


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


class SpaceTokenizer(Tokenizer):
    def tokenize(self, doc):
        return doc.split(" ")


class NLTKTokenizer(Tokenizer):
    def tokenize(self, doc):
        return word_tokenize(doc)


class WordPunctTokenizer(Tokenizer):
    def tokenize(self, doc):
        return wordpunct_tokenize(doc)


class TweetTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = Tweetokenizer(usernames='', urls='', phonenumbers='', times='', numbers='')

    def tokenize(self, doc):
        return self.tokenizer.tokenize(doc)


class PorterStemmer(Stemmer, Mapper):
    def __init__(self):
        self.stemmer = stem.PorterStemmer()

    def map(self, token):
        return self.stem(token)

    def stem(self, token):
        return self.stemmer.stem(token)


class LowercaseMapper(Mapper):
    def map(self, token):
        return token.lower()


class ASCIINormalizer(Mapper):
    def map(self, token):
        return unicodedata.normalize('NFD', token).encode('ascii', 'ignore')


class StopWordFilter(Filter):
    def __init__(self):
        self.stop_words = stopwords.words('english')

    def filter(self, word):
        return word not in self.stop_words


class PunctuationFilter(Filter):
    def filter(self, word):
        return word not in string.punctuation


class LengthFilter(Filter):
    def __init__(self, length=3):
        self.length = length

    def filter(self, word):
        return len(word) > self.length


class UnnecessaryWordsFilter(Filter):
    def __init__(self, words=None):
        self.words = words
        if self.words is None:
            words = ['http', 'it\'s', 'won\'t', ' ', 'https', '...', 'i', 'rt', '']
            self.words = set(words)

    def filter(self, word):
        return word.lower() not in self.words


class ASCIIMapper(Mapper):
    def __init__(self):
        self.ascii = string.printable

    def map(self, word):
        return re.sub(r'[^\x00-\x7F]+', ' ', word).strip()


class LDAPipeline(Pipeline):
    def __init__(self, tokenizer=None, mappers=[], filters=[]):
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            self.tokenizer = NLTKTokenizer()

        self.mappers = mappers
        if self.mappers is None:
            self.mappers = [ASCIINormalizer(), LowercaseMapper()]

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
