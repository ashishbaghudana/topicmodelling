"""
This module contains all the data structures used by LDA
"""
import codecs
from repr import repr
from collections import defaultdict
import numpy as np

from topicmodelling.preprocess import (
    LDAPipeline,
    LowercaseMapper,
    Pipeline,
    PorterStemmer,
    PunctuationFilter,
    StopWordFilter
)


class Vocabulary(object):
    """Vocabulary is a set of words present in corpus

    Vocabulary builds a set of unique words occurring in the corpus.
    """

    def __init__(self, docs):
        """
        Initialize Vocabulary data structure with a collection of documents

        :param docs: Documents
        """
        self.docs_ = docs
        self.id2word = {}
        self.word2id = {}
        self.vocab_ = set()
        self.__build_vocab()
        self.__build_id2word()

    def __build_vocab(self):
        """
        Build vocabulary from the documents

        :return: None
        """
        for doc in self.docs_:
            for token in doc:
                if isinstance(doc, tuple):
                    self.vocab_.add(token[1])
                else:
                    self.vocab_.add(token)

    def __build_id2word(self):
        for id_, token in enumerate(self.vocab_):
            self.id2word[id_] = token
            self.word2id[token] = id_

    def doc2bow(self, tokenized_doc):
        bow = [self.word2id[token] for token in tokenized_doc if token in self.vocab_]
        return bow

    def id2doc(self, tokenized_doc):
        tokens = [self.id2word[token] for token in tokenized_doc if token in self.id2word]
        return tokens

    def filter_extremes(self, no_above=0.5, no_below=5, restrict_size=100000):
        raise NotImplementedError('This feature has not yet been implemented')

    def __len__(self):
        return len(self.vocab_)

    def __iter__(self):
        return self.vocab_.__iter__()

    def __str__(self):
        return "Vocabulary ({} tokens) - {}".format(self.__len__(), repr(list(self.vocab_)))

    def __repr__(self):
        return self.__str__()


class Documents(object):
    """Documents streams documents from file iteratively

    Documents allows users to stream documents from a file without
    having to load everything in memory
    """
    def __init__(self, filename, pipeline=None):
        self.filename = filename
        self.length = 0

        self.pipeline = pipeline
        if self.pipeline is None:
            self.pipeline = LDAPipeline(
                mappers=[LowercaseMapper(), PorterStemmer()],
                filters=[StopWordFilter(), PunctuationFilter()])

        try:
            assert isinstance(self.pipeline, Pipeline)
        except AssertionError:
            raise ValueError('The type of pipeline should be Pipeline')

        for _ in self.__iter_without_preprocess__():
            self.length += 1

    def __len__(self):
        return self.length

    def __iter_without_preprocess__(self):
        with codecs.open(self.filename, encoding='utf8') as freader:
            for line in freader:
                yield line

    def __iter__(self):
        with codecs.open(self.filename, encoding='utf8') as freader:
            for line in freader:
                yield self.pipeline.preprocess(line)

    def __str__(self):
        return "Documents ({} docs)".format(self.__len__())

    def __repr__(self):
        return self.__str__()


class TwitterDocuments(object):
    def __init__(self, filename, pipeline):
        self.filename = filename
        self.pipeline = pipeline

        self.length = 0
        self.num_users = 0

        self.users = {}
        self.user_tweet_count = defaultdict(int)
        self.docs = []
        self.vectors = []
        self.vocab = None

        self.preprocess()

    def __len__(self):
        return self.length

    def preprocess(self):
        self.length = 0
        self.num_users = 0

        with codecs.open(self.filename, encoding='utf-8') as freader:
            for line in freader:
                user, tweet = line.split(',')[5], line.split(',')[1]
                self.length += 1
                # If user has not been seen before, add user to self.users
                if user not in self.users:
                    self.users[user] = self.num_users
                    self.num_users += 1
                    self.docs.append([])
                self.user_tweet_count[user] += 1
                self.docs[self.users[user]].append(self.pipeline.preprocess(tweet))

        tweets = []

        for user in range(len(self.docs)):
            for tweet in self.docs[user]:
                tweets.append(tweet)

        self.vocab = Vocabulary(tweets)

        for user in range(len(self.docs)):
            self.vectors.append([])
            for tweet in self.docs[user]:
                self.vectors[user].append(self.vocab.doc2bow(tweet))
        self.vectors = np.array(self.vectors)

    def __getitem__(self, item):
        return self.vectors[item]

    def __iter__(self):
        return self.vectors.__iter__()


class Corpus(object):
    """Corpus stream documents and represents it as a vector

    Corpus helps build the vocabulary and then returns documents in the corpus
    as a vector of the bag of words representation
    """
    def __init__(self, filename, pipeline=None):
        self.filename = filename
        self.docs = Documents(filename=self.filename, pipeline=pipeline)
        self.vocab = Vocabulary(docs=self.docs)
        self.vectors = None

    def vectorize(self):
        self.vectors = []
        for doc in self.docs:
            self.vectors.append(self.vocab.doc2bow(doc))

    def __iter__(self):
        if self.vectors:
            for vector in self.vectors:
                yield vector
        else:
            for doc in self.docs:
                yield self.vocab.doc2bow(doc)

    def __len__(self):
        return len(self.docs)

    def __str__(self):
        return "Documents ({} docs) with Vocabulary ({} tokens)".format(self.__len__(), self.vocab.__len__())

    def __repr__(self):
        return self.__str__()


class TwitterCorpus(object):
    def __init__(self, filename, pipeline):
        self.filename = filename
        self.pipeline = pipeline
        self.docs = TwitterDocuments(filename=self.filename, pipeline=self.pipeline)
        self.vocab = self.docs.vocab
        self.word_occurrence = []
        for user, user_tweets in enumerate(self.docs):
            self.word_occurrence.append([])
            for tweet, tokens in enumerate(user_tweets):
                self.word_occurrence[user].append([])
                self.word_occurrence[user][tweet] = self.count_word_occurrences(tokens)

    def count_word_occurrences(self, tokens):
        word_count = defaultdict(int)
        for token in tokens:
            word_count[token] += 1
        word_occurences = []
        for token in tokens:
            count = [token, word_count[token]]
            word_occurences.append(count)
        return word_occurences

