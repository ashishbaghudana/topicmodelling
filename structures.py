"""
This module contains all the data structures used by LDA
"""
from .preprocess import (
    LDAPipeline,
    LowerCaseMapper,
    Pipeline,
    PunctuationFilter,
    StopWordFilter
)
from repr import repr

import codecs


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
    def __init__(self, fname, pipeline=None):
        self.fname = fname
        self.length = 0

        self.pipeline = pipeline
        if self.pipeline is None:
            self.pipeline = LDAPipeline(
                mappers=[LowerCaseMapper()],
                filters=[StopWordFilter(), PunctuationFilter()])

        try:
            assert isinstance(self.pipeline, Pipeline)
        except AssertionError:
            raise ValueError('The type of pipeline should be Pipeline')

        for _ in self:
            self.length += 1

    def __len__(self):
        return self.length

    def __iter__(self):
        with codecs.open(self.fname, encoding='utf8') as freader:
            for line in freader:
                yield self.pipeline.preprocess(line)

    def __str__(self):
        return "Documents ({} docs)".format(self.__len__())

    def __repr__(self):
        return self.__str__()


class Corpus(object):
    """Corpus stream documents and represents it as a vector

    Corpus helps build the vocabulary and then returns documents in the corpus
    as a vector of the bag of words representation
    """
    def __init__(self, fname, pipeline=None):
        self.fname = fname
        self.docs = Documents(fname=self.fname, pipeline=pipeline)
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
