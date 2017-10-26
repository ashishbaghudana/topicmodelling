from topicmodelling.lda import LDA
from topicmodelling.structures import Corpus

import numpy as np


def main():
    corpus = Corpus(fname='/Users/ashish/Downloads/smsspamcollection/20newsgroups.txt.small')
    corpus.vectorize()
    print (corpus.vocab)
    lda = LDA(corpus, num_topics=20)
    lda.create_model()
    # lda.print_topics()
    for doc, topic in zip(corpus, lda.topic_distribution):
        print 'Topic:', np.argmax(topic) + 1, '\t\t', corpus.vocab.id2doc(doc)


if __name__ == '__main__':
    main()
