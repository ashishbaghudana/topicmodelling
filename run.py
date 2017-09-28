from .structures import Corpus
from .lda import LDA


def main():
    corpus = Corpus(fname='/Users/ashish/Desktop/sample.txt')
    corpus.vectorize()

    print (corpus.vocab)

    lda = LDA(corpus, num_topics=2)

    for i in range(100):
        lda.iteration()

if __name__ == '__main__':
    main()
