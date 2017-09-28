import numpy as np
import logging
import operator


class LDA(object):
    """A Gibbs Sampler for collapsed LDA

    Allows approximate sampling from the posterior distribution over
    assignments of topic labels to words in a collapsed LDA model
    """

    def __init__(self, corpus, num_topics=10, alpha=0.1, beta=0.01):
        """Initialize LDA with number of topics, alpha and beta

        :param docs:
        :param num_topics:
        :param alpha:
        :param beta:
        """
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.corpus = corpus
        self.num_words_topic = np.zeros(self.num_topics)
        self.num_words_topic_doc = np.zeros((len(self.corpus), self.num_topics))
        self.num_times_word_topic = np.zeros((self.num_topics, len(self.corpus.vocab)))
        self.topic_distribution = []
        self.initialize()

    def initialize(self):
        self.num_words_topic_doc += self.alpha
        self.num_words_topic += self.beta
        self.num_times_word_topic += self.beta

        for m, doc in enumerate(self.corpus):
            topic_distribution_doc = []
            for token in doc:
                topic = np.random.randint(0, self.num_topics)
                topic_distribution_doc.append(topic)
                self.num_words_topic[topic] += 1
                self.num_words_topic_doc[m][topic] += 1
                self.num_times_word_topic[topic][token] += 1
            self.topic_distribution.append(np.array(topic_distribution_doc))

    def iteration(self):
        for m, doc in enumerate(self.corpus):
            for n, token in enumerate(doc):
                old_topic = self.topic_distribution[m][n]

                # Decrement counts for the old topic
                self.num_words_topic[old_topic] -= 1
                self.num_words_topic_doc[m][old_topic] -= 1
                self.num_times_word_topic[old_topic][token] -= 1

                # Calculate what the new topic will be
                new_topic = self.new_topic(token, m)

                self.topic_distribution[m][n] = new_topic

                self.num_words_topic[new_topic] += 1
                self.num_words_topic_doc[m][new_topic] += 1
                self.num_times_word_topic[new_topic][token] += 1

    def new_topic(self, token, document):
        probabilities = (self.num_times_word_topic[:, token] + self.beta) \
                    * (self.num_words_topic_doc[document]) / self.num_words_topic
        return probabilities.argmax()

