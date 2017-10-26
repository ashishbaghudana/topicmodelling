import numpy as np
import logging


class TwitterLDA(object):
    def __init__(self, corpus, num_topics=10, num_users=100, iterations=100, alpha=0.01, beta_k=0.001, beta_b=0.001,
                 gamma=0.01):
        self.corpus = corpus
        self.num_topics = num_topics
        self.num_users = num_users
        self.iterations = iterations
        self.alpha = alpha
        self.beta_k = beta_k
        self.beta_b = beta_b
        self.gamma = gamma

        self.nuk = np.zeros((self.num_users, self.num_topics))
        self.nuk_sum = np.zeros(self.num_users)

        self.ngv = np.zeros(len(self.corpus.vocab))
        self.ngv_sum = 0

        self.nakv = np.zeros((self.num_topics, len(self.corpus.vocab)))
        self.nakv_sum = np.zeros(self.num_topics)
        self.nc = np.zeros(2)
        self.nc_sum = len(self.corpus.vocab)

        self.z_tweets = []
        self.word_category = []
        self.logger = logging.getLogger('TwitterLDA')

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

        np.random.seed(10)

    def initialize(self):
        for u in range(self.num_users):
            self.z_tweets.append([])
            self.word_category.append([])
            for t in range(len(self.corpus.docs[u])):
                topic = np.random.randint(0, self.num_topics)
                self.z_tweets[u].append(topic)
                self.nuk[u][topic] += 1
                self.nuk_sum[u] += 1
                self.word_category[u].append([])

                for w in range(len(self.corpus.docs[u][t])):
                    category = np.random.randint(0, 2)
                    self.word_category[u][t].append(category)
                    if category == 0:
                        self.ngv[self.corpus.docs[u][t][w]] += 1
                        self.ngv_sum += 1
                        self.nc[0] += 1
                    else:
                        self.nakv[topic][self.corpus.docs[u][t][w]] += 1
                        self.nakv_sum[topic] += 1
                        self.nc[1] += 1

    def sample_z(self, u, t):
        old_topic = self.z_tweets[u][t]

        self.nuk[u][old_topic] -= 1
        self.nuk_sum[u] -= 1

        for n in range(len(self.corpus.docs[u][t])):
            if self.word_category[u][t][n] == 1:
                self.nakv[old_topic][self.corpus.docs[u][t][n]] -= 1
                self.nakv_sum[old_topic] -= 1
                self.nc[1] -= 1

        probabilities = np.zeros(self.num_topics)
        probabilities_sum = 0
        for k in range(self.num_topics):
            term1, term2 = 0, 0
            term1 = self.nuk[u][k] + self.alpha
            count = 0
            for w in range(len(self.corpus.word_occurrence[u][t])):
                key = self.corpus.word_occurrence[u][t][w][0]
                val = self.corpus.word_occurrence[u][t][w][1]
                for j in range(val):
                    term2 = (term2 * self.nakv[k][key] + self.beta_k + j) / (self.nakv_sum[k] + len(self.corpus.vocab) +
                                                                             count)
                count += val
            probabilities[k] = term1 * term2
            probabilities_sum += probabilities[k]

        probabilities = probabilities / probabilities_sum

        new_topic = 0
        rand_sum = np.random.rand()
        tmp_sum = 0

        for k in range(self.num_topics):
            tmp_sum += probabilities[k]
            if rand_sum <= tmp_sum:
                new_topic = k
                break

        self.nuk[u][new_topic] += 1
        self.nuk_sum[u] += 1

        for n in range(len(self.corpus.docs[u][t])):
            if self.word_category[u][t][n] == 1:
                self.nakv[new_topic][self.corpus.docs[u][t][n]] += 1
                self.nakv_sum[new_topic] += 1
                self.nc[1] += 1

        return new_topic

    def sample_c(self, u, t, n):
        old_category = self.word_category[u][t][n]
        topic = self.z_tweets[u][t]

        if old_category == 0:
            self.ngv[self.corpus.docs[u][t][n]] -= 1
            self.ngv_sum -= 1
            self.nc[0] -= 1
        else:
            self.nakv[topic][self.corpus.docs[u][t][n]] -= 1
            self.nakv_sum[topic] -= 1
            self.nc[1] -= 1

        probabilities = np.zeros(2)

        probabilities[0] = (self.nc[0] + self.gamma) * (self.ngv[self.corpus.docs[u][t][n]] + self.beta_b) / \
                           (self.ngv_sum + len(self.corpus.vocab) * self.beta_b)
        probabilities[1] = (self.nc[1] + self.gamma) * (self.nakv[topic][self.corpus.docs[u][t][n]] + self.beta_k) / \
                           (self.nakv_sum[topic] + len(self.corpus.vocab) * self.beta_k)
        probabilities_sum = probabilities[0] * probabilities[1]

        rand_sum = np.random.rand() * probabilities_sum
        if rand_sum <= probabilities[0]:
            new_category = 0
        else:
            new_category = 1

        if new_category == 0:
            self.ngv[self.corpus.docs[u][t][n]] += 1
            self.ngv_sum += 1
            self.nc[0] += 1
        else:
            self.nakv[topic][self.corpus.docs[u][t][n]] += 1
            self.nakv_sum[topic] += 1
            self.nc[1] += 1

        return new_category

    def inference(self, iteration):
        for u in range(self.num_users):
            self.logger.info('[Iteration %d] Performing Gibbs Sampling for user %d' % (iteration, u))
            for t in range(len(self.corpus.docs[u])):
                self.z_tweets[u][t] = self.sample_z(u, t)
                for w in range(len(self.corpus.docs[u][t])):
                    self.word_category[u][t][w] = self.sample_c(u, t, w)

    def run_twitter_lda(self):
        self.logger.info('Initializing tweets with random topics')
        self.initialize()

        for iter in range(self.iterations):
            self.logger.info('[Iteration %d] Starting Gibbs Sampling' % iter)
            self.inference(iter)
