# Topic Modelling (LDA and Twitter-LDA)

This repository hosts code for LDA and Twitter-LDA. Both implementations are single-core and unoptimized and purely for learning purposes.

## LDA

LDA assumes that each document is a mixture over topics of words. Given that the probability distributions of the topics is not initially known, this version of LDA uses Gibbs Sampling to estimate the probability distribution of topics over documents and words over topics.

## Twitter-LDA

Twitter LDA is more suited for short documents and tweets. This assumes that tweets are mixtures over users, i.e. each user tweets about a few topics. Like with LDA, Twitter-LDA uses Gibbs Sampling to estimate the probability distribution of topics over users and words over topics.
