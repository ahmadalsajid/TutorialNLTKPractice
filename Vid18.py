import random
import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
import pickle


class VoteClassifier(ClassifierI):
    def __init__(self, *classifier):
        self._classifier = classifier

    def classify(self, features):
        votes = []
        for c in self._classifier:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifier:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open("positive.txt", "r").read()
short_neg = open("negative.txt", "r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r, "pos"))

for r in short_neg.split('\n'):
    documents.append((r, "neg"))


all_words = []

short_pos_words = nltk.word_tokenize(short_pos)
short_neg_words = nltk.word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())
for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = nltk.word_tokenize(document)
    features = {}
    for wrd in word_features:
        features[wrd] = (wrd in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)


# positive data example
training_set = featuresets[:10000]
testing_set = featuresets[10000:]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes accuracy % : ", nltk.classify.accuracy(classifier, testing_set))
classifier.show_most_informative_features(15)

# MultinomialNB
MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
print("MultinomialNB Naive Bayes accuracy % : ", nltk.classify.accuracy(MultinomialNB_classifier, testing_set))

# BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB Naive Bayes accuracy % : ", nltk.classify.accuracy(BernoulliNB_classifier, testing_set))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Naive Bayes accuracy % : ", nltk.classify.accuracy(LogisticRegression_classifier, testing_set))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Naive Bayes accuracy % : ", nltk.classify.accuracy(SGDClassifier_classifier, testing_set))

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Naive Bayes accuracy % : ", nltk.classify.accuracy(LinearSVC_classifier, testing_set))

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Naive Bayes accuracy % : ", nltk.classify.accuracy(NuSVC_classifier, testing_set))


voted_classifier = VoteClassifier(classifier, MultinomialNB_classifier, BernoulliNB_classifier,
                                  LogisticRegression_classifier, SGDClassifier_classifier, LinearSVC_classifier,
                                  NuSVC_classifier)

print("voted_classifier accuracy % : ", nltk.classify.accuracy(voted_classifier, testing_set))
