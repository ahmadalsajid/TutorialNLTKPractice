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


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
             ]
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for wrd in word_features:
        features[wrd] = (wrd in words)
    return features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier_f = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
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

'''# GaussianNB has errors
GaussianNB_classifier = SklearnClassifier(GaussianNB())
GaussianNB_classifier.train(training_set)
print("GaussianNB Naive Bayes accuracy % : ", nltk.classify.accuracy(GaussianNB_classifier, testing_set))'''

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Naive Bayes accuracy % : ", nltk.classify.accuracy(LogisticRegression_classifier, testing_set))

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Naive Bayes accuracy % : ", nltk.classify.accuracy(SGDClassifier_classifier, testing_set))

'''SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Naive Bayes accuracy % : ", nltk.classify.accuracy(SVC_classifier, testing_set))'''

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
print("Classification : ", voted_classifier.classify(testing_set[0][0]), "confidence %:",
      voted_classifier.confidence(testing_set[0][0]))
print("Classification : ", voted_classifier.classify(testing_set[1][0]), "confidence %:",
      voted_classifier.confidence(testing_set[1][0]))
print("Classification : ", voted_classifier.classify(testing_set[2][0]), "confidence %:",
      voted_classifier.confidence(testing_set[2][0]))
print("Classification : ", voted_classifier.classify(testing_set[3][0]), "confidence %:",
      voted_classifier.confidence(testing_set[3][0]))
print("Classification : ", voted_classifier.classify(testing_set[4][0]), "confidence %:",
      voted_classifier.confidence(testing_set[4][0]))
print("Classification : ", voted_classifier.classify(testing_set[5][0]), "confidence %:",
      voted_classifier.confidence(testing_set[5][0]))
