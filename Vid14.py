import random
import nltk
from nltk.corpus import movie_reviews
import pickle

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

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes accuracy % : ", nltk.classify.accuracy(classifier,testing_set))
classifier.show_most_informative_features(15)


save_classifier = open('naivebayes.pickle', 'wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()
