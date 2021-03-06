import random
import nltk
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
             ]
'''
# this is same as teh above one-liner
documents = []
for category in movie_reviews.categories():
    for fieldid in movie_reviews.fieldids(category):
        documents.append((list(movie_reviews.words(fieldid)), category))
'''

random.shuffle(documents)
# print(documents[0])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
# print(all_words["stupid"])
