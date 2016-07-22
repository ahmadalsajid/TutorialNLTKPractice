from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
'''
print(lemmatizer.lemmatize("cat"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("horses"))
print(lemmatizer.lemmatize("cat"))
'''
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a"))


