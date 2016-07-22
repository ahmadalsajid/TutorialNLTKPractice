from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_text = "This is an example showing off stop word filtration."
stop_words = set(stopwords.words("english"))
words = word_tokenize(example_text)
'''
# long version
filtered_sentence = []
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
'''
filtered_sentence = [w for w in words if w not in stop_words]
print(filtered_sentence)



