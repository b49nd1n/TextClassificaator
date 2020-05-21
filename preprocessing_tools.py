import re

import nltk
from nltk.corpus import stopwords


def delete_non_letters(words):
    new_words = []
    for word in words:
        new_word = "".join(c for c in word if c.isalpha())
        if new_word != '':
            new_words.append(new_word)
    return new_words


def delete_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english') and word not in stopwords.words('russian'):
            new_words.append(word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def normalize(text, tokenized=False, del_stopwords=False):
    if not tokenized:
        text = nltk.word_tokenize(text)

    text = delete_non_letters(text)
    if del_stopwords:
        text = delete_stopwords(text)

    text = to_lowercase(text)

    text = [word for word in text if len(word) > 1]
    return text