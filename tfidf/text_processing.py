from concurrent.futures import process
from hashlib import new
from multiprocessing.resource_sharer import stop
import re
import numpy as np
import os
import math


class PreProcessing():
    def manual_tokenization(self, text):
        container = ""
        newText = []
        for i in range(len(text)):
            if (text[i] != ' ' and text[i] != '\t'):
                container = container + text[i]
                if (i == len(text)-1):
                    newText.append(text[i])
            else:
                newText.append(container)
                container = ""

        return newText

    def dot_tokenization(self, text):
        container = ""
        newText = []
        for i in range(len(text)):
            if (text[i] != '.' and text[i] != '\t'):
                container = container + text[i]
                if (i == len(text)-1):
                    newText.append(text[i])
            else:
                newText.append(container)
                container = ""

        return newText

    def toLowerCase(self, text):
        text = [word.lower() for word in text]
        return text

    def removeStopWords(self, text):
        stopwords_path = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
        with open(stopwords_path, 'r') as f:
            stop_words = f.read().splitlines()
        text = [word for word in text if word not in stop_words]
        return text

    def removeSpecialCharacters(self, text):
        return re.sub(r"[^a-zA-Z0-9]+", ' ', text)

    def cleanRawFrequency(self, termFrequency={}):
        fromStopWords = self.removeStopWords(termFrequency.keys())
        for k in tuple(termFrequency.keys()):
            if k not in fromStopWords:
                del termFrequency[k]

        return termFrequency

    def removeNumbers(self, diction={}):
        for k in tuple(diction.keys()):
            if k.isdigit():
                del diction[k]
        return diction
