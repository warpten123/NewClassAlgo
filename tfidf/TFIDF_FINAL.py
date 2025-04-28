
import time
import pdfplumber
import re
import csv
import pandas as pd
import glob
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import math
import numpy as np
from fpdf import FPDF
from collections import ChainMap
# use this when running main.py in backend
from tfidf.text_processing import PreProcessing
# from text_processing as preProc uncomment this shit if you want to run this file only
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


class Processing():
    def __init__(self, path):
        self.path = path

    def getFromPDF(self, filename):  # notused
        finalText = " "
        with pdfplumber.open('assets/temp/' + filename) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                break
            extractFromPDF = ""
        return finalText

    def toCSV(self, goals, index, dict):  # noteused
        filename = str(index) + ".csv"
        direc = 'tfidf/Term/' + goals + "/"
        with open(direc + filename, 'w') as csvFile:
            for list in dict:
                w = csv.DictWriter(csvFile, list.keys())
                w.writeheader()
                w.writerow(list)

    def fromPDFFolders(self, goalName):
        tf_idf = {}
        count = 0
        directory = (glob.glob("../Data Set/" + goalName + "/*.pdf"))
        extractedText = " "
        finalText = " "
        for file in directory:
            with pdfplumber.open(file) as pdf:
                count += 1
              
                for page in pdf.pages:
                    extractedText = page.extract_text()
            finalText = finalText + extractedText
           
            extractedText = ""
            tf_idf = self.mainProcessing(finalText, goal, count)
        return tf_idf

    def preProcessing(self, text):
        # stop_words = set(stopwords.words("english"))
        # lemmatizer = WordNetLemmatizer()
        # lemmatized_tokens = []
        # # Tokenize document
        # tokens = word_tokenize(text)

        # # Remove stopwords and punctuations, and convert to lowercase
        # filtered_tokens = [
        #     token.lower() for token in tokens if token.isalnum() and token not in stop_words]

        # # Lemmatize tokens
        # lemmatized_tokens.append([lemmatizer.lemmatize(
        #     token) for token in filtered_tokens])

        # # Join tokens back into a string
        # # preprocessed_doc = " ".join(lemmatized_tokens)
        # # preprocessed_docs.append(preprocessed_doc)

        # return lemmatized_tokens
        preProc = PreProcessing()
        text = preProc.removeSpecialCharacters(text)
        text = preProc.manual_tokenization(text)
        text = preProc.removeStopWords(text)
        text = preProc.toLowerCase(text)
        return text

    def preprocess_documents(docs):
        preprocessed_docs = []
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = []
        # Tokenize document
        tokens = word_tokenize(doc)

        # Remove stopwords and punctuations, and convert to lowercase
        filtered_tokens = [
            token.lower() for token in tokens if token.isalnum() and token not in stop_words]

        # Lemmatize tokens
        lemmatized_tokens.append([lemmatizer.lemmatize(
            token) for token in filtered_tokens])

        # Join tokens back into a string
        # preprocessed_doc = " ".join(lemmatized_tokens)
        # preprocessed_docs.append(preprocessed_doc)

        return lemmatized_tokens

    def populateClass(self, text):
        preProc = PreProcessing()
        initialList = {}
        for t in text:
            initialList[t.lower()] = 0
        initialList = preProc.removeNumbers(initialList)
        initialList = preProc.cleanRawFrequency(initialList)
        return initialList

    def term_vectors(self, text, populatedDict):
        for word in text:
            for dic in populatedDict:
                if (dic == word):
                    populatedDict[dic] = populatedDict[dic] + 1
        return populatedDict

    def term_frequency(self, populatedDict):

        for dic in populatedDict:
            populatedDict[dic] = populatedDict[dic] / len(populatedDict)
        return populatedDict

    def inverse_frequency(self, features,  listOfDict=[{}]):
        idf = features
        number_of_documents = 17
        number_of_word = 0
        count = 0
        for col in features:
            for f in listOfDict:
                if f.__contains__(col):
                    number_of_word += 1
            idf[col] = math.log10(number_of_documents / number_of_word)
            number_of_word = 0
        return idf

    def calculateTFIDF(self, listofDict, idf, tf_idf):
        temp = {}
        count = 1
        for list in listofDict:
            temp = list
            for features in idf:
                if temp.__contains__(features):
                    temp[features] = temp[features] * idf[features]
            tf_idf.append(temp)

        return tf_idf
        # return tf_idf

    def computeTF_IDF(self, tf, idf):
        tf_idf = tf
        for word in tf:
            if idf.__contains__(word):
                tf_idf[word] = float(tf_idf[word]) * float(idf[word])
            else:
                tf_idf[word] = 0

        return tf_idf

    def countNumberofDocs(self):  # noteused
        count = 0
        dir_path = r"Term/Goal 1"
        for path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        return count

    def mergeDictionaries(self, listofDict):
        finalDict = {**listofDict}
        return finalDict

    def csvToDict(self):  # not used
        with open('tfidf/Results/TFIDF.csv') as f:
            a = [{k: float(v) for k, v in row.items()}
                 for row in csv.DictReader(f, skipinitialspace=True)]
        return a

    def convertingToDP(self, tf_idf):
        df = pd.DataFrame.from_dict(tf_idf)
        df2 = df.replace(np.nan, 0)
        output_dir = os.path.join(os.path.dirname(__file__), 'Results')
        os.makedirs(output_dir, exist_ok=True)

    
        output_path = os.path.join(output_dir, 'TFIDF.csv')
        print("output_path",output_path)
        df2.to_csv(output_path)
        return df2

    def extractAllPDF(self, goal):
        directory = (glob.glob("tfidf/Data Set/" + goal + "/*.pdf"))
        extractedText = " "
        count = 0
        finalText = " "
      
        for file in directory:
            with pdfplumber.open(file) as pdf:
                count += 1
            
                for page in pdf.pages:
                    extractedText = page.extract_text()
                 
                    finalText = finalText + extractedText
        return finalText

    def listToPDF(self, processedText, goal):
        # Define base directories
        base_dir = os.path.dirname(__file__)
        text_files_dir = os.path.join(base_dir, "Text Files")
        training_set_dir = os.path.join(base_dir, "Training Set")

        # Ensure directories exist
        os.makedirs(text_files_dir, exist_ok=True)
        os.makedirs(training_set_dir, exist_ok=True)

        # Construct file paths
        text_file_path = os.path.join(text_files_dir, f"{goal}.txt")
        pdf_file_path = os.path.join(training_set_dir, f"{goal} Training Set.pdf")

        # Write processed text to a text file
        with open(text_file_path, 'w') as fp:
            fp.write(' '.join(processedText))

        # Read the text file and create a PDF
        with open(text_file_path, "r") as f:
            pdf = FPDF()
            pdf.set_font('Times', '', 12)
            pdf.add_page()
            for line in f:
                pdf.cell(200, 10, txt=line.strip(), ln=1, align='C')
            pdf.output(pdf_file_path)

    def mergeAllDict(self, l):
        d = {}
        for dictionary in l:
            d.update(dictionary)
        return d

    def lemmatization(self, text):
        lemmatizer = WordNetLemmatizer()
        temp = []
        for str in text:
            lemma = lemmatizer.lemmatize(str)
            temp.append(lemma)
        return temp

    def createTFIDF(self, rawText):
        start_time = time.time()
        goals = ['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4', 'Goal 5',
                 'Goal 6', 'Goal 7', 'Goal 8', 'Goal 9', 'Goal 10', 'Goal 11', 'Goal 12',
                 'Goal 13',
                 'Goal 14', 'Goal 15', 'Goal 16', 'Goal 17'
                 ]
        TFIDF = Processing(rawText)
        tf = [{}]  # create list of dicts
        count = 1
        final_features = {}
        idf = {}
        tf_idf = [{}]
        temp = {}
        merge = {}
        for goal in goals:
            rawText = TFIDF.extractAllPDF(goal)
            preprocessedText = TFIDF.preProcessing(rawText)
            TFIDF.listToPDF(preprocessedText, goal)
            temp = TFIDF.populateClass(preprocessedText)
            temp = TFIDF.term_vectors(preprocessedText, temp)
            temp = TFIDF.term_frequency(temp)
            tf.append(temp)
            count += 1
        merge = TFIDF.mergeAllDict(tf)
        idf = TFIDF.inverse_frequency(merge, tf)
        tf_idf = TFIDF.calculateTFIDF(tf, idf, tf_idf)
        tf_idf = TFIDF.convertingToDP(tf_idf)
        end_time = time.time()
        execution_time = end_time - start_time
      
        return tf_idf

    def insertNewData(self, result={}):
        length = 0
        newData = " "
        preProc = PreProcessing()
        for value in result:
            newData = newData + result[value]
        newData = self.preProcessing(newData)
        newData = self.lemmatization(newData)
        temp = self.populateClass(newData)
        temp = self.term_vectors(newData, temp)
        term_frequency = self.term_frequency(temp)
        listOfDict = self.csvToDict()
        idf = self.inverse_frequency(term_frequency, listOfDict)
     


if __name__ == '__main__':
    rawText = ""
    print("main on tfif")
    TFIDF = Processing(rawText)
    TFIDF.createTFIDF(rawText)
