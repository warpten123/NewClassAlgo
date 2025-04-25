import time
import pdfplumber
import os
from tfidf.text_processing import PreProcessing
from nltk import pos_tag
from nltk.tokenize import word_tokenize


class Helper:
    def getRequiredChapters(self):  # get abstract, intro, res methodology
        return True

    def main_logic(self, filename):
        appendedData = ""
        abstract = self.getFromPDFAbstract(filename)
        introduction = self.getFromPDFIntro(filename)
        method = self.getFromPDFMethod(filename)
        appendedData = abstract + introduction + method
       
        return {'abstract': abstract, 'introduction': introduction, 'method': method, 'appendedData': appendedData}

    def passDataToClassify(data):
        return data

    def getFromPDFAbstract(self, filename):
        count = 1
        finalText, final_abstract = " ", " "
        limitPages, currentPage = 10, 0

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(base_dir, 'assets', 'upload', filename)
       
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF

                checkAbs = self.getAbstract(finalText, count)
                if checkAbs:
                    final_abstract = self.cleanString(finalText)
                    break

                if currentPage == limitPages:
                    break

                count += 1
                currentPage += 1
                final_abstract = " "
                finalText = " "

        return final_abstract

    def getFromPDFIntro(self, filename):
        count = 1
        finalText, final_intro = " ", " "
        limitPages, currentPage = 10, 0
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(base_dir, 'assets', 'upload', filename)
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                checkAbs = self.getIntroduction(finalText)
                if (checkAbs):
                    final_intro = finalText
                    final_intro = self.cleanString(final_intro)
                    break
                if (currentPage == limitPages):
                    break
                count += 1
                final_intro = " "
                currentPage += 1
                finalText = " "
        return final_intro

    def getFromPDFMethod(self, filename):
        count = 1
        finalText = " "
        final_method = " "
        limitPages = 10
        currentPage = 0
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(base_dir, 'assets', 'upload', filename)
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                checkAbs = self.getMethodology(finalText)
                if (checkAbs):
                    final_method = finalText
                    final_method = self.cleanString(final_method)
                    break
                count += 1
                if (currentPage == limitPages):
                    break
                currentPage += 1
                final_method = " "
                finalText = " "
        return final_method

    def getAbstract(self, processedText, page):
        count = 0
        pageAbstract = 0
        abstract = False
        if (("ABSTRACT" in processedText or "Abstract" in processedText)
                and ("TABLE OF CONTENTS" not in processedText and "Table of Contents" not in processedText)):
            if (count == 0):
                abstract = True
                pageAbstract = page
            count += 1
        return abstract

    def getIntroduction(self, processedText):
        count = 0
        introduction = False
        if (("INTRODUCTION" in processedText or "Introduction" in processedText)
                and ("TABLE OF CONTENTS" not in processedText and "Table of Contents" not in processedText)):
            if (count == 0):
                introduction = True
            count += 1
        return introduction

    def getMethodology(self, processedText):
        count = 0
        methodology = False
        if (("Research Methodology" in processedText or "RESEARCH METHODOLOGY" in processedText)
                and ("TABLE OF CONTENTS" not in processedText and "Table of Contents" not in processedText)):
            if (count == 0):
                methodology = True
            count += 1
        return methodology

    def cleanString(self, text):
        if ("\n" in text):
            text = text.replace('\n', ' ')
        return text

    def getRules(self):
        rules = []
        file = open('tfidf/Results/rules.txt', 'r')
        Lines = file.readlines()
        for line in Lines:
            rules.append(line.strip())
        print(rules)
        return rules

    def populateRules(self):
        finalText = " "
        with pdfplumber.open('assets/' + "rules_data_set.pdf") as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
        finalText = self.cleanString(finalText)
        sentences = self.extract_sentences(finalText)
        pos_tagger = self.pos_tagging(sentences)
        return pos_tagger

    def extract_sentences(self, text):
        preProc = PreProcessing()
        sentences = preProc.dot_tokenization(text)
        return sentences

    def pos_tagging(self, sentences):
        list_of_rules = []
        for str in sentences:
            sentences_tag = pos_tag(word_tokenize(str))
            list_of_rules.append(sentences_tag)
        print(list_of_rules)
        tags = [[tag for word, tag in sent] for sent in list_of_rules]
        return tags

    def checkPages(self, filename):
        count = 1
        finalText = " "
        final_method = " "
        count = 0
        upload = False
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(base_dir, 'assets', 'upload', filename)
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                count += 1
            finalText = self.cleanString(finalText)
        return count

    def endorsementExtraction(self, filename):
        finalText = " "
        endorsement = " "
        go = False
        count = 0
        limitPages = 10
        currentPage = 0  # 1 to 10
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(base_dir, 'assets', 'upload', filename)
        with pdfplumber.open(file_path) as pdf:
            # endorsement = pdf.pages[1]
            # print(endorsement.extract_text())
            for page in pdf.pages:
                extractFromPDF = page.extract_text()
                finalText = finalText + extractFromPDF
                print(finalText)
                if (self.endorsementChecker(finalText)):
                    go = True
                    break
                if (currentPage == limitPages):
                    break
                print(currentPage)
                currentPage += 1
                count += 1
        if (go):
            finalText = self.cleanString(finalText)
            endorsement = finalText
        return endorsement

    def endorsementChecker(self, text):
        endorsement = False
        count = 0
        if ((("Endorsement" in text or "ENDORSEMENT" in text) or ("APPROVAL SHEET" in text or "Approval Sheet" in text))
           and ("TABLE OF CONTENTS" not in text or "Table of Contents" not in text)):
            endorsement = True
        else:
            count += 1
        return endorsement

    def acceptanceChecker(self, filename):
        start_time = time.time()
        go = False
        endorsement = " "
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(base_dir, 'assets', 'upload', filename)
        if (self.checkPages(filename) >= 5):
            endorsement = self.endorsementExtraction(filename)
            if ("PASSED" in endorsement):
                go = True
            else:
                os.remove(file_path)
        else:
            os.remove(file_path)
        end_time = time.time()
        execution_time = end_time - start_time
        return go

    # def getIntroduction(self,processedText,page):
