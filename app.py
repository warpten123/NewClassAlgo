import json
import time
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
import os
import sys
import requests
import glob
import nltk
from flask import Flask


from tfidf.TFIDF_FINAL import Processing
from knn.cosine import Cosine

from tfidf.extraction_helper import Helper

app = Flask(__name__)







def initializeDataSet():
    tfidf = Processing(" ")
    tfidf.createTFIDF(" ")

def checkDataSet():
    print("checkDataSet")
    cont = False
    csv = os.path.abspath("tfidf/Results/TFIDF.csv")  # Get the absolute path of the target file
    directory = glob.glob("tfidf/Results/*.csv")  # Get all CSV files in the directory

    for file in directory:
        if os.path.abspath(file) == csv:  # Compare absolute paths
            cont = True
            break  # Exit the loop early if the file is found

    return cont



@app.route("/")
def home():
    return "Hello, Flask from VS Code! 3"


@app.route('/upload-file', methods=['POST'])
def upload_file():
    file = request.files['file']
    research_id = request.form['research_id']
    result = {}

    filename = file.filename
    upload_dir = os.path.abspath(os.path.join("/opt/render/project/src/", 'assets', 'upload'))
    print("upload_dir",upload_dir)

    os.makedirs(upload_dir, exist_ok=True)

 
    file_path = os.path.join(upload_dir, filename)
    
    file.save(file_path)

    result["message"] = "File uploaded successfully"
    result["filename"] = filename
    result["research_id"] = research_id
    return jsonify(result)


@app.route('/python/classify/<filename>', methods=['GET'])
def classify(filename):
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    helper = Helper()
    cosine = Cosine()

    appendedData = helper.main_logic(filename)
   
    data = cosine.classifyResearch(appendedData['appendedData'], False)
    
    
    sorted_dict = dict(sorted(data.items(), key=lambda item: item[1]))

    # str = ','.join(newList)
    return sorted_dict




@app.before_request
def before_first_request_func():
    print("before_first_request_func")
    if (checkDataSet() != True):
        initializeDataSet()


if __name__ == "__main__":
    before_first_request_func()
    app.run()