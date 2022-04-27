import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle

pickle_in = open("classifier.pkl", "rb")
clf = pickle.load(pickle_in)

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/') #make tokens after splitting by slash
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-') #make tokens after splitting by dash
        tkns_ByDot = []
        for j in range(0,len(tokens)):
            temp_Tokens = str(tokens[j]).split('.') # make tokens after splitting by dot
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens)) #remove redundant tokens
    if 'com' in total_Tokens:
        total_Tokens.remove('com') #removing .com since it occurs a lot of times and it should not be included in our features
    return total_Tokens

pickle_in_tfidf = open("tfidf.pkl","rb")
tfidf = pickle.load(pickle_in_tfidf)

app = FastAPI()



class Data(BaseModel):
	X : str


@app.get("/")
def index():
	return {'message' : f'Hello Stranger'}


@app.post("/predict")
def preProcessData(data:Data):
    data = data.dict()
    X= data['X']
    X_predict1 = tfidf.transform([X])
    New_predict1 = clf.predict(X_predict1)
    print(New_predict1)
    probability = np.max(clf.predict_proba(X_predict1))*100
    print(probability)
    
    results = {"prediction" : New_predict1[0] , "probability" : probability}
    return results
