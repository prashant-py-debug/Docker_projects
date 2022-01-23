from fastapi import FastAPI , UploadFile,File,Form
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import io 

app = FastAPI()

pickle_in = open("classifier.pkl" , "rb")

classifier = pickle.load(pickle_in)

@app.post("/")
async def root(file:UploadFile = File(...)):
	df = pd.read_csv(file.file)
	predictions = predict(data = df)
	print(predictions)
	return {'status': "ok"}

	
	
def predict(data):
	predictions = classifier.predict(data)
	return predictions


