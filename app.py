from flask import Flask, render_template,request
import pickle
import pandas as pd 
import numpy as np 
import copy 
import joblib
import re 

app = Flask(__name__)
model = pickle.load(open("RandomForest_solarpanel",'rb'))
Winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
label = joblib.load('label_encoding')

@app.route('/')
def home():
    result = ''
    return render_template('indexf.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    Ipv = float(request.form['Ipv']) 
    Vpv = float(request.form['Vpv'])
    Vdc = float(request.form['Vdc'])
    ia = float(request.form['ia'])
    ib = float(request.form['ib'])
    ic = float(request.form['ic'])
    va = float(request.form['va'])
    vb = float(request.form['vb'])
    vc = float(request.form['vc'])
    Iabc = float(request.form['Iabc'])
    If = float(request.form['If'])
    Vabc = float(request.form['Vabc'])
    Vf = float(request.form['Vf'])
    result =model.predict([[Ipv,Vpv,Vdc,ia,ib,ic,va,vb,vc,Iabc,If,Vabc,Vf]])[0]
    return render_template('indexf.html',y = str(result))
    
if __name__ == "__main__":
    app.run(debug = True)
